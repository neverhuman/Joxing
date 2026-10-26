"""Bounded native scalar-loop attribution; no inference/physics backend change."""

import argparse
import cProfile
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import pstats
import resource
import time
import traceback
from pathlib import Path

import numpy as np

from joxing.native_transfer import load_controller, predict
from joxing.physical_motor import PhysicalMotorTask
from joxing.provenance import ROOT, digest, manifest, write_json
from joxing.stance_train import file_hash

PROTOCOL_SHA = "2607c2dc95358427dca566f79dd4a1920661388a556dca0996125731f256c1d3"


def planned_cases(protocol):
    return [
        {"case_id": f"{seed}:{mode}", "seed": seed, "mode": mode, "status": "not_run"}
        for i, seed in enumerate(protocol["seeds"])
        for mode in (protocol["modes"] if i % 2 == 0 else protocol["modes"][::-1])
    ]


def profile_rows(profiler):
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(
        profiler
    ).stats.items():
        if filename.startswith(str(ROOT)):
            filename = str(Path(filename).relative_to(ROOT))
        elif "/site-packages/" in filename:
            filename = filename.split("/site-packages/")[-1]
        rows.append(
            {
                "filename": filename,
                "line": line,
                "function": name,
                "primitive_calls": primitive,
                "calls": calls,
                "own_s": own,
                "cumulative_s": cumulative,
            }
        )
    return sorted(rows, key=lambda r: -r["own_s"])


def echo_worker(connection):
    """A diagnostic Pipe transfer only; not a delivered actor service."""
    try:
        while True:
            value = connection.recv_bytes()
            if not value:
                break
            connection.send_bytes(hashlib.sha256(value).digest())
    finally:
        connection.close()


def memory():
    p = Path("/proc/self/status")
    if p.exists():
        fields = {
            line.split(":")[0]: line.split()[1:]
            for line in p.read_text().splitlines()
            if ":" in line
        }
        return {key: int(fields[key][0]) * 1024 for key in ("VmRSS", "VmHWM")}
    return {"peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run(protocol_path, inputs, out):
    p = json.loads(protocol_path.read_text())
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Registered H1 protocol changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".physical-profiler.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    rows = planned_cases(p)
    report = manifest() | {
        "status": "running",
        "scope": p["scope"],
        "cases": rows,
        "protocol_sha256": digest(p),
        "script_sha256": file_hash(Path(__file__)),
        "boxing_games": 0,
        "physics_steps": 0,
        "memory_stages": {"before_actor": memory()},
        "profile_warning": "cProfile adds overhead; cumulative timings overlap and must not be summed.",
        "ipc_scope": "Supplementary serialized-snapshot Pipe request/SHA reply, not an integrated rollout pool. Excluded from scalar loop TPS.",
    }
    task = child = parent = connection = active = snapshot = None
    references = {}
    write_json(out / "protocol.json", p)
    write_json(out / "result.json", report)

    def budget():
        from joxing.wave import check_resources

        if time.perf_counter() - started >= p["caps"]["wall_s"]:
            raise TimeoutError("H1 wall cap")
        if (
            task is not None
            and task.physics_total + task.env.frame_skip > p["caps"]["physics_steps"]
        ):
            raise TimeoutError("H1 physics cap")
        if (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            >= p["caps"]["raw_bytes"]
        ):
            raise TimeoutError("H1 raw cap")
        return check_resources(out, p["caps"])

    try:
        import torch

        from joxing.wave import canonical_actor_hash, check_resources

        report["initial_capacity"] = check_resources(out, p["caps"], True)
        config_path = ROOT / "experiments/v0.0.8/protocol.json"
        if file_hash(config_path) != p["v8_protocol_file_sha256"]:
            raise ValueError("v8 physical configuration changed")
        config = json.loads(config_path.read_text())
        model, n = load_controller(inputs)
        model.policy.set_training_mode(False)
        actor_hash = canonical_actor_hash(model.actor)
        report["memory_stages"]["actor_loaded"] = memory()
        report.update(
            actor_hash=actor_hash,
            normalizer_stats_hash=n.fingerprint,
            torch=torch.__version__,
            threads=torch.get_num_threads(),
            model_sha256=file_hash(inputs / "model.zip"),
            normalizer_sha256=file_hash(inputs / "normalizer.pkl"),
        )
        task = PhysicalMotorTask(config["environment"], config["variants"][0])
        report["specification"] = task.specification()
        report["memory_stages"]["physical_task_loaded"] = memory()
        report["construction_s"] = time.perf_counter() - started
        context = mp.get_context("spawn")
        parent, connection = context.Pipe()
        child = context.Process(target=echo_worker, args=(connection,))
        child.start()
        connection.close()
        report["ipc_helper_pid"] = child.pid
        with torch.inference_mode():
            for row in rows:
                active = row
                budget()
                row["status"] = "running"
                write_json(out / "result.json", report)
                physics_before = task.physics_total
                reset_start = time.perf_counter()
                task.reset(row["seed"])
                row["reset_s"] = time.perf_counter() - reset_start
                resets = [row["seed"]]
                initial = task.snapshot()
                tape, observations, states, telemetry = [], [], [], []
                timings = {
                    k: []
                    for k in (
                        "snapshot",
                        "normalize_and_scalar_predict",
                        "physical_step_including_telemetry",
                    )
                }
                profiler = cProfile.Profile()
                if row["mode"] == "profiled":
                    profiler.enable()
                loop_start = time.perf_counter()
                try:
                    for i in range(p["control_steps"]):
                        budget()
                        if task.done:
                            seed = row["seed"] + 1000000 * len(resets)
                            task.reset(seed)
                            resets.append(seed)
                        begin = time.perf_counter()
                        snapshot = task.snapshot()
                        timings["snapshot"].append(time.perf_counter() - begin)
                        begin = time.perf_counter()
                        action, clipped = predict(model, n, task.last_obs)
                        timings["normalize_and_scalar_predict"].append(
                            time.perf_counter() - begin
                        )
                        begin = time.perf_counter()
                        _, reward, terminated, truncated, _, info = task.step(action)
                        timings["physical_step_including_telemetry"].append(
                            time.perf_counter() - begin
                        )
                        states.append(snapshot["state"])
                        observations.append(
                            np.asarray(
                                snapshot["last_raw_observation"], dtype=np.float32
                            )
                        )
                        tape.append(action.copy())
                        telemetry.append(
                            info
                            | {
                                "clipped_fraction": clipped,
                                "reward": reward,
                                "terminated": terminated,
                                "truncated": truncated,
                            }
                        )
                finally:
                    profiler.disable()
                    row["physics_steps"] = task.physics_total - physics_before
                    report["physics_steps"] = task.physics_total
                loop_s = time.perf_counter() - loop_start
                final = task.snapshot()
                parity = {
                    "initial": digest(initial),
                    "final": digest(final),
                    "resets": resets,
                    "telemetry": digest(telemetry),
                }
                arrays = {
                    "actions": np.stack(tape),
                    "observations": np.stack(observations),
                    "states": np.asarray(states),
                }
                # Compare every action/input/integration state and all monitor telemetry,
                # not only a final pose; source core remains the frozen v8 backend.
                if row["seed"] in references:
                    previous, previous_arrays = references.pop(row["seed"])
                    if parity != previous or any(
                        not np.array_equal(a, previous_arrays[k])
                        for k, a in arrays.items()
                    ):
                        raise RuntimeError(
                            "Profiled/unprofiled physics/features/actions/telemetry differ"
                        )
                    row["paired_cpu_exact"] = True
                else:
                    references[row["seed"]] = (parity, arrays)
                case_out = out / row["case_id"].replace(":", "-")
                case_out.mkdir()
                begin = time.perf_counter()
                payload = json.dumps(
                    {"initial": initial, "final": final, "telemetry": telemetry},
                    allow_nan=False,
                ).encode()
                serialization_s = time.perf_counter() - begin
                begin = time.perf_counter()
                with (case_out / "trace.json").open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                disk_s = time.perf_counter() - begin
                begin = time.perf_counter()
                np.savez_compressed(case_out / "trajectory.npz", **arrays)
                compressed_array_write_s = time.perf_counter() - begin
                ipc_times = []
                for _ in range(p["ipc_probe_rounds"]):
                    begin = time.perf_counter()
                    parent.send_bytes(payload)
                    if (
                        not parent.poll(5)
                        or parent.recv_bytes() != hashlib.sha256(payload).digest()
                    ):
                        raise RuntimeError("IPC diagnostic identity/timeout fault")
                    ipc_times.append(time.perf_counter() - begin)
                n.check()
                if canonical_actor_hash(model.actor) != actor_hash:
                    raise RuntimeError("Actor changed")
                row.update(
                    status="completed",
                    controls=len(tape),
                    resets=resets,
                    loop_s=loop_s,
                    control_steps_per_s=len(tape) / loop_s,
                    stage_seconds=timings,
                    parity=parity,
                    serialization_s=serialization_s,
                    disk_fsync_s=disk_s,
                    compressed_array_write_s=compressed_array_write_s,
                    serialized_bytes=len(payload),
                    ipc_probe_roundtrip_s=ipc_times,
                    memory=memory(),
                    functions=profile_rows(profiler)
                    if row["mode"] == "profiled"
                    else [],
                    arrays_sha256=file_hash(case_out / "trajectory.npz"),
                    trace_sha256=file_hash(case_out / "trace.json"),
                )
                report["last_capacity"] = budget()
                write_json(out / "result.json", report)
        if references:
            raise RuntimeError("Missing paired profiling cases")
        report["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- retain all attempted/future cases
        if active is not None:
            active.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
            )
        report.update(
            status="halted", error=str(error), traceback=traceback.format_exc()
        )
        if task is not None:
            write_json(
                out / "fault.json",
                {
                    "reason": str(error),
                    "last_valid_snapshot": snapshot,
                    "attempted_physics_steps": task.physics_total,
                },
            )
    finally:
        if task is not None:
            report["physics_steps"] = task.physics_total
            task.close()
        if child is not None:
            if child.is_alive():
                try:
                    parent.send_bytes(b"")
                except (OSError, BrokenPipeError):
                    pass
                child.join(5)
            if child.is_alive():
                child.terminate()
                child.join(5)
            report["ipc_helper_exitcode"] = child.exitcode
        if parent is not None:
            parent.close()
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "halted_before_case"
        report.update(wall_s=time.perf_counter() - started, final_memory=memory())
        write_json(out / "result.json", report)
        lease.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("protocol", "inputs", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.inputs, args.out)
    print(
        json.dumps(
            {k: result.get(k) for k in ("status", "physics_steps", "wall_s", "error")},
            indent=2,
        )
    )
    if result["status"] != "completed":
        raise SystemExit(1)
