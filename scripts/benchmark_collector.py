"""Registered exact collector comparison; all physics and actor math unchanged."""

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import resource
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from joxing.native_transfer import load_controller, predict
from joxing.physical_fast import AllocationReducedTask
from joxing.physical_motor import INTEGRATION, PhysicalMotorTask
from joxing.provenance import ROOT, digest, manifest, write_json
from joxing.stance_train import file_hash

PROTOCOL_SHA = "b4ad191f93a6645557ebeb23e8a5cd80e643370195cc0a8423de4a150cc41373"
START_GATE = None
TASKS = {"baseline": PhysicalMotorTask, "allocation_reduced": AllocationReducedTask}


def initialize(gate):
    global START_GATE
    START_GATE = gate


def schedule(p):
    return [
        {
            "repeat": repeat,
            "arm": arm,
            "status": "not_run",
            "cases": [
                {
                    "case_id": f"{repeat}:{arm}:{seed}",
                    "seed": seed,
                    "arm": arm,
                    "status": "not_run",
                }
                for seed in p["seeds"]
            ],
        }
        for repeat in range(p["repeats"])
        for arm in (p["arms"] if repeat % 2 == 0 else p["arms"][::-1])
    ]


def identity(row):
    return {
        key: row[key]
        for key in (
            "seed",
            "controls",
            "physics_steps",
            "resets",
            "initial_hash",
            "final_hash",
            "pre_snapshot_hashes",
            "arrays_sha256",
            "substeps_sha256",
            "trace_sha256",
        )
    }


def save_fault(out, task, reason, last_valid):
    """NaN engine state belongs in NPZ; a strict JSON writer must not mask it."""
    state = np.empty(mujoco.mj_stateSize(task.model, INTEGRATION))
    mujoco.mj_getState(task.model, task.data, state, INTEGRATION)
    np.savez_compressed(
        out / "fault-arrays.npz",
        state=state,
        observation=np.asarray(task.last_obs)
        if task.last_obs is not None
        else np.empty(0),
        **{
            key: np.asarray(getattr(task.env, key, []))
            for key in ("qpos_ref", "xpos_ref", "qpos_ref_future")
        },
    )
    try:
        digest(last_valid)
    except (ValueError, TypeError):
        last_valid = None
    write_json(
        out / "fault.json",
        {
            "reason": str(reason),
            "last_valid_snapshot": last_valid,
            "current_engine_arrays": "fault-arrays.npz",
        },
    )


def worker(inputs, p, row, out, stop, deadline):
    from joxing.wave import canonical_actor_hash

    started = time.perf_counter()
    task = snapshot = None
    record = {"pid": os.getpid(), "phase": "starting", "case": row}
    out.mkdir(exist_ok=False)

    def save():
        row["physics_steps"] = task.physics_total if task is not None else 0
        write_json(out / "progress.json", record)

    def check():
        if stop.exists() or time.monotonic() >= deadline:
            raise TimeoutError("H3 parent stop or wall cap")

    try:
        save()
        if row["arm"] not in TASKS:
            raise ValueError("Unknown collector implementation")
        model, n = load_controller(inputs)
        model.policy.set_training_mode(False)
        actor_hash = canonical_actor_hash(model.actor)
        v8 = json.loads((ROOT / "experiments/v0.0.8/protocol.json").read_text())
        task = TASKS[row["arm"]](v8["environment"], v8["variants"][0])
        record.update(
            phase="ready",
            startup_s=time.perf_counter() - started,
            actor_hash=actor_hash,
            normalizer_stats_hash=n.fingerprint,
        )
        save()
        while not START_GATE.wait(0.2):
            check()
        import torch

        if torch.get_num_threads() != 1:
            raise RuntimeError("H3 computational thread cap")
        record["threads"] = torch.get_num_threads()
        task.reset(row["seed"])
        initial = task.snapshot()
        arrays = {key: [] for key in ("actions", "observations", "states")}
        resets = [{"control_index": 0, "seed": row["seed"]}]
        trace, hashes, evidence = [], [], []
        record["phase"] = "running"
        row["status"] = "running"
        save()
        loop_start = time.perf_counter()
        with torch.inference_mode():
            for i in range(p["control_steps"]):
                check()
                if task.done:
                    reset_seed = row["seed"] + 1000000 * len(resets)
                    task.reset(reset_seed)
                    resets.append({"control_index": i, "seed": reset_seed})
                snapshot = task.snapshot()
                action, clipped = predict(model, n, task.last_obs)
                _, reward, terminated, truncated, _, telemetry = task.step(action)
                arrays["actions"].append(action.copy())
                arrays["observations"].append(
                    np.asarray(snapshot["last_raw_observation"], dtype=np.float32)
                )
                arrays["states"].append(snapshot["state"])
                hashes.append(digest(snapshot))
                trace.append(
                    telemetry
                    | {
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "clipped_fraction": clipped,
                    }
                )
                evidence.extend(task.interval_evidence)
                row["controls"] = i + 1
                if (i + 1) % p["progress_controls"] == 0:
                    save()
        loop_s = time.perf_counter() - loop_start
        final = task.snapshot()
        n.check()
        if canonical_actor_hash(model.actor) != actor_hash:
            raise RuntimeError("H3 actor mutated")
        write_json(out / "initial.json", initial)
        write_json(out / "final.json", final)
        write_json(out / "trace.json", trace)
        np.savez_compressed(
            out / "trajectory.npz", **{k: np.asarray(v) for k, v in arrays.items()}
        )
        PhysicalMotorTask.dump_substeps(
            SimpleNamespace(evidence=evidence), out / "substeps.npz"
        )
        row.update(
            status="completed",
            loop_s=loop_s,
            control_steps_per_s=p["control_steps"] / loop_s,
            resets=resets,
            initial_hash=digest(initial),
            final_hash=digest(final),
            pre_snapshot_hashes=hashes,
            arrays_sha256=file_hash(out / "trajectory.npz"),
            substeps_sha256=file_hash(out / "substeps.npz"),
            trace_sha256=file_hash(out / "trace.json"),
            substeps=len(evidence),
        )
        record["phase"] = "completed"
    except Exception as error:  # noqa: BLE001 -- retain all faults/partial physics
        row.update(
            status="incomplete" if isinstance(error, TimeoutError) else "invalid",
            reason=str(error),
        )
        record.update(
            phase="failed", error=str(error), traceback=traceback.format_exc()
        )
        if task is not None:
            save_fault(out, task, error, snapshot)
    finally:
        record.update(
            wall_s=time.perf_counter() - started,
            peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        save()
        if task is not None:
            task.close()
    return record


def qualify(p, inputs, out, deadline, rows):
    """Candidate must replay all source v8 native contacts/actions/state exactly."""
    from joxing.wave import canonical_actor_hash, check_resources

    out.mkdir()
    model, n = load_controller(inputs)
    model.policy.set_training_mode(False)
    actor_hash = canonical_actor_hash(model.actor)
    cfg = json.loads((ROOT / "experiments/v0.0.8/protocol.json").read_text())
    base = ROOT / "artifacts/v0.0.8/physical/native_2ms"
    rec = json.loads((base / "replay.json").read_text())
    with np.load(base / "actions-74000.npz", allow_pickle=False) as data:
        actions = data["actions"].copy()
    task = AllocationReducedTask(cfg["environment"], cfg["variants"][0])

    def check():
        if (
            time.monotonic() >= deadline
            or task.physics_total + 10 > p["qualification_caps"]["physics_steps"]
        ):
            raise TimeoutError("H3 source-replay cap")
        if (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            >= p["qualification_caps"]["raw_bytes"]
        ):
            raise TimeoutError("H3 source-replay raw cap")
        check_resources(out, p["caps"])

    try:
        if task.contract != json.loads((base / "specification.json").read_text()):
            raise RuntimeError("H3 compiled model contract changed")
        for row in rows:
            row["status"] = "running"
            snap = (
                rec["snapshot"]
                if row["snapshot"] == "boundary"
                else rec["mid_snapshot"]
            )
            last_valid = snap
            task.restore(snap)
            before = task.physics_total
            first = (
                snap["physical_control"]["completed_controls"] * 10
                + snap["physical_control"]["substep"]
            )
            index = rec["continuation_start"]
            try:
                if task.pending:
                    check()
                    if not np.array_equal(task.action, actions[index]):
                        raise RuntimeError("H3 held source action differs")
                    task.resume_control()
                    index += 1
                while not task.done:
                    check()
                    last_valid = task.snapshot()
                    command = (
                        actions[index]
                        if row["path"] == "recorded_actions"
                        else predict(model, n, task.last_obs)[0]
                    )
                    if not np.array_equal(command, actions[index]):
                        raise RuntimeError("H3 source policy command differs")
                    task.step(command)
                    index += 1
                if index != len(actions) or digest(task.snapshot()) != digest(
                    rec["final_snapshot"]
                ):
                    raise RuntimeError(
                        "H3 source engine/cache/application snapshot differs"
                    )
                stem = row["snapshot"] + "-" + row["path"]
                write_json(out / (stem + "-final.json"), task.snapshot())
                task.dump_substeps(out / (stem + ".npz"))
                with (
                    np.load(base / "substeps-74000.npz", allow_pickle=False) as a,
                    np.load(out / (stem + ".npz"), allow_pickle=False) as b,
                ):
                    offset = int(a["contact_offsets"][first])
                    for key in a.files:
                        expected = (
                            a[key][first:] - offset
                            if key == "contact_offsets"
                            else a[key][first:]
                            if key in {"intervals_s", "loads_N", "tilt_drop"}
                            else a[key][offset:]
                        )
                        if expected.dtype != b[key].dtype or not np.array_equal(
                            expected, b[key]
                        ):
                            raise RuntimeError(
                                "H3 source contact/posture differs: " + key
                            )
                n.check()
                if canonical_actor_hash(model.actor) != actor_hash:
                    raise RuntimeError("H3 qualification actor changed")
                row.update(
                    status="completed",
                    first_source_substep=first,
                    source_state_contact_action_exact=True,
                )
            except Exception as error:
                row.update(
                    status="incomplete"
                    if isinstance(error, TimeoutError)
                    else "invalid",
                    reason=str(error),
                )
                save_fault(out, task, error, last_valid)
                raise
            finally:
                row["physics_steps"] = task.physics_total - before
                write_json(
                    out / "result.json",
                    {"cases": rows, "physics_steps": task.physics_total},
                )
    except Exception as error:
        for row in rows:
            if row["status"] == "running":
                row.update(
                    status="incomplete"
                    if isinstance(error, TimeoutError)
                    else "invalid",
                    reason=str(error),
                    physics_steps=0,
                )
        raise
    finally:
        task.close()


def run(protocol_path, inputs, out):
    from joxing.wave import check_resources

    p = json.loads(protocol_path.read_text())
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Registered H3 protocol changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".collector-h3.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    deadline = time.monotonic() + p["caps"]["wall_s"]
    waves = schedule(p)
    qualification = [
        {"snapshot": point, "path": mode, "status": "not_run"}
        for point in ("boundary", "mid_control")
        for mode in ("recorded_actions", "closed_loop")
    ]
    report = manifest() | {
        "status": "running",
        "protocol_sha256": digest(p),
        "script_sha256": file_hash(Path(__file__)),
        "qualification": qualification,
        "waves": waves,
        "physics_steps": 0,
        "boxing_games": 0,
        "maximum_allocated_physics_steps": p["planned_physics_steps"],
    }
    references = {}
    stop = out / "stop.json"
    active = None
    write_json(out / "protocol.json", p)
    write_json(out / "result.json", report)

    def capacity():
        if time.monotonic() >= deadline:
            raise TimeoutError("H3 global wall cap")
        if (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            >= p["caps"]["raw_bytes"]
        ):
            raise TimeoutError("H3 raw cap")
        return check_resources(out, p["caps"])

    try:
        import torch

        report["initial_capacity"] = check_resources(out, p["caps"], True)
        for path, expected in p["source_files"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError("H3 source/reference changed: " + path)
        if (
            p["planned_physics_steps"] > p["caps"]["physics_steps"]
            or len(p["seeds"]) != p["caps"]["simulation_workers"]
        ):
            raise ValueError("H3 allocation exceeds cap")
        with torch.inference_mode():
            qualify(
                p,
                inputs,
                out / "qualification",
                min(deadline, time.monotonic() + p["qualification_caps"]["wall_s"]),
                qualification,
            )
        for wave in waves:
            active = wave
            wave["status"] = "running"
            wave["capacity_before"] = capacity()
            required = (
                p["caps"]["simulation_workers"] * p["worker_memory_reservation_bytes"]
                + p["caps"]["minimum_memory_bytes"]
            )
            if wave["capacity_before"]["available_memory_bytes"] < required:
                raise RuntimeError("H3 worker memory reservation")
            write_json(out / "result.json", report)
            gate = mp.get_context("spawn").Event()
            paths = [
                out / f"{wave['repeat']}-{wave['arm']}-{row['seed']}"
                for row in wave["cases"]
            ]
            launched = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=len(paths),
                mp_context=mp.get_context("spawn"),
                initializer=initialize,
                initargs=(gate,),
            ) as pool:
                futures = [
                    pool.submit(worker, inputs, p, row, path, stop, deadline)
                    for row, path in zip(wave["cases"], paths, strict=True)
                ]
                try:
                    while not all(
                        (path / "progress.json").exists()
                        and json.loads((path / "progress.json").read_text())["phase"]
                        == "ready"
                        for path in paths
                    ):
                        capacity()
                        if any(f.done() for f in futures):
                            raise RuntimeError("H3 worker failed before common start")
                        wait(futures, timeout=0.2)
                    steady = time.perf_counter()
                    gate.set()
                    pending, records = set(futures), []
                    while pending:
                        capacity()
                        done, pending = wait(pending, timeout=0.2)
                        for future in done:
                            record = future.result()
                            records.append(record)
                            row = next(
                                r
                                for r in wave["cases"]
                                if r["case_id"] == record["case"]["case_id"]
                            )
                            row.update(record["case"])
                            if record["phase"] != "completed":
                                raise RuntimeError(
                                    record.get("error", "H3 partial worker")
                                )
                    wave["steady_parent_s"] = time.perf_counter() - steady
                except Exception as error:
                    write_json(stop, {"reason": str(error)})
                    gate.set()
                    raise
            wave["startup_inclusive_s"] = time.perf_counter() - launched
            for row in wave["cases"]:
                reference = references.setdefault(row["seed"], identity(row))
                if reference != identity(row):
                    row.update(
                        status="invalid",
                        reason="H3 complete trajectory/evidence differs from baseline",
                    )
                    raise RuntimeError(row["reason"])
                row["paired_exact"] = True
            controls = len(paths) * p["control_steps"]
            wave.update(
                status="completed",
                worker_receipts=records,
                steady_control_steps_per_s=controls / wave["steady_parent_s"],
                startup_inclusive_control_steps_per_s=controls
                / wave["startup_inclusive_s"],
            )
            report["physics_steps"] = sum(
                r.get("physics_steps", 0) for r in qualification
            ) + sum(r.get("physics_steps", 0) for w in waves for r in w["cases"])
            write_json(out / "result.json", report)
        medians = {
            arm: float(
                np.median(
                    [w["steady_control_steps_per_s"] for w in waves if w["arm"] == arm]
                )
            )
            for arm in p["arms"]
        }
        startup = {
            arm: float(
                np.median(
                    [
                        w["startup_inclusive_control_steps_per_s"]
                        for w in waves
                        if w["arm"] == arm
                    ]
                )
            )
            for arm in p["arms"]
        }
        ratio = medians["allocation_reduced"] / medians["baseline"]
        report.update(
            status="completed",
            median_steady_control_steps_per_s=medians,
            median_startup_inclusive_control_steps_per_s=startup,
            steady_speedup=ratio,
            performance_gate_met=ratio >= p["criteria"]["minimum_steady_speedup"]
            and startup["allocation_reduced"]
            >= p["criteria"]["minimum_startup_ratio"] * startup["baseline"],
        )
    except Exception as error:  # noqa: BLE001 -- preserve qualification/paired/performance failures
        if active is not None and active["status"] != "completed":
            active.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
            )
        report.update(
            status="halted", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        exact = True
        for path in out.glob("*/progress.json"):
            record = json.loads(path.read_text())
            exact &= record["phase"] in {"completed", "failed"}
            target = next(
                r
                for w in waves
                for r in w["cases"]
                if r["case_id"] == record["case"]["case_id"]
            )
            previous_status, reason = target["status"], target.get("reason")
            target.update(record["case"])
            if previous_status == "invalid":
                target.update(status=previous_status, reason=reason)
        for row in qualification + [r for w in waves for r in w["cases"]]:
            if row["status"] == "not_run":
                row["reason"] = "halted_before_case"
            elif row["status"] == "running":
                row.update(
                    status="incomplete",
                    reason="No final worker receipt; physics is a lower bound",
                )
                exact = False
        for wave in waves:
            if wave["status"] == "not_run":
                wave["reason"] = "halted_before_wave"
        report.update(
            wall_s=time.perf_counter() - started,
            physics_accounting_exact=exact,
            physics_steps=sum(r.get("physics_steps", 0) for r in qualification)
            + sum(r.get("physics_steps", 0) for w in waves for r in w["cases"]),
        )
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
            {
                k: result.get(k)
                for k in (
                    "status",
                    "physics_steps",
                    "wall_s",
                    "steady_speedup",
                    "performance_gate_met",
                    "error",
                )
            },
            indent=2,
        )
    )
    if result["status"] != "completed":
        raise SystemExit(1)
