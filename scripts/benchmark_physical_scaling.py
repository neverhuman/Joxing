"""Bounded CPU scaling with serial trajectory-hash anchors; no backend changes."""

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import resource
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from joxing.native_transfer import load_controller, predict
from joxing.physical_motor import PhysicalMotorTask
from joxing.provenance import ROOT, digest, manifest, write_json
from joxing.stance_train import file_hash

PROTOCOL_SHA = "e741ede38d48b8070924aa71850a36456b5adef868466cfc8a8e0b43c6e80953"
START_GATE = None


def initialize(gate):
    global START_GATE
    START_GATE = gate


def schedule(p):
    anchors = [
        {"case_id": f"anchor:{seed}", "seed": seed, "status": "not_run"}
        for seed in p["seeds"]
    ]
    waves = []
    for repeat in range(p["repeats"]):
        for workers in p["workers"] if repeat % 2 == 0 else p["workers"][::-1]:
            wave_id = f"{repeat}:{workers}"
            waves.append(
                {
                    "wave_id": wave_id,
                    "repeat": repeat,
                    "workers": workers,
                    "status": "not_run",
                    "cases": [
                        {
                            "case_id": f"{wave_id}:{seed}",
                            "seed": seed,
                            "status": "not_run",
                        }
                        for seed in p["seeds"][:workers]
                    ],
                }
            )
    return anchors, waves


def worker(inputs, p, rows, progress_path, stop_path, deadline):
    from joxing.wave import canonical_actor_hash

    started = time.perf_counter()
    record = {
        "pid": os.getpid(),
        "phase": "starting",
        "cases": rows,
        "physics_steps": 0,
    }
    task = active = None

    def save():
        record["physics_steps"] = task.physics_total if task is not None else 0
        write_json(progress_path, record)

    def check():
        if stop_path.exists() or time.monotonic() >= deadline:
            raise TimeoutError("Parent stop or registered H2 wall cap")

    try:
        save()
        model, n = load_controller(inputs)
        model.policy.set_training_mode(False)
        actor_hash = canonical_actor_hash(model.actor)
        v8 = json.loads((ROOT / "experiments/v0.0.8/protocol.json").read_text())
        task = PhysicalMotorTask(v8["environment"], v8["variants"][0])
        record.update(
            phase="ready",
            startup_s=time.perf_counter() - started,
            actor_hash=actor_hash,
            normalizer_stats_hash=n.fingerprint,
        )
        save()
        if START_GATE is not None:
            while not START_GATE.wait(0.2):
                check()
        import torch

        record["threads"] = torch.get_num_threads()
        if record["threads"] != 1:
            raise RuntimeError("Worker computational thread limit changed")
        with torch.inference_mode():
            for row in rows:
                active = row
                check()
                row["status"] = "running"
                before = task.physics_total
                reset_start = time.perf_counter()
                task.reset(row["seed"])
                reset_s = time.perf_counter() - reset_start
                resets = [row["seed"]]
                initial_hash = digest(task.snapshot())
                hashes, action_hash = [], hashlib.sha256()
                record["phase"] = "running"
                save()
                loop_start = time.perf_counter()
                try:
                    for i in range(p["control_steps"]):
                        check()
                        if task.done:
                            reset_seed = row["seed"] + 1000000 * len(resets)
                            task.reset(reset_seed)
                            resets.append(reset_seed)
                        snap = task.snapshot()
                        action, clipped = predict(model, n, task.last_obs)
                        _, reward, terminated, truncated, _, telemetry = task.step(
                            action
                        )
                        action_hash.update(
                            np.asarray(action, dtype=np.float32).tobytes()
                        )
                        hashes.append(
                            digest(
                                {
                                    "pre_snapshot": digest(snap),
                                    "action_sha256": hashlib.sha256(
                                        action.tobytes()
                                    ).hexdigest(),
                                    "reward": reward,
                                    "terminated": terminated,
                                    "truncated": truncated,
                                    "telemetry": telemetry,
                                    "clipped": clipped,
                                }
                            )
                        )
                        row.update(
                            controls=i + 1, physics_steps=task.physics_total - before
                        )
                        if (i + 1) % p["progress_controls"] == 0:
                            save()
                finally:
                    row["physics_steps"] = task.physics_total - before
                loop_s = time.perf_counter() - loop_start
                n.check()
                if canonical_actor_hash(model.actor) != actor_hash:
                    raise RuntimeError("Frozen actor tensors changed")
                row.update(
                    status="completed",
                    loop_s=loop_s,
                    control_steps_per_s=p["control_steps"] / loop_s,
                    reset_s=reset_s,
                    resets=resets,
                    initial_hash=initial_hash,
                    final_hash=digest(task.snapshot()),
                    step_hashes=hashes,
                    action_hash=action_hash.hexdigest(),
                )
                save()
        record["phase"] = "completed"
    except Exception as error:  # noqa: BLE001 -- keep startup/partial/future failures
        if active is not None:
            active.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
            )
        record.update(
            phase="failed", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "worker_halted_before_case"
        record.update(
            wall_s=time.perf_counter() - started,
            peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        save()
        if task is not None:
            task.close()
    return record


def fingerprint(row):
    return {
        k: row[k]
        for k in (
            "seed",
            "controls",
            "physics_steps",
            "resets",
            "initial_hash",
            "final_hash",
            "step_hashes",
            "action_hash",
        )
    }


def run(protocol_path, inputs, out):
    from joxing.wave import check_resources

    p = json.loads(protocol_path.read_text())
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Registered H2 protocol changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".physical-scaling.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    deadline = time.monotonic() + p["caps"]["wall_s"]
    anchors, waves = schedule(p)
    report = manifest() | {
        "status": "running",
        "protocol_sha256": digest(p),
        "script_sha256": file_hash(Path(__file__)),
        "scope": p["scope"],
        "anchors": anchors,
        "waves": waves,
        "physics_steps": 0,
        "boxing_games": 0,
    }
    stop = out / "stop.json"
    write_json(out / "protocol.json", p)
    write_json(out / "result.json", report)
    active = None

    def capacity():
        if time.monotonic() >= deadline:
            raise TimeoutError("H2 wall cap")
        if (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            >= p["caps"]["raw_bytes"]
        ):
            raise TimeoutError("H2 raw cap")
        return check_resources(out, p["caps"])

    def batch(cases, workers, name):
        cap = capacity()
        required = (
            p["worker_memory_reservation_bytes"] * workers
            + p["caps"]["minimum_memory_bytes"]
        )
        if (
            cap["available_memory_bytes"] is None
            or cap["available_memory_bytes"] < required
        ):
            raise RuntimeError(f"Cannot reserve {required} bytes for {workers} workers")
        gate = mp.get_context("spawn").Event() if name != "anchors" else None
        launched = time.perf_counter()
        receipts = []
        progress = [out / f"{name}-worker-{i}.json" for i in range(workers)]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize,
            initargs=(gate,),
        ) as pool:
            packets = [cases] if name == "anchors" else [[r] for r in cases]
            futures = [
                pool.submit(worker, inputs, p, packet, path, stop, deadline)
                for packet, path in zip(packets, progress, strict=True)
            ]
            try:
                if gate is not None:
                    while not all(
                        path.exists()
                        and json.loads(path.read_text())["phase"] == "ready"
                        for path in progress
                    ):
                        capacity()
                        if any(f.done() for f in futures):
                            raise RuntimeError("Worker failed before common start gate")
                        wait(futures, timeout=0.2)
                    steady_started = time.perf_counter()
                    gate.set()
                else:
                    steady_started = launched
                pending = set(futures)
                while pending:
                    capacity()
                    done, pending = wait(pending, timeout=0.2)
                    for future in done:
                        record = future.result()
                        receipts.append(record)
                        for result in record["cases"]:
                            next(
                                row
                                for row in cases
                                if row["case_id"] == result["case_id"]
                            ).update(result)
                        if record["phase"] != "completed":
                            raise RuntimeError(record.get("error", "Incomplete worker"))
                steady_s = time.perf_counter() - steady_started
            except Exception as error:
                write_json(stop, {"reason": str(error)})
                if gate is not None:
                    gate.set()
                # Exiting the pool waits only for owned bounded workers. Their private
                # progress files retain partial work even if a future crashes.
                raise
        return {
            "worker_receipts": receipts,
            "startup_inclusive_s": time.perf_counter() - launched,
            "steady_parent_s": steady_s,
            "capacity_before": cap,
        }

    try:
        report["initial_capacity"] = check_resources(out, p["caps"], True)
        cfg = ROOT / "experiments/v0.0.8/protocol.json"
        if file_hash(cfg) != p["v8_protocol_file_sha256"]:
            raise ValueError("Physical configuration changed")
        expected_steps = (
            (len(anchors) + sum(len(w["cases"]) for w in waves))
            * p["control_steps"]
            * 10
        )
        if expected_steps > p["caps"]["physics_steps"]:
            raise ValueError("Plan exceeds physics cap")
        report["maximum_allocated_physics_steps"] = expected_steps
        report["anchor_run"] = batch(anchors, 1, "anchors")
        reference = {r["seed"]: fingerprint(r) for r in anchors}
        declines = 0
        previous_tps = None
        for wave in waves:
            active = wave
            wave["status"] = "running"
            write_json(out / "result.json", report)
            wave.update(
                batch(wave["cases"], wave["workers"], wave["wave_id"].replace(":", "-"))
            )
            for row in wave["cases"]:
                if fingerprint(row) != reference[row["seed"]]:
                    row.update(
                        status="invalid",
                        reason="Serial trajectory-hash anchor mismatch",
                    )
                    raise RuntimeError("CPU scaling changed a frozen trajectory")
                row["serial_anchor_exact"] = True
            controls = len(wave["cases"]) * p["control_steps"]
            wave.update(
                status="completed",
                controls=controls,
                steady_control_steps_per_s=controls / wave["steady_parent_s"],
                startup_inclusive_control_steps_per_s=controls
                / wave["startup_inclusive_s"],
            )
            report["physics_steps"] = sum(
                r.get("physics_steps", 0) for r in anchors
            ) + sum(r.get("physics_steps", 0) for w in waves for r in w["cases"])
            write_json(out / "result.json", report)
            if wave["repeat"] == 0:
                current = wave["steady_control_steps_per_s"]
                declines = (
                    declines + 1 if previous_tps and current < 0.9 * previous_tps else 0
                )
                previous_tps = current
                if declines >= 2:
                    raise RuntimeError(
                        "Two consecutive throughput deteriorations; stop scaling"
                    )
        report["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- no hidden failed/not_run attempts
        if active is not None and active["status"] != "completed":
            active.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
            )
        report.update(
            status="halted", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        # Reconcile every worker's last private receipt, including failed futures.
        accounting_exact = True
        for path in out.glob("*-worker-*.json"):
            value = json.loads(path.read_text())
            accounting_exact &= value["phase"] in {"completed", "failed"}
            for row in value["cases"]:
                target = next(
                    r
                    for r in anchors + [r for w in waves for r in w["cases"]]
                    if r["case_id"] == row["case_id"]
                )
                for key, val in row.items():
                    if key not in target or key not in {"status", "reason"}:
                        target[key] = val
                if target["status"] in {"not_run", "running"}:
                    target.update(status=row["status"], reason=row.get("reason"))
        for row in anchors + [r for w in waves for r in w["cases"]]:
            if row["status"] == "not_run":
                row["reason"] = "halted_before_case"
            if row["status"] == "running":
                row.update(
                    status="incomplete",
                    reason="Worker did not return a final receipt; progress is a lower bound",
                )
        for wave in waves:
            if wave["status"] == "not_run":
                wave["reason"] = "halted_before_wave"
        report.update(
            wall_s=time.perf_counter() - started,
            physics_accounting_exact=accounting_exact,
            physics_accounting_note="Exact worker counters"
            if accounting_exact
            else "Confirmed lower bound; maximum_allocated_physics_steps is the conservative upper bound",
            physics_steps=sum(r.get("physics_steps", 0) for r in anchors)
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
            {k: result.get(k) for k in ("status", "physics_steps", "wall_s", "error")},
            indent=2,
        )
    )
    if result["status"] != "completed":
        raise SystemExit(1)
