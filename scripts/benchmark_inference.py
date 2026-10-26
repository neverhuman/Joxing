"""Registered isolated CPU scalar/batch inference; never game throughput."""

import argparse
import fcntl
import json
import resource
import time
import traceback
from pathlib import Path

import numpy as np

from joxing.native_transfer import load_controller
from joxing.provenance import digest, manifest, write_json
from joxing.stance_train import file_hash

PROTOCOL_SHA = "2386db6977460ed338770839bc66ddb64ecd028286a0794a6d1485a9a17e8040"


def evaluate_batch(model, normalizer, raw, mode):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 3601 or not np.isfinite(raw).all():
        raise ValueError("Malformed batched observations")
    normalized = np.stack([normalizer.apply(row) for row in raw])
    if mode == "scalar":
        results = [model.predict(row, deterministic=True) for row in normalized]
        if any(hidden is not None for _, hidden in results):
            raise RuntimeError("Unexpected actor memory")
        action = np.stack([a for a, _ in results])
    elif mode == "batched":
        action, hidden = model.predict(normalized, deterministic=True)
        if hidden is not None:
            raise RuntimeError("Unexpected actor memory")
    else:
        raise ValueError("Unknown inference mode")
    action = np.asarray(action)
    if (
        action.shape != (len(raw), 700)
        or not np.isfinite(action).all()
        or np.max(abs(action)) > 1
    ):
        raise ValueError("Malformed batched muscle actions")
    return action


def planned_cases(p):
    return [
        {
            "case_id": f"{repeat}:{batch}:{mode}",
            "repeat": repeat,
            "batch": batch,
            "mode": mode,
            "status": "not_run",
        }
        for repeat in range(p["repeats"])
        for batch in (p["batches"] if repeat % 2 == 0 else list(reversed(p["batches"])))
        for mode in p["modes"]
    ]


def run(protocol_path, v8, v7, out):
    p = json.loads(protocol_path.read_text())
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Registered H0 protocol changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".inference-h0.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        out.mkdir(exist_ok=False)
    except Exception:
        lease.close()
        raise
    started = time.perf_counter()
    rows = planned_cases(p)
    report = manifest() | {
        "scope": p["scope"],
        "protocol_sha256": digest(p),
        "script_sha256": file_hash(Path(__file__)),
        "cases": rows,
        "status": "running",
        "physics_steps": 0,
        "boxing_games": 0,
    }
    write_json(out / "result.json", report)
    write_json(out / "protocol.json", p)
    active = None

    def budget():
        if time.perf_counter() - started >= p["caps"]["wall_s"]:
            raise TimeoutError("Registered inference wall cap")
        if (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            >= p["caps"]["raw_bytes"]
        ):
            raise TimeoutError("Registered inference raw cap")

    try:
        import torch

        from joxing.wave import check_resources

        check_resources(out, p["caps"], True)
        torch.set_num_threads(1)
        if file_hash(v8.parent / "audit-linux.json") != p["v8_audit_sha256"]:
            raise ValueError("Prerequisite v8 audit changed")
        audit = json.loads((v8.parent / "audit-linux.json").read_text())
        if not audit["passed"]:
            raise ValueError("Prerequisite v8 audit failed")
        model, n = load_controller(v8 / "inputs")
        model.policy.set_training_mode(False)
        source = []
        obs = []
        for item in p["observation_sources"]:
            path = ({"v8": v8, "v7": v7}[item["run"]]) / item["path"]
            value = json.loads(path.read_text())
            for key in item["keys"]:
                value = value[key]
            obs.append(np.asarray(value, dtype=np.float32))
            source.append(item | {"file_sha256": file_hash(path)})
        obs = np.stack(obs)
        if obs.shape != (8, 3601) or not np.isfinite(obs).all():
            raise ValueError("Unexpected observation bank")
        np.savez_compressed(out / "observations.npz", observations=obs)
        report.update(
            observation_sources=source,
            observation_file_sha256=file_hash(out / "observations.npz"),
            normalizer_stats_hash=n.fingerprint,
            torch=torch.__version__,
            threads=torch.get_num_threads(),
            startup_s=time.perf_counter() - started,
            device="cpu",
        )
        with torch.inference_mode():
            for row in rows:
                active = row
                budget()
                row["status"] = "running"
                write_json(out / "result.json", report)
                raw = obs[np.arange(row["batch"]) % len(obs)]
                expected = evaluate_batch(model, n, raw, "scalar")
                actual = evaluate_batch(model, n, raw, row["mode"])
                error = float(np.max(abs(actual - expected)))
                row["scalar_reference_max_action_error"] = error
                if error > p["action_max_error"]:
                    np.savez_compressed(
                        out / "parity-fault.npz",
                        raw=raw,
                        reference=expected,
                        actual=actual,
                    )
                    raise RuntimeError("Registered scalar/batched action parity failed")
                timings = []
                for i in range(p["warmup_calls"] + p["timed_calls"]):
                    budget()
                    begin = time.perf_counter_ns()
                    output = evaluate_batch(model, n, raw, row["mode"])
                    elapsed = (time.perf_counter_ns() - begin) / 1e6
                    if not np.array_equal(output, actual):
                        raise RuntimeError("Inference self-repeat changed")
                    if i >= p["warmup_calls"]:
                        timings.append(elapsed)
                n.check()
                row.update(
                    status="completed",
                    milliseconds=timings,
                    p50_ms=float(np.median(timings)),
                    p95_ms=float(np.quantile(timings, 0.95)),
                    observations_per_s=row["batch"] / (np.median(timings) / 1000),
                    self_repeat_exact=True,
                )
                write_json(out / "result.json", report)
        n.check()
        report["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- all failed/skipped cases retained
        if active is not None and active["status"] != "completed":
            active.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
            )
        report.update(
            status="halted", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "halted_before_case"
        report.update(
            wall_s=time.perf_counter() - started,
            max_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            rss_units="KiB on Linux, bytes on macOS",
            completed=sum(r["status"] == "completed" for r in rows),
        )
        write_json(out / "result.json", report)
        lease.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("protocol", "v8-run", "v7-run", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.v8_run, args.v7_run, args.out)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "status",
                    "wall_s",
                    "completed",
                    "error",
                    "physics_steps",
                    "boxing_games",
                )
            },
            indent=2,
        )
    )
    if result["status"] != "completed":
        raise SystemExit(1)
