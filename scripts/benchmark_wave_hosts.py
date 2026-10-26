"""User-authorized H0: larger CPU pools, fixed policy, no learner claim."""

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from joxing.provenance import manifest, write_json
from joxing.wave import check_resources

_env = None
_actor = None
_barrier = None


def initialize(checkpoint, barrier):
    import torch
    from stable_baselines3.sac.policies import Actor

    from joxing.stance_env import StanceEnv

    global _env, _actor, _barrier
    torch.set_num_threads(1)
    _env = StanceEnv(training=False)
    _actor = Actor.load(checkpoint, device="cpu")
    _barrier = barrier
    barrier.wait(timeout=180)


def workload(job):
    index, steps = job
    _barrier.wait(timeout=180)
    obs, _ = _env.reset(options={"scenario_id": 63000 + index})
    _actor.predict(obs, deterministic=True)
    before = _env.physics_total, _env.preparation_total
    started = time.perf_counter()
    resets = 0
    for _ in range(steps):
        snapshot = _env.snapshot()
        action, _ = _actor.predict(obs, deterministic=True)
        try:
            obs, _, _, _, _ = _env.step(action)
        except Exception as error:  # noqa: BLE001 -- parent persists fault then raises
            return {
                "invalid": True,
                "reason": str(error),
                "last_valid": snapshot,
                "action": action.tolist(),
                "worker": index,
            }
        if _env.done:
            obs, _ = _env.reset(options={"scenario_id": 63000 + index})
            resets += 1
    return {
        "worker": index,
        "scenario_id": 63000 + index,
        "transitions": steps,
        "physics_substeps": _env.physics_total - before[0],
        "preparation_substeps": _env.preparation_total - before[1],
        "wall_s": time.perf_counter() - started,
        "resets": resets,
        "invalid": False,
        "pid": os.getpid(),
        "lifetime_physics_substeps": _env.physics_total,
        "lifetime_preparation_substeps": _env.preparation_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--concurrent-workers", type=int, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.workers)) != len(args.workers) or any(
        w not in {4, 8, 12, 16} or w + args.concurrent_workers > 16
        for w in args.workers
    ):
        parser.error("Aggregate simulation-worker cap is16")
    args.out.mkdir(parents=True, exist_ok=False)
    caps = {"minimum_disk_bytes": 10737418240, "minimum_memory_bytes": 4294967296}
    receipt = manifest() | {
        "experiment": "H0",
        "workers": args.workers,
        "concurrent_workers": args.concurrent_workers,
        "actor_sha256": hashlib.sha256(args.actor.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "addendum_sha256": hashlib.sha256(args.addendum.read_bytes()).hexdigest(),
        "steps_per_worker_repeat": 200,
        "repeats": 3,
        "resource_check": check_resources(args.out, caps, True),
        "load": os.getloadavg(),
        "scope": "rollout/inference/support/reset; excludes optimizer; shared-host timing",
    }
    write_json(args.out / "description.json", receipt)
    samples = []
    warmups = []
    started_wave = time.perf_counter()
    try:
        for workers in args.workers:
            context = multiprocessing.get_context("spawn")
            started = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=initialize,
                initargs=(str(args.actor), context.Barrier(workers)),
                mp_context=context,
            ) as pool:
                warmed = list(pool.map(workload, [(i, 1) for i in range(workers)]))
                warmups.extend(warmed)
                write_json(args.out / "warmups.json", warmups)
                if any(r["invalid"] for r in warmed):
                    raise RuntimeError("H0 warmup fault")
                setup_s = time.perf_counter() - started
                for repeat in range(3):
                    resource_check = check_resources(args.out, caps)
                    started = time.perf_counter()
                    rows = list(pool.map(workload, [(i, 200) for i in range(workers)]))
                    seconds = time.perf_counter() - started
                    sample = {
                        "workers": workers,
                        "repeat": repeat,
                        "setup_s": setup_s,
                        "wall_s": seconds,
                        "rows": rows,
                        "resource_check": resource_check,
                        "load": os.getloadavg(),
                    }
                    samples.append(sample)
                    write_json(args.out / "samples.json", samples)
                    if any(r["invalid"] for r in rows):
                        raise RuntimeError("H0 numerical fault")
                    sample["transitions_per_s"] = (
                        sum(r["transitions"] for r in rows) / seconds
                    )
                    write_json(args.out / "samples.json", samples)
                    print(
                        json.dumps({k: v for k, v in sample.items() if k != "rows"}),
                        flush=True,
                    )
                    if time.perf_counter() - started_wave > 900:
                        raise RuntimeError("H0 900-second cap")
        lifetime = {}
        for row in warmups + [r for x in samples for r in x["rows"]]:
            lifetime[row["pid"]] = row
        write_json(
            args.out / "result.json",
            receipt
            | {
                "complete": True,
                "samples": samples,
                "wall_s": time.perf_counter() - started_wave,
                "median_transitions_per_s": {
                    str(w): float(
                        np.median(
                            [
                                x["transitions_per_s"]
                                for x in samples
                                if x["workers"] == w
                            ]
                        )
                    )
                    for w in args.workers
                },
                "physics_substeps": sum(
                    r["lifetime_physics_substeps"] for r in lifetime.values()
                ),
                "preparation_substeps": sum(
                    r["lifetime_preparation_substeps"] for r in lifetime.values()
                ),
                "invalids": 0,
            },
        )
    except Exception as error:
        write_json(args.out / "fault.json", {"error": str(error), "samples": samples})
        raise


if __name__ == "__main__":
    main()
