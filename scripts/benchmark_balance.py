"""Bounded 1/2/4-worker feedback workload; includes support, resets and inference."""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from joxing.provenance import manifest, write_json

_env = None
_policy = None


def initialize(checkpoint, barrier):
    import torch
    from stable_baselines3 import SAC

    from joxing.stance_env import StanceEnv

    global _env, _policy
    torch.set_num_threads(1)
    _env = StanceEnv(training=False)
    _policy = SAC.load(checkpoint, device="cpu")
    barrier.wait(timeout=120)


def workload(job):
    index, steps = job
    obs, _ = _env.reset(options={"scenario_id": 30000 + index})
    # Compile/load and one inference are outside the measured interval.
    _policy.predict(obs, deterministic=True)
    physics_before = _env.physics_total
    preparation_before = _env.preparation_total
    started = time.perf_counter()
    resets = 0
    for _ in range(steps):
        action, _ = _policy.predict(obs, deterministic=True)
        obs, _, _, _, _ = _env.step(action)
        if _env.done:
            obs, _ = _env.reset(options={"scenario_id": 30000 + index})
            resets += 1
    return {
        "transitions": steps,
        "physics_substeps": _env.physics_total - physics_before,
        "preparation_substeps": _env.preparation_total - preparation_before,
        "worker_wall_s": time.perf_counter() - started,
        "resets": resets,
        "invalids": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be 1..1000")
    samples = []
    for workers in (1, 2, 4):
        setup_started = time.perf_counter()
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(workers)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize,
            initargs=(str(args.run / "final.zip"), barrier),
            mp_context=context,
        ) as pool:
            # Start every worker and warm each accepted scenario before timing.
            list(pool.map(workload, [(i, 1) for i in range(workers)]))
            setup_s = time.perf_counter() - setup_started
            for repeat in range(3):
                started = time.perf_counter()
                rows = list(
                    pool.map(workload, [(i, args.steps) for i in range(workers)])
                )
                elapsed = time.perf_counter() - started
                transitions = sum(r["transitions"] for r in rows)
                physics = sum(r["physics_substeps"] for r in rows)
                samples.append(
                    {
                        "workers": workers,
                        "repeat": repeat,
                        "wall_s": elapsed,
                        "setup_and_warmup_s": setup_s,
                        "transitions": transitions,
                        "physics_substeps": physics,
                        "transitions_per_s": transitions / elapsed,
                        "physics_substeps_per_s": physics / elapsed,
                        "rows": rows,
                    }
                )
                write_json(
                    args.out, manifest() | {"samples": samples, "complete": False}
                )
    report = manifest() | {
        "profile": "native-gravity-closed-loop-support-and-cached-reset",
        "complete": True,
        "samples": samples,
        "median_transitions_per_s": {
            str(w): float(
                np.median(
                    [s["transitions_per_s"] for s in samples if s["workers"] == w]
                )
            )
            for w in (1, 2, 4)
        },
        "limitations": [
            "optimizer excluded",
            "no game referee",
            "short development workload",
            "cached reset states",
            "no GPU physics",
        ],
    }
    write_json(args.out, report)
    print(report["median_transitions_per_s"])


if __name__ == "__main__":
    main()
