"""Offline CPU actor-call latency; excludes physics and is not a real-time proof."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from joxing.provenance import manifest, write_json
from joxing.stance_env import StanceEnv
from joxing.stance_train import file_hash


def main():
    import torch
    from stable_baselines3.sac.policies import Actor

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.samples < 100 or args.warmup < 1:
        raise ValueError("Require >=100 timed calls and >=1 warmup call")
    torch.set_num_threads(1)
    env = StanceEnv(training=False)
    rows = []
    for arm in "ABCD":
        for name, checkpoint in (("initial", "initial"), ("trained", "final")):
            record_path = args.run / f"S-{arm}/{name}-replay.json"
            record = json.loads(record_path.read_text())
            env.restore(record["snapshot"])
            obs = env.observe()
            actor_path = args.run / f"S-{arm}/{checkpoint}-actor.pt"
            actor = Actor.load(actor_path, device="cpu")
            for batch in (1, 4):
                inputs = obs if batch == 1 else np.tile(obs, (batch, 1))
                expected, _ = actor.predict(inputs, deterministic=True)
                for _ in range(args.warmup):
                    actor.predict(inputs, deterministic=True)
                timings = []
                for _ in range(args.samples):
                    started = time.perf_counter_ns()
                    actual, hidden = actor.predict(inputs, deterministic=True)
                    timings.append((time.perf_counter_ns() - started) / 1e6)
                    assert hidden is None and np.array_equal(actual, expected)
                rows.append(
                    {
                        "arm": arm,
                        "policy": name,
                        "batch": batch,
                        "actor_sha256": file_hash(actor_path),
                        "snapshot_sha256": file_hash(record_path),
                        "milliseconds": timings,
                        "percentiles_ms": dict(
                            zip(
                                ("p50", "p95", "p99", "max"),
                                np.quantile(timings, [0.5, 0.95, 0.99, 1]).tolist(),
                            )
                        ),
                        "calls_exceeding_10ms": int(sum(t > 10 for t in timings)),
                    }
                )
    receipt = manifest() | {
        "script_sha256": file_hash(Path(__file__)),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
        "warmup_calls_per_case": args.warmup,
        "timed_calls_per_case": args.samples,
        "device": "cpu",
        "control_physics_steps_executed": env.physics_total,
        "preparation_steps_executed": env.preparation_total,
        "total_physics_steps_executed": env.physics_total + env.preparation_total,
        "scope": "Isolated actor.predict call on one restored recorded observation, including input conversion and output conversion; wall-clock samples on a shared desktop. No optimizer, observation construction, action transport, contact, physics, scheduling guarantee or two-player game.",
        "limitations": [
            "Repeated fixed inputs do not cover all future policies, observations or host load.",
            "Percentiles and observed misses are measurements, not hard deadline guarantees.",
            "Batch-four latency is one call for four observations, not four sequential calls.",
        ],
        "samples": rows,
    }
    write_json(args.out, receipt)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "arm",
                        "policy",
                        "batch",
                        "percentiles_ms",
                        "calls_exceeding_10ms",
                    )
                }
                for r in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
