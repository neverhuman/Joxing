"""Profile an existing recorded-action continuation without changing its dynamics."""

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

import numpy as np

from joxing.provenance import manifest, write_json
from joxing.stance_env import StanceEnv
from joxing.stance_train import FaultRecorder, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    record = json.loads(args.record.read_text())
    env = StanceEnv(training=False)
    # FaultRecorder snapshots before each step, as it does in training. A fault
    # writes beside the diagnostic receipt; never into the published source run.
    wrapped = FaultRecorder(env, args.out.parent / (args.out.stem + "-faults"))
    samples = []
    functions = []
    for profiled in (False, True):
        for repeat in range(3):
            env.restore(record["snapshot"])
            initial_physics = env.physics_total
            profiler = cProfile.Profile()
            started = time.perf_counter()
            if profiled:
                profiler.enable()
            for action in record["actions"]:
                wrapped.step(np.asarray(action))
            if profiled:
                profiler.disable()
            elapsed = time.perf_counter() - started
            actual = np.asarray(env.snapshot()["state"])
            reference_error = float(
                np.max(abs(actual - np.asarray(record["final_state"])))
            )
            assert reference_error < 1e-5
            outcome = env.result()
            assert all(
                outcome[key] == record["result"][key]
                for key in (
                    "scenario_id",
                    "physics_steps",
                    "status",
                    "reason",
                    "quiet_success",
                    "recovery_success",
                )
            )
            samples.append(
                {
                    "profiled": profiled,
                    "repeat": repeat,
                    "control_steps": len(record["actions"]),
                    "physics_steps": env.physics_total - initial_physics,
                    "wall_s": elapsed,
                    "control_steps_per_s": len(record["actions"]) / elapsed,
                    "recorded_reference_error": reference_error,
                }
            )
            if profiled:
                for (filename, line, name), stats in pstats.Stats(
                    profiler
                ).stats.items():
                    primitive, calls, own, cumulative, _ = stats
                    # Omit machine-specific checkout paths; keep source identity.
                    filename = filename.split("/JOXING/")[-1]
                    if "/site-packages/" in filename:
                        filename = filename.split("/site-packages/")[-1]
                    functions.append(
                        {
                            "repeat": repeat,
                            "filename": filename,
                            "line": line,
                            "function": name,
                            "primitive_calls": primitive,
                            "calls": calls,
                            "own_wall_s": own,
                            "cumulative_wall_s": cumulative,
                        }
                    )
    result = manifest() | {
        "script_sha256": file_hash(Path(__file__)),
        "record_sha256": file_hash(args.record),
        "scenario_id": record["result"]["scenario_id"],
        "specification": env.specification(),
        "scope": "One preselected recorded-action continuation, three unprofiled and three cProfile repeats on Mac; training FaultRecorder wrapper retained. No policy inference, IPC, reset, optimizer or production two-player game is timed. Profiling adds overhead; cumulative times overlap and must not be summed.",
        "samples": samples,
        "functions": sorted(
            functions, key=lambda row: (row["repeat"], -row["own_wall_s"])
        ),
        "control_physics_steps_executed": sum(row["physics_steps"] for row in samples),
        "preparation_steps_executed": env.preparation_total,
        "total_physics_steps_executed": sum(row["physics_steps"] for row in samples)
        + env.preparation_total,
        "unprofiled_median_control_steps_per_s": float(
            np.median([r["control_steps_per_s"] for r in samples if not r["profiled"]])
        ),
    }
    write_json(args.out, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"functions", "specification"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
