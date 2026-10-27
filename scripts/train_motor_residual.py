"""Run only the source-frozen registered physical motor pilot."""

import argparse
import fcntl
import json
import os
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

from joxing.motor_run import PilotRun
from joxing.provenance import ROOT, digest
from joxing.stance_train import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    registered = json.loads((ROOT / "experiments/v0.0.10/protocol.json").read_text())
    if digest(p) != digest(registered) or p["schema"] != "joxing-motor-pilot-D2-0.0.10":
        raise ValueError("Unregistered pilot protocol")
    for name, sha in p["source_files"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError("Frozen source drift: " + name)
    for name, sha in p["inputs"].items():
        if file_hash(args.inputs / name) != sha:
            raise ValueError("Input drift: " + name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (args.out.parent / ".motor-pilot.lock").open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = PilotRun(p, args.inputs, args.out).run()
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "status",
                    "reason",
                    "formal_optimizer_updates",
                    "main_physics_steps_lower_bound",
                    "audit_physics_steps",
                )
            }
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
