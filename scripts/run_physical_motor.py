"""Execute the registered physical diagnostic; thresholds do not determine exit."""

import argparse
import json
from pathlib import Path

from joxing.physical_motor import run_physical

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_physical(args.protocol, args.archive, args.out)
    print(json.dumps(result, indent=2))
    if not all(
        result[k]
        for k in ("all_attempts_completed", "source_unchanged", "inputs_unchanged")
    ):
        raise SystemExit(1)
