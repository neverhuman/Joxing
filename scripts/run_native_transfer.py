"""Execute the preregistered bounded v7 factorial, retaining every attempted row."""

import argparse
import json
from pathlib import Path

from joxing.native_transfer import run_transfer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = run_transfer(args.protocol, args.archive, args.out)
    print(json.dumps(result, indent=2))
    if (
        not result["all_attempts_completed"]
        or not result["source_unchanged"]
        or not result["inputs_unchanged"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
