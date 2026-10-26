"""Run the separately preregistered, explicitly authorized v0.0.5 extension."""

import argparse
from pathlib import Path

from joxing.wave_extension import run_extension

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_extension(args.protocol, args.original, args.out)
