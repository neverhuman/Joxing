"""Execute the committed bounded v0.0.5 protocol, once, in a new directory."""

import argparse
from pathlib import Path

from joxing.wave import run_wave

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_wave(args.protocol, args.out)
