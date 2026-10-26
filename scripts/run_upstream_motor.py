"""Run the bounded v0.0.6 protocol with the hash-pinned upstream archive."""

import argparse
import json
from pathlib import Path

from joxing.upstream_motor import run_qualification

parser = argparse.ArgumentParser()
parser.add_argument("--protocol", type=Path, required=True)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
print(json.dumps(run_qualification(args.protocol, args.archive, args.out), indent=2))
