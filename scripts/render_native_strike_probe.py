"""Render only the first fixed independently audited native interaction."""

import argparse
from pathlib import Path

from joxing.native_strike_display import render
from joxing.native_strike_qualification import packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = packet(root, args.protocol_sha256, args.source_sha)
    render(root, p, args.source_sha, args.protocol_sha256)


if __name__ == "__main__":
    main()
