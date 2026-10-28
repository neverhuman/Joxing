"""Frozen scoring diagnostic entry point; not competitive training."""

import argparse
from pathlib import Path

from joxing.score_qualification import RUN, packet, qualify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = packet(root, args.protocol_sha256, args.source_sha)
    out = root / RUN
    result = qualify(p, root, out, args.source_sha, args.protocol_sha256)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
