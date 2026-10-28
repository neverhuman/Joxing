"""Launch the frozen prerequisite chain; never duplicate an active wave."""

import argparse
from pathlib import Path

from joxing.native_strike_wave import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--native-protocol-sha256", required=True)
    args = parser.parse_args()
    result = run(
        Path.cwd(), args.source_sha, args.protocol_sha256, args.native_protocol_sha256
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
