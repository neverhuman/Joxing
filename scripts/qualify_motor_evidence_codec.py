"""Execute the frozen zero-physics corpus only after the final Linux source gate."""

import argparse
import json
import os
import signal
from pathlib import Path

from joxing_process_ownership import belongs_to_supervisor

from joxing.motor_codec_qualification import RUN, qualify, read_packet, require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = read_packet(root, args.protocol_sha256, args.source_sha)
    gate = json.loads(
        (root / "runs/v0.0.11/motor-codec-supervision-01/result.json").read_text()
    )
    require(
        gate["status"] == "running"
        and gate["source_commit"] == args.source_sha
        and gate["protocol_sha256"] == args.protocol_sha256,
        "Codec supervisor provenance",
    )
    tests = next(s for s in gate["stages"] if s["stage"] == "tests")
    require(
        tests["status"] == "completed"
        and tests["returncode"] == 0
        and not tests["live_recorded_processes"],
        "Codec full Linux suite required",
    )
    require(
        belongs_to_supervisor(os.getpid(), gate["process"]) is True,
        "Codec worker outside owned supervisor",
    )

    def stop(signum, frame):
        raise TimeoutError("Owned codec supervisor requested stop")

    signal.signal(signal.SIGTERM, stop)
    result = qualify(p, root, root / RUN, args.source_sha, args.protocol_sha256)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
