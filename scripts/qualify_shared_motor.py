"""Run exactly the public frozen shared motor packet after the full Linux gate."""

import argparse
import json
import os
import signal
from pathlib import Path

from joxing_process_ownership import belongs_to_supervisor

from joxing.shared_motor_proof import RUN, read_packet, run_proof
from joxing.shared_motor_saved_audit import require


def verify_linux_gate(root, *, source_sha, protocol_sha):
    gate = json.loads(
        (
            root / "runs/v0.0.11/shared-motor-physical-supervision-01/result.json"
        ).read_text()
    )
    require(
        gate["status"] == "running"
        and gate["source_commit"] == source_sha
        and gate["protocol_sha256"] == protocol_sha,
        "Motor supervisor/gate provenance",
    )
    test = next(s for s in gate["stages"] if s["stage"] == "tests")
    require(
        test["status"] == "completed"
        and test["returncode"] == 0
        and not test["live_recorded_processes"],
        "Motor final Linux suite required",
    )
    require(
        belongs_to_supervisor(os.getpid(), gate["process"]) is True,
        "Motor runner outside owned supervisor",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = read_packet(root, args.protocol_sha256, args.source_sha)
    verify_linux_gate(
        root, source_sha=args.source_sha, protocol_sha=args.protocol_sha256
    )

    def owned_stop(signum, frame):
        raise TimeoutError("Owned supervisor requested stop")

    signal.signal(signal.SIGTERM, owned_stop)
    result = run_proof(
        p,
        root,
        root / RUN,
        source_sha=args.source_sha,
        protocol_sha=args.protocol_sha256,
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
