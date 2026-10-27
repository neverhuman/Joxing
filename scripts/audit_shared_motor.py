"""Independent saved measurement/policy/replay audit, zero new physics."""

import argparse
from pathlib import Path

from joxing.motor_storage import RawCredit, write_json
from joxing.shared_motor_proof import RUN, audit_proof, read_packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = read_packet(root, args.protocol_sha256, args.source_sha)
    out = root / RUN
    if (out / "independent-audit.json").exists():
        raise FileExistsError("Do not overwrite an independent audit")
    RawCredit(out, p["raw_bytes"] - 512 * 1024)
    result = audit_proof(
        p, root, out, source_sha=args.source_sha, protocol_sha=args.protocol_sha256
    )
    write_json(out / "independent-audit.json", result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
