"""Independently regenerate/decode the bounded codec corpus in a second process."""

import argparse
from pathlib import Path

from joxing.motor_codec_qualification import RUN, audit_qualification, read_packet
from joxing.motor_storage import RawCredit, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = read_packet(root, args.protocol_sha256, args.source_sha)
    out = root / RUN
    if (out / "independent-audit.json").exists() or (out / "audit-fault.json").exists():
        raise FileExistsError("Do not overwrite codec audit attempts")
    owner = RawCredit(out, p["raw_bytes"] - 256 * 1024)
    try:
        result = audit_qualification(
            p, root, out, args.source_sha, args.protocol_sha256
        )
        write_json(out / "independent-audit.json", result)
    except Exception as error:
        import traceback

        with owner.emergency():
            write_json(
                out / "audit-fault.json",
                {
                    "status": "failed",
                    "new_physics": 0,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
