"""Restore exact small codec inputs on a fresh host; never mirror executions."""

import base64
import hashlib
import json
import os
from pathlib import Path

from receipt_encoding import decode_gzip


def materialize(root, config, charged_bytes, base=Path("/home/ubuntu/joxing")):
    root, base = Path(root).resolve(), Path(base).resolve()
    if not root.is_relative_to(base) or root == base:
        raise ValueError("Dedicated owned checkout required")
    aliases = config["aliases"]
    if (
        not aliases
        or len(set(aliases)) != len(aliases)
        or any(Path(n).name != n or n in (".", "..") for n in aliases)
    ):
        raise ValueError("Explicit simple alias roots required")
    # Refuse before touching any directory: existing executions/mirrors immutable.
    if any((base / n).exists() or (base / n).is_symlink() for n in aliases):
        raise FileExistsError("Never overwrite historical root or prior mirror")
    destination = root / "artifacts/generated/h28-receipt-inputs"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Fresh input mirror required")
    blob = (root / config["archive"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != config["encoded_sha256"]:
        raise ValueError("Frozen mirror archive changed")
    raw = decode_gzip(blob, limit=512 * 1024)
    if (
        len(raw) != config["decoded_archive_bytes"]
        or hashlib.sha256(raw).hexdigest() != config["decoded_archive_sha256"]
    ):
        raise ValueError("Mirror decoded archive identity")
    pack = json.loads(raw)
    if set(pack["files"]) != set(config["files"]):
        raise ValueError("Mirror input denominator")
    pending = {}
    for name, row in pack["files"].items():
        relative = Path(name).relative_to("/home/ubuntu/joxing")
        if relative.parts[0] not in aliases or ".." in relative.parts:
            raise ValueError("Foreign mirror input path")
        data = base64.b64decode(row["data_base64"], validate=True)
        expected = config["files"][name]
        if (
            len(data) != expected["bytes"]
            or hashlib.sha256(data).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Historical input bytes differ")
        pending[relative] = data
    marker = (
        json.dumps(config["marker"], sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if sum(map(len, pending.values())) + len(marker) != charged_bytes:
        raise ValueError("All mirror bytes must remain charged")
    destination.mkdir(parents=True, exist_ok=False)
    for relative, data in pending.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        target.chmod(0o444)
    marker_path = destination / "INPUT_MIRROR_NOT_EXECUTION.json"
    marker_path.write_bytes(marker)
    marker_path.chmod(0o444)
    for name in aliases:
        (base / name).symlink_to(destination / name, target_is_directory=True)
    return {
        "scope": "Read-only byte INPUT mirrors, NOT historical execution checkouts",
        "destination": str(destination),
        "charged_fixture_bytes": charged_bytes,
        "aliases": {str(base / n): str(destination / n) for n in aliases},
        "files": config["files"],
        "owner_uid": os.getuid(),
    }
