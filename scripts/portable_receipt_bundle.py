"""Materialize a complete exact fixture bundle without host-global aliases."""

import base64
import hashlib
import json
from pathlib import Path

from receipt_encoding import decode_gzip

from joxing.receipt_fixture_paths import MARKER, compact, validate_root


def hydrate(root, config, output):
    root, output = Path(root).resolve(), Path(output)
    if (
        not output.resolve().is_relative_to(root)
        or output.is_symlink()
        or output.exists()
    ):
        raise ValueError("Fresh owned fixture output required")
    blob = (root / config["archive"]).read_bytes()
    if (
        len(blob) != config["encoded_bytes"]
        or hashlib.sha256(blob).hexdigest() != config["encoded_sha256"]
    ):
        raise ValueError("Frozen fixture archive identity")
    raw = decode_gzip(blob, limit=1024 * 1024)
    if (
        len(raw) != config["decoded_bytes"]
        or hashlib.sha256(raw).hexdigest() != config["decoded_sha256"]
    ):
        raise ValueError("Decoded fixture archive identity")
    pack = json.loads(raw)
    manifest = config["manifest"]
    marker = compact(manifest)
    if hashlib.sha256(marker).hexdigest() != config["manifest_sha256"] or set(
        pack["files"]
    ) != set(manifest["files"]):
        raise ValueError("Exact complete fixture manifest required")
    pending = {}
    for name, row in pack["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) == ".":
            raise ValueError("Unsafe portable fixture path")
        data = base64.b64decode(row["data_base64"], validate=True)
        expected = manifest["files"][name]
        if (
            len(data) != expected["bytes"]
            or hashlib.sha256(data).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Historical receipt bytes changed")
        pending[name] = data
    if sum(map(len, pending.values())) + len(marker) != config["charged_bytes"]:
        raise ValueError("All decoded input bytes must remain charged")
    output.mkdir(parents=True, exist_ok=False)
    for name, data in pending.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        target.chmod(0o444)
    (output / MARKER).write_bytes(marker)
    (output / MARKER).chmod(0o444)
    validate_root(output, config["manifest_sha256"])
    return {
        "files": len(pending),
        "charged_bytes": config["charged_bytes"],
        "manifest_sha256": config["manifest_sha256"],
        "root": str(output.resolve()),
        "host_aliases_created": False,
    }
