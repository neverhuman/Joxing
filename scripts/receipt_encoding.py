"""Explicit lossless receipt encodings. Watchdog publication is unaffected."""

import base64
import gzip
import hashlib
import io
import json
import os
import stat
import zlib
from pathlib import Path

MODE = "compact-main-gzip-deployment-v1"
V2 = "gzip-main-stdout-bootstrap-v2"


def report_path(folder, mode="legacy"):
    if mode not in ("legacy", MODE, V2):
        raise ValueError("Unknown receipt encoding")
    return Path(folder) / ("result.json.gz" if mode == V2 else "result.json")


def compact_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def save_report(path, value, *, mode="legacy", legacy_save):
    if mode == "legacy":
        return legacy_save(path, value)
    if mode not in (MODE, V2):
        raise ValueError("Unknown receipt encoding")
    path = Path(path)
    data = compact_bytes(value)
    if len(data) > 512 * 1024:
        raise ValueError("Source gate report credit")
    if mode == V2:
        data = gzip.compress(data, compresslevel=1, mtime=0)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def wire_metadata(blob, raw, encoding):
    return {
        "encoding": encoding,
        "decoded_bytes": len(raw),
        "decoded_sha256": hashlib.sha256(raw).hexdigest(),
        "encoded_bytes": len(blob),
        "encoded_sha256": hashlib.sha256(blob).hexdigest(),
    }


def main_metadata(path):
    blob = Path(path).read_bytes()
    raw = decode_bootstrap(blob, limit=512 * 1024)
    return wire_metadata(blob, raw, V2)


def terminal_line(value, mode="legacy", *, main=None):
    raw = (json.dumps(value, allow_nan=mode != V2) + "\n").encode()
    if mode in ("legacy", MODE):
        return raw.decode().removesuffix("\n")
    if mode != V2 or len(raw) > 512 * 1024:
        raise ValueError("Terminal receipt mode/bound")
    blob = gzip.compress(raw, compresslevel=1, mtime=0)
    return json.dumps(
        wire_metadata(blob, raw, V2)
        | {
            "schema": "joxing-terminal-capsule-v1",
            "data_base64": base64.b64encode(blob).decode(),
            "main_report": main,
        },
        separators=(",", ":"),
    )


def decode_terminal(line):
    capsule = json.loads(line)
    if (
        capsule["schema"] != "joxing-terminal-capsule-v1"
        or capsule["encoding"] != V2
        or len(capsule["data_base64"]) > 1048576
    ):
        raise ValueError("Terminal capsule contract")
    blob = base64.b64decode(capsule["data_base64"], validate=True)
    raw = decode_bootstrap(blob, limit=512 * 1024)
    if any(capsule[k] != v for k, v in wire_metadata(blob, raw, V2).items()):
        raise ValueError("Terminal capsule hashes")
    return raw


def decode_gzip(data, *, limit=1048576):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
            decoded = stream.read(limit + 1)
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError("Corrupt bootstrap") from error
    if len(decoded) > limit:
        raise ValueError("Decoded bootstrap first limit")
    return decoded


def decode_bootstrap(data, *, limit=1048576):
    decoded = decode_gzip(data, limit=limit)
    json.loads(decoded)
    return decoded


def pack_owned_control(path, expected_sha256):
    """After producer exit: exact owned XML encoding, never remove old inputs."""
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
        raise ValueError("Control report ownership")
    raw = path.read_bytes()
    if len(raw) > 1048576 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Control report changed/bound")
    destination = path.with_suffix(path.suffix + ".gz")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Control encoding already exists")
    blob = gzip.compress(raw, compresslevel=1, mtime=0)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(blob)
    temporary.replace(destination)
    if decode_gzip(destination.read_bytes()) != raw or path.read_bytes() != raw:
        raise ValueError("Control encoding fidelity")
    after = path.lstat()
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        raise ValueError("Control report identity changed")
    path.unlink()
    return wire_metadata(blob, raw, V2) | {
        "path": destination.name,
        "scope": "exact owned NEWJUnit bytes after producer exit",
    }


def encode_bootstrap(value, *, limit=1048576):
    raw = compact_bytes(value)
    if len(raw) > limit:
        raise ValueError("Bootstrap first limit")
    encoded = gzip.compress(raw, compresslevel=1, mtime=0)
    if decode_bootstrap(encoded, limit=limit) != raw:
        raise ValueError("Bootstrap round-trip mismatch")
    return encoded, {
        "encoding": MODE,
        "decoded_bytes": len(raw),
        "decoded_sha256": hashlib.sha256(raw).hexdigest(),
        "encoded_bytes": len(encoded),
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
    }
