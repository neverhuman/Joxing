"""Explicit lossless receipt encodings. Watchdog publication is unaffected."""

import gzip
import hashlib
import io
import json
import zlib
from pathlib import Path

MODE = "compact-main-gzip-deployment-v1"


def compact_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def save_report(path, value, *, mode="legacy", legacy_save):
    if mode == "legacy":
        return legacy_save(path, value)
    if mode != MODE:
        raise ValueError("Unknown receipt encoding")
    path = Path(path)
    data = compact_bytes(value)
    if len(data) > 512 * 1024:
        raise ValueError("Source gate report credit")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def decode_bootstrap(data, *, limit=1048576):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
            decoded = stream.read(limit + 1)
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError("Corrupt bootstrap") from error
    if len(decoded) > limit:
        raise ValueError("Decoded bootstrap first limit")
    json.loads(decoded)
    return decoded


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
