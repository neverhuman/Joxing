"""Explicit portable byte-fixture lookup; default historical paths stay unchanged."""

import hashlib
import json
import os
from pathlib import Path

ENV_ROOT = "JOXING_RECEIPT_FIXTURE_ROOT"
ENV_HASH = "JOXING_RECEIPT_FIXTURE_MANIFEST_SHA256"
MARKER = "INPUT_ONLY_NOT_EXECUTION.json"
BASE = Path("/home/ubuntu/joxing")
SCHEMA = "joxing-portable-receipt-fixtures-v1"


def compact(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def relative(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Explicit historical absolute fixture path required")
    return path.relative_to(BASE)


def validate_root(root, expected_hash):
    root = Path(root)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ValueError("Owned nonsymlink fixture root required")
    root = root.resolve()
    marker = root / MARKER
    if marker.is_symlink() or marker.stat().st_uid != os.getuid():
        raise ValueError("Owned fixture manifest required")
    raw = marker.read_bytes()
    if not expected_hash or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("Pinned portable manifest hash required")
    manifest = json.loads(raw)
    if manifest["schema"] != SCHEMA or manifest["scope"] != "INPUT_ONLY_NOT_EXECUTION":
        raise ValueError("Explicit fixture-only manifest required")
    names = set(manifest["files"]) | set(manifest["absent"]) | set(manifest["prefixes"])
    allowed = set()
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or str(path) in (".", ""):
            raise ValueError("Unsafe portable logical name")
        allowed.add(name)
        allowed.update(str(p) for p in path.parents if str(p) != ".")
    if set(manifest["files"]) & set(manifest["absent"]):
        raise ValueError("Conflicting positive/negative fixture")
    for name, expected in manifest["files"].items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root)
            or path.is_symlink()
            or path.stat().st_uid != os.getuid()
        ):
            raise ValueError("Foreign or symlinked fixture payload")
        raw = path.read_bytes()
        if (
            len(raw) != expected["bytes"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Portable historical bytes changed")
    for name in manifest["absent"]:
        path = root / name
        if path.exists() or path.is_symlink():
            raise ValueError("Required plaintext sibling absence violated")
    return root, manifest, allowed


def receipt_path(value):
    root = os.environ.get(ENV_ROOT)
    if root is None:
        return Path(value)
    root, _manifest, allowed = validate_root(root, os.environ.get(ENV_HASH))
    logical = str(relative(value))
    if logical not in allowed:
        raise ValueError("Unregistered portable receipt path; no host fallback")
    target = root / logical
    if not target.resolve().is_relative_to(root):
        raise ValueError("Portable fixture path escapes root")
    return target
