"""Lossless, bounded evidence containers; no simulation or controller changes.

A directory manifest binds container bytes AND decoded semantics. Receipts must
pin the manifest hash externally; no self-described hash authenticates itself.
Failed temporary writes are retained and charged to the same RawCredit.
"""

import gzip
import hashlib
import io
import json
import lzma
import math
import re
import struct
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .motor_storage import credit
from .motor_storage import write_json as plain_write_json

SCHEMA = "joxing-lossless-sensor-evidence-v1"
MANIFEST = "storage-manifest.json"
MAX_STORED = 64 * 1024**2
MAX_DECODED = 128 * 1024**2
MAX_MANIFEST = 512 * 1024
MAX_MEMBERS = 256
MAX_DEPTH = 32
MAX_NODES = 8_000_000
KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}\Z")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree(value):
    pending = [(value, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise ValueError("JSON evidence tree bound")
        if isinstance(node, dict):
            if not all(type(k) is str for k in node):
                raise ValueError("JSON evidence string keys required")
            pending.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, (list, tuple)):
            pending.extend((v, depth + 1) for v in node)
        elif type(node) not in (str, int, float, bool, type(None)):
            raise ValueError("JSON evidence primitive required")
        elif type(node) is float and not math.isfinite(node):
            raise ValueError("Nonfinite JSON evidence")


def canonical(value):
    _tree(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def json_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("Duplicate JSON key")
        result[k] = v
    return result


def _parse(raw):
    if len(raw) > MAX_DECODED:
        raise ValueError("JSON decoded byte bound")
    # Check nesting before the JSON decoder allocates a recursively nested tree.
    depth, quoted, escaped = 0, False, False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (91, 123):
            depth += 1
            if depth > MAX_DEPTH + 1:
                raise ValueError("JSON evidence tree bound")
        elif char in (93, 125):
            depth -= 1
    value = json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
    )
    _tree(value)
    return value


class _Sink:
    """Unbuffered, pre-write credit checks including ZIP seek/backpatch writes."""

    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("xb", buffering=0)

    def write(self, data):
        end = self.file.tell() + len(data)
        if end > MAX_STORED:
            raise ValueError("Stored evidence byte bound")
        growth = max(0, end - self.path.stat().st_size)
        owner = credit(self.path)
        if owner:
            owner.check(self.path, growth, append=True)
        return self.file.write(data)

    def tell(self):
        return self.file.tell()

    def seek(self, *args):
        return self.file.seek(*args)

    def flush(self):
        return self.file.flush()


@contextmanager
def _atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    # Never overwrite an unexamined fault from an earlier failed write.
    sink = _Sink(temporary)
    try:
        yield sink
        sink.flush()
        sink.file.close()
        temporary.replace(path)
    finally:
        sink.file.close()


def _manifest(folder):
    path = Path(folder) / MANIFEST
    if not path.exists():
        return {"schema": SCHEMA, "files": {}}
    if path.stat().st_size > MAX_MANIFEST:
        raise ValueError("Manifest byte bound")
    result = _parse(path.read_bytes())
    if set(result) != {"schema", "files"} or result["schema"] != SCHEMA:
        raise ValueError("Storage schema")
    if type(result["files"]) is not dict or len(result["files"]) > MAX_MEMBERS:
        raise ValueError("Manifest file bound")
    return result


def _record(logical, stored, semantics, kind):
    logical, stored = Path(logical), Path(stored)
    manifest = _manifest(logical.parent)
    manifest["files"][logical.name] = {
        "stored": stored.name,
        "kind": kind,
        "bytes": stored.stat().st_size,
        "sha256": sha_file(stored),
        "decoded": semantics,
    }
    if len(canonical(manifest)) > MAX_MANIFEST:
        raise ValueError("Manifest byte bound")
    plain_write_json(logical.parent / MANIFEST, manifest)


def _entry(path, kind):
    path = Path(path)
    manifest = _manifest(path.parent)
    entry = manifest["files"].get(path.name)
    if entry is None:
        return None
    if set(entry) != {"stored", "kind", "bytes", "sha256", "decoded"}:
        raise ValueError("Manifest entry fields")
    expected = path.name + (".gz" if kind == "json" else "")
    if entry["stored"] != expected or entry["kind"] != kind:
        raise ValueError("Manifest entry path/kind")
    stored = path.parent / expected
    if not stored.is_file() or stored.stat().st_size > MAX_STORED:
        raise ValueError("Stored evidence missing/byte bound")
    if stored.stat().st_size != entry["bytes"] or sha_file(stored) != entry["sha256"]:
        raise ValueError("Container digest mismatch")
    return entry


def write_json(path, value):
    path = Path(path)
    if path.suffix != ".json" or path.exists():
        raise ValueError("Packed JSON requires unambiguous .json logical path")
    raw = canonical(value)
    if len(raw) > MAX_DECODED:
        raise ValueError("JSON decoded byte bound")
    stored = path.with_name(path.name + ".gz")
    with (
        _atomic(stored) as sink,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=sink, compresslevel=1, mtime=0
        ) as stream,
    ):
        stream.write(raw)
    _record(
        path,
        stored,
        {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        "json",
    )


def read_json(path):
    path = Path(path)
    compressed = path.with_name(path.name + ".gz")
    if path.exists() and compressed.exists():
        raise ValueError("Ambiguous JSON evidence")
    entry = _entry(path, "json")
    if entry is None:
        if compressed.exists():
            raise ValueError("Packed JSON requires manifest")
        if path.stat().st_size > MAX_DECODED:
            raise ValueError("JSON byte bound")
        return _parse(path.read_bytes())
    with gzip.open(compressed, "rb") as stream:
        raw = stream.read(MAX_DECODED + 1)
    value = _parse(raw)
    if {
        "bytes": len(raw),
        "sha256": hashlib.sha256(canonical(value)).hexdigest(),
    } != entry["decoded"]:
        raise ValueError("Decoded JSON digest mismatch")
    return value


def array_descriptor(value):
    value = np.asarray(value)
    if value.dtype.kind not in "biuf" or value.dtype.hasobject:
        raise ValueError("Only primitive numerical arrays")
    if value.ndim > 8 or value.nbytes > MAX_DECODED:
        raise ValueError("Array dimension/byte bound")
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "bytes": value.nbytes,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def save_npz(path, **arrays):
    path = Path(path)
    if (
        not arrays
        or len(arrays) > MAX_MEMBERS
        or any(not KEY.fullmatch(k) for k in arrays)
    ):
        raise ValueError("Array key/member bound")
    descriptions = {k: array_descriptor(v) for k, v in arrays.items()}
    if sum(v["bytes"] for v in descriptions.values()) > MAX_DECODED:
        raise ValueError("Array aggregate byte bound")
    with (
        _atomic(path) as sink,
        zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_LZMA) as archive,
    ):
        for key, value in arrays.items():
            with archive.open(key + ".npy", "w") as member:
                np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)
    _record(path, path, descriptions, "npz")


def _lzma_member(archive, member):
    # zipfile's LZMA adapter has no max_length. Decode with a bounded standard
    # LZMA decoder instead, after checking the dictionary BEFORE allocation.
    stream = archive.fp
    stream.seek(member.header_offset)
    header = stream.read(30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        raise ValueError("ZIP local header")
    name_size, extra_size = struct.unpack("<HH", header[26:30])
    stream.seek(name_size + extra_size, 1)
    encoded = stream.read(member.compress_size)
    if len(encoded) != member.compress_size or len(encoded) < 9:
        raise ValueError("Truncated LZMA member")
    if struct.unpack("<H", encoded[2:4])[0] != 5:
        raise ValueError("LZMA property bound")
    prop = encoded[4]
    dictionary = int.from_bytes(encoded[5:9], "little")
    if prop >= 225 or not 4096 <= dictionary <= 8 * 1024**2:
        raise ValueError("LZMA dictionary/property bound")
    decoder = lzma.LZMADecompressor(
        lzma.FORMAT_RAW,
        filters=[
            {
                "id": lzma.FILTER_LZMA1,
                "dict_size": dictionary,
                "lc": prop % 9,
                "lp": (prop // 9) % 5,
                "pb": prop // 45,
            }
        ],
    )
    raw = decoder.decompress(encoded[9:], max_length=member.file_size + 1)
    if len(raw) != member.file_size or not decoder.eof or decoder.unused_data:
        raise ValueError("LZMA decoded length/stream bound")
    if zlib.crc32(raw) != member.CRC:
        raise ValueError("LZMA CRC mismatch")
    return io.BytesIO(raw)


def read_arrays(path):
    path = Path(path)
    if path.stat().st_size > MAX_STORED:
        raise ValueError("Stored arrays byte bound")
    entry = _entry(path, "npz")
    result = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if not members or len(members) > MAX_MEMBERS or len(set(names)) != len(names):
            raise ValueError("Array member/duplicate bound")
        if sum(m.file_size for m in members) > MAX_DECODED + MAX_MEMBERS * 65536:
            raise ValueError("Array aggregate decoded bound")
        total = 0
        for member in members:
            name = member.filename
            if not name.endswith(".npy") or not KEY.fullmatch(name[:-4]):
                raise ValueError("Unsafe array member")
            if member.compress_type == zipfile.ZIP_LZMA and entry is None:
                raise ValueError("Packed arrays require manifest")
            if member.flag_bits & 1 or member.compress_type not in (0, 8, 14):
                raise ValueError("Unsupported array encoding")
            if member.file_size > MAX_DECODED + 65536:
                raise ValueError("Array decoded byte bound")
            with (
                _lzma_member(archive, member)
                if member.compress_type == zipfile.ZIP_LZMA
                else archive.open(member)
            ) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(
                        stream, max_header_size=65536
                    )
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(
                        stream, max_header_size=65536
                    )
                else:
                    raise ValueError("Unsupported NPY version")
                size = math.prod(shape) * dtype.itemsize
                if len(shape) > 8 or dtype.kind not in "biuf" or dtype.hasobject:
                    raise ValueError("Unsafe array dtype/dimension")
                if size > MAX_DECODED or stream.tell() + size != member.file_size:
                    raise ValueError("Array declared shape/size mismatch")
                total += size
                if total > MAX_DECODED:
                    raise ValueError("Array aggregate byte bound")
                raw = stream.read(size + 1)
                if len(raw) != size:
                    raise ValueError("Truncated array")
                result[name[:-4]] = (
                    np.frombuffer(raw, dtype=dtype)
                    .reshape(shape, order="F" if order else "C")
                    .copy()
                )
    if (
        entry is not None
        and {k: array_descriptor(v) for k, v in result.items()} != entry["decoded"]
    ):
        raise ValueError("Decoded array digest mismatch")
    return result
