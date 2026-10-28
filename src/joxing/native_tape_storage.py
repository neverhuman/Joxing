"""Opt-in exact ZIP_STORED candidate; no existing collector uses this writer.

Externally pinned receipts authenticate both the file and logical C-order bytes.
The reader bounds the central directory before zipfile allocates member objects.
Interrupted writes retain their temporary bytes and never return a receipt.
"""

import hashlib
import io
import math
import os
import re
import struct
import zipfile
from pathlib import Path

import numpy as np

from .motor_storage import credit

SCHEMA = "joxing-native-stored-tape-v1"
MAX_BYTES = 64 * 1024**2
MAX_MEMBERS = 32
MAX_HEADER = 4096
KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}\Z")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _layout(arrays, maximum_payload_bytes):
    if (
        not isinstance(arrays, dict)
        or not 1 <= len(arrays) <= MAX_MEMBERS
        or any(type(k) is not str or not KEY.fullmatch(k) for k in arrays)
    ):
        raise ValueError("Stored tape keys/member bound")
    total = 0
    for a in arrays.values():
        if (
            not isinstance(a, np.ndarray)
            or a.dtype.kind not in "biuf"
            or a.dtype.hasobject
            or a.ndim > 8
            or any(n > 2**31 for n in a.shape)
        ):
            raise ValueError("Stored tape array dtype/dimension")
        total += a.nbytes
        if total > maximum_payload_bytes:
            raise ValueError("Stored tape aggregate payload bound")
    return total


def _header(a):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {"descr": a.dtype.str, "fortran_order": False, "shape": a.shape},
    )
    result = stream.getvalue()
    if len(result) > MAX_HEADER:
        raise ValueError("Stored tape NPY header bound")
    return result


def plan(arrays, *, maximum_payload_bytes=MAX_BYTES):
    if not 0 <= maximum_payload_bytes <= MAX_BYTES:
        raise ValueError("Stored tape requested payload bound")
    total = _layout(arrays, maximum_payload_bytes)
    descriptors, headers = {}, {}
    archive_bytes = 22  # EOCD; no archive comment, extra fields or ZIP64.
    for key in sorted(arrays):
        a = arrays[key]
        headers[key] = _header(a)
        name_size = len(key + ".npy")
        archive_bytes += 30 + 46 + 2 * name_size + len(headers[key]) + a.nbytes
        descriptors[key] = {
            "dtype": a.dtype.str,
            "shape": list(a.shape),
            "bytes": a.nbytes,
            "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest(),
        }
    return {
        "schema": SCHEMA,
        "payload_bytes": total,
        "archive_bytes": archive_bytes,
        "arrays": descriptors,
    }


class _Sink:
    def __init__(self, path, maximum):
        self.path, self.maximum = Path(path), maximum
        self.file = self.path.open("xb", buffering=0)

    def write(self, data):
        end = self.file.tell() + len(data)
        if end > self.maximum:
            raise ValueError("Stored tape output bound")
        owner = credit(self.path)
        if owner:
            owner.check(
                self.path,
                max(0, end - self.path.stat().st_size),
                append=True,
            )
        return self.file.write(data)

    def tell(self):
        return self.file.tell()

    def seek(self, *args):
        return self.file.seek(*args)

    def flush(self):
        return self.file.flush()


def store(path, arrays, *, maximum_archive_bytes=MAX_BYTES):
    """Caller owns one output directory; refuse previous final or temporary files."""
    path = Path(path)
    if not 0 < maximum_archive_bytes <= MAX_BYTES:
        raise ValueError("Stored tape requested archive bound")
    proposal = plan(arrays)
    if proposal["archive_bytes"] > maximum_archive_bytes:
        raise ValueError("Stored tape archive preflight bound")
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("Stored tape refuses previous output")
    owner = credit(temporary)
    if owner:
        owner.check(temporary, proposal["archive_bytes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    sink = _Sink(temporary, proposal["archive_bytes"])
    try:
        with zipfile.ZipFile(sink, "w", allowZip64=False) as archive:
            for key in proposal["arrays"]:
                a = np.ascontiguousarray(arrays[key])
                # ascontiguousarray promotes scalars; keep their original shape.
                a = a.reshape(arrays[key].shape)
                z = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                z.compress_type = zipfile.ZIP_STORED
                z.create_system = 3
                z.external_attr = 0o600 << 16
                z.file_size = len(_header(a)) + a.nbytes
                with archive.open(z, "w") as member:
                    member.write(_header(a))
                    raw = memoryview(a.reshape(-1)).cast("B")
                    for start in range(0, len(raw), 1024 * 1024):
                        member.write(raw[start : start + 1024 * 1024])
        sink.flush()
        os.fsync(sink.file.fileno())
        if temporary.stat().st_size != proposal["archive_bytes"]:
            raise ValueError("Stored tape exact preflight size differs")
        digest = sha_file(temporary)
        if path.exists():
            raise FileExistsError("Stored tape final appeared during owned write")
        temporary.rename(path)
        return {**proposal, "sha256": digest}
    finally:
        sink.file.close()


def _central_bound(path, size):
    if size < 22:
        raise ValueError("Stored tape truncated end record")
    with Path(path).open("rb") as f:
        f.seek(-22, 2)
        sig, disk, cd_disk, local_n, n, cd_size, offset, comment = struct.unpack(
            "<4s4H2IH", f.read(22)
        )
    if (
        sig != b"PK\x05\x06"
        or disk
        or cd_disk
        or local_n != n
        or not 1 <= n <= MAX_MEMBERS
        or cd_size > MAX_MEMBERS * (46 + 84)
        or cd_size < n * 46
        or offset + cd_size != size - 22
        or comment
    ):
        raise ValueError("Stored tape central directory bound")
    return n, offset


def load(path, receipt):
    """Authenticate the externally bound receipt, then parse bounded NPY members."""
    path = Path(path)
    size = path.stat().st_size
    if (
        not 0 < size <= MAX_BYTES
        or receipt.get("schema") != SCHEMA
        or size != receipt.get("archive_bytes")
        or sha_file(path) != receipt.get("sha256")
    ):
        raise ValueError("Stored tape external container identity")
    count, central_start = _central_bound(path, size)
    expected = receipt["arrays"]
    if (
        type(expected) is not dict
        or len(expected) != count
        or any(type(k) is not str or not KEY.fullmatch(k) for k in expected)
    ):
        raise ValueError("Stored tape receipt member bound")
    result, cursor, total = {}, 0, 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if [m.filename for m in members] != [k + ".npy" for k in sorted(expected)]:
            raise ValueError("Stored tape member identity/order")
        for m in members:
            if (
                m.compress_type != zipfile.ZIP_STORED
                or m.flag_bits
                or m.extra
                or m.comment
                or m.header_offset != cursor
                or m.file_size != m.compress_size
                or not 0 < m.file_size <= MAX_BYTES + MAX_HEADER
            ):
                raise ValueError("Stored tape member encoding/extent")
            cursor += 30 + len(m.filename) + m.file_size
            if cursor > central_start:
                raise ValueError("Stored tape member overlap")
            with archive.open(m) as stream:
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError("Stored tape NPY version")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                    stream, max_header_size=MAX_HEADER
                )
                if (
                    fortran
                    or len(shape) > 8
                    or any(n > 2**31 for n in shape)
                    or dtype.kind not in "biuf"
                    or dtype.hasobject
                ):
                    raise ValueError("Stored tape NPY layout")
                payload = math.prod(shape) * dtype.itemsize
                total += payload
                if (
                    total > MAX_BYTES
                    or stream.tell() + payload != m.file_size
                    or stream.tell() > MAX_HEADER
                ):
                    raise ValueError("Stored tape declared payload bound")
                raw = stream.read(payload + 1)
                if len(raw) != payload:
                    raise ValueError("Stored tape truncated payload")
                descriptor = {
                    "dtype": dtype.str,
                    "shape": list(shape),
                    "bytes": payload,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                key = m.filename[:-4]
                if descriptor != expected[key]:
                    raise ValueError("Stored tape logical identity")
                result[key] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if cursor != central_start or total != receipt["payload_bytes"]:
        raise ValueError("Stored tape aggregate extent")
    return result
