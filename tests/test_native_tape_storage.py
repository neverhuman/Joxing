import copy
import hashlib
import io
import struct
import zipfile

import numpy as np
import pytest

from joxing import native_tape_storage as storage
from joxing.motor_storage import CREDITS, RawCredit


def assert_bits(actual, expected):
    assert actual.keys() == expected.keys()
    for k, a in expected.items():
        assert actual[k].dtype == a.dtype
        assert actual[k].shape == a.shape
        assert actual[k].tobytes() == a.tobytes(order="C")


def rebound(path, receipt):
    return {
        **receipt,
        "archive_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_two_decoders_exact_for_nan_payload_endian_strides_empty_and_scalar(tmp_path):
    arrays = {
        "bits": np.array(
            [0, 1 << 63, 0x7FF8000000000001, 0xFFF8000000000023], dtype=np.uint64
        ).view(np.float64),
        "bigendian": np.arange(12, dtype=">f4").reshape(3, 4),
        "fortran": np.asfortranarray(np.arange(24, dtype=np.int32).reshape(4, 6)),
        "slice": np.arange(40, dtype=np.int64)[::-3],
        "empty": np.zeros((0, 3, 8), dtype=np.float32),
        "scalar": np.array(17, dtype=np.uint16),
        "mask": np.array([True, False]),
    }
    path = tmp_path / "stored.npz"
    receipt = storage.store(path, arrays)
    assert path.stat().st_size == storage.plan(arrays)["archive_bytes"]
    assert_bits(storage.load(path, receipt), arrays)
    with np.load(path, allow_pickle=False) as other:
        assert_bits({k: other[k] for k in other.files}, arrays)
    with zipfile.ZipFile(path) as other:
        assert all(m.compress_type == zipfile.ZIP_STORED for m in other.infolist())


def test_deterministic_bytes_across_input_order(tmp_path):
    arrays = {"z": np.arange(12), "a": np.array([-0.0, 1.0])}
    first = storage.store(tmp_path / "one.npz", arrays)
    second = storage.store(tmp_path / "two.npz", dict(reversed(list(arrays.items()))))
    assert first == second
    assert (tmp_path / "one.npz").read_bytes() == (tmp_path / "two.npz").read_bytes()


@pytest.mark.parametrize(
    "arrays",
    [
        {},
        {"../escape": np.arange(2)},
        {"a": np.array([object()])},
        {"a": np.zeros((1,) * 9)},
        {"a": [1, 2]},
        {"a": np.array([1j])},
    ],
)
def test_rejects_unsafe_input_before_output(tmp_path, arrays):
    with pytest.raises(ValueError):
        storage.store(tmp_path / "x.npz", arrays)
    assert not list(tmp_path.iterdir())


def test_huge_view_and_archive_limit_fail_before_allocation_or_output(tmp_path):
    huge = np.lib.stride_tricks.as_strided(
        np.zeros(1, np.uint8), shape=(storage.MAX_BYTES + 1,), strides=(0,)
    )
    with pytest.raises(ValueError, match="aggregate payload"):
        storage.store(tmp_path / "huge.npz", {"a": huge})
    a = {"a": np.arange(12)}
    with pytest.raises(ValueError, match="archive preflight"):
        storage.store(tmp_path / "small.npz", a, maximum_archive_bytes=5)
    assert not list(tmp_path.iterdir())


def test_credit_preflight_preserves_fault_reserve(tmp_path):
    owner = RawCredit(tmp_path, 2 * 1024**2 + 10)
    try:
        with pytest.raises(TimeoutError, match="credit"):
            storage.store(tmp_path / "x.npz", {"a": np.arange(12)})
        assert owner.used() == 0
    finally:
        CREDITS.pop(tmp_path.resolve())


@pytest.mark.parametrize("previous", ["x.npz", "x.npz.tmp"])
def test_previous_final_or_temporary_is_never_overwritten(tmp_path, previous):
    (tmp_path / previous).write_bytes(b"retained fault or result")
    with pytest.raises(FileExistsError):
        storage.store(tmp_path / "x.npz", {"a": np.arange(12)})
    assert (tmp_path / previous).read_bytes() == b"retained fault or result"


def test_interrupted_write_has_temporary_but_no_final_or_receipt(tmp_path, monkeypatch):
    original = storage._Sink.write

    def fail_after_header(self, data):
        if self.file.tell() >= 40:
            raise OSError("injected disk fault")
        return original(self, data)

    monkeypatch.setattr(storage._Sink, "write", fail_after_header)
    with pytest.raises(OSError, match="injected disk fault"):
        storage.store(tmp_path / "x.npz", {"a": np.arange(12)})
    assert not (tmp_path / "x.npz").exists()
    assert (tmp_path / "x.npz.tmp").stat().st_size > 0


def test_external_hash_and_logical_receipt_mismatch_rejected(tmp_path):
    path = tmp_path / "x.npz"
    receipt = storage.store(path, {"a": np.arange(12)})
    wrong = {**receipt, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="container identity"):
        storage.load(path, wrong)
    wrong = copy.deepcopy(receipt)
    wrong["arrays"]["a"]["dtype"] = "<f8"
    with pytest.raises(ValueError, match="logical identity"):
        storage.load(path, wrong)


def test_crc_corruption_rejected_even_with_rebound_container_hash(tmp_path):
    path = tmp_path / "x.npz"
    receipt = storage.store(path, {"a": np.arange(120)})
    raw = bytearray(path.read_bytes())
    raw[30 + len("a.npy") + len(storage._header(np.arange(120))) + 9] ^= 1
    path.write_bytes(raw)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        storage.load(path, rebound(path, receipt))


def test_truncated_and_forged_central_directory_rejected_before_zip_allocation(
    tmp_path, monkeypatch
):
    path = tmp_path / "x.npz"
    receipt = storage.store(path, {"a": np.arange(12)})
    complete = path.read_bytes()
    monkeypatch.setattr(
        storage.zipfile,
        "ZipFile",
        lambda *a, **kw: pytest.fail("Unbounded central directory reached zipfile"),
    )
    path.write_bytes(complete[:-1])
    with pytest.raises(ValueError, match="central directory"):
        storage.load(path, rebound(path, receipt))
    raw = bytearray(complete)
    struct.pack_into("<HH", raw, len(raw) - 14, 33, 33)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="central directory"):
        storage.load(path, rebound(path, receipt))


@pytest.mark.parametrize("kind", ["shape", "object", "compressed"])
def test_malicious_shape_object_and_compressed_members_rejected(tmp_path, kind):
    path = tmp_path / "x.npz"
    receipt = storage.store(path, {"a": np.arange(12)})
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "descr": "|O" if kind == "object" else "<f8",
            "fortran_order": False,
            "shape": (2**31,) if kind == "shape" else (1,),
        },
    )
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED if kind == "compressed" else 0
    ) as z:
        z.writestr("a.npy", stream.getvalue() + b"\0" * 8)
    with pytest.raises(ValueError, match="layout|payload|encoding"):
        storage.load(path, rebound(path, receipt))
