import gzip
import io
import json
import zipfile

import numpy as np
import pytest

from joxing import evidence_storage as storage
from joxing.motor_storage import CREDITS, RawCredit


def test_lossless_numeric_shapes_and_fault_bits(tmp_path):
    arrays = {
        "empty": np.empty((0, 2, 8), dtype=np.int64),
        "float": np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64),
        "endian": np.arange(24, dtype=">f4").reshape(2, 3, 4),
        "fortran": np.asfortranarray(np.arange(16, dtype=np.float32).reshape(4, 4)),
        "mask": np.array([True, False]),
    }
    path = tmp_path / "tape.npz"
    storage.save_npz(path, **arrays)
    decoded = storage.read_arrays(path)
    assert decoded.keys() == arrays.keys()
    for key, value in arrays.items():
        assert storage.array_descriptor(decoded[key]) == storage.array_descriptor(value)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)


def test_json_big_rng_integers_tick_keys_and_signed_zero(tmp_path):
    value = {"rng": 2**128 - 1, "samples": {"203": [-0.0, None, True], "1000": []}}
    path = tmp_path / "snapshot.json"
    storage.write_json(path, value)
    actual = storage.read_json(path)
    assert actual == value
    assert np.signbit(actual["samples"]["203"][0])
    assert storage.json_sha(actual) == storage.json_sha(value)
    path.write_text("{}")
    with pytest.raises(ValueError, match="Ambiguous"):
        storage.read_json(path)


def test_legacy_arrays_and_json_are_read_without_conversion(tmp_path):
    np.savez_compressed(tmp_path / "tape.npz", empty=np.zeros((0, 2)), a=np.arange(4))
    (tmp_path / "state.json").write_text('{"203": 999999999999999999999999999}')
    assert storage.read_arrays(tmp_path / "tape.npz")["a"].tolist() == list(range(4))
    assert (
        storage.read_json(tmp_path / "state.json")["203"] == 999999999999999999999999999
    )


@pytest.mark.parametrize("kind", ["array", "dtype", "shape", "json"])
def test_decoded_corruption_rejected_after_container_hash_repair(tmp_path, kind):
    path = tmp_path / ("state.json" if kind == "json" else "tape.npz")
    if kind == "json":
        storage.write_json(path, {"tick": 203})
        stored = tmp_path / "state.json.gz"
        stored.write_bytes(gzip.compress(b'{"tick":204}', mtime=0))
    else:
        storage.save_npz(path, a=np.arange(6, dtype=np.int64).reshape(2, 3))
        stored = path
        a = np.arange(6, dtype=np.int64).reshape(2, 3)
        if kind == "array":
            a[0, 0] = 9
        elif kind == "dtype":
            a = a.astype(np.float64)
        else:
            a = a.reshape(3, 2)
        with stored.open("wb") as f:
            np.savez_compressed(f, a=a)
    manifest = json.loads((tmp_path / storage.MANIFEST).read_text())
    entry = manifest["files"][path.name]
    entry.update(bytes=stored.stat().st_size, sha256=storage.sha_file(stored))
    (tmp_path / storage.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Decoded"):
        (storage.read_json if kind == "json" else storage.read_arrays)(path)


@pytest.mark.parametrize(
    "kind", ["duplicate", "traversal", "object", "shape", "truncated", "aggregate"]
)
def test_unsafe_archive_rejected_before_array_allocation(tmp_path, kind, monkeypatch):
    path = tmp_path / "bad.npz"
    data = io.BytesIO()
    np.save(data, np.arange(3), allow_pickle=False)
    raw = data.getvalue()
    if kind == "object":
        data = io.BytesIO()
        np.save(data, np.array([{}], object))
        raw = data.getvalue()
    elif kind == "shape":
        data = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            data, {"descr": "<f8", "fortran_order": False, "shape": (2**40,)}
        )
        raw = data.getvalue()
    elif kind == "truncated":
        raw = raw[:-1]
    elif kind == "aggregate":
        monkeypatch.setattr(storage, "MAX_DECODED", 8)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../a.npy" if kind == "traversal" else "a.npy", raw)
        if kind == "duplicate":
            with pytest.warns(UserWarning):
                archive.writestr("a.npy", raw)
    with pytest.raises((ValueError, EOFError)):
        storage.read_arrays(path)


def test_credit_fault_reserve_and_atomic_overwrite_peak_are_charged(tmp_path):
    owner = RawCredit(tmp_path, 2 * 1024**2 + 200_000)
    try:
        rng = np.random.default_rng(42)
        storage.save_npz(tmp_path / "a.npz", a=rng.standard_normal(12000))
        prior = (tmp_path / "a.npz").read_bytes()
        with pytest.raises(TimeoutError):
            storage.save_npz(tmp_path / "a.npz", a=rng.standard_normal(22000))
        assert (tmp_path / "a.npz").read_bytes() == prior
        assert (tmp_path / "a.npz.tmp").exists()
        assert owner.used() <= owner.maximum - owner.reserve
        with pytest.raises(FileExistsError):
            storage.save_npz(tmp_path / "a.npz", a=np.zeros(1))
        with owner.emergency():
            from joxing.motor_storage import write_json

            write_json(tmp_path / "fault.json", {"reason": "retained write failure"})
        assert owner.used() <= owner.maximum
    finally:
        CREDITS.pop(tmp_path.resolve())


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b"[[[[[0]]]]]"])
def test_malformed_or_deep_json_rejected(tmp_path, raw, monkeypatch):
    monkeypatch.setattr(storage, "MAX_DEPTH", 3)
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        storage.read_json(path)


def test_json_gzip_bomb_and_missing_manifest_rejected(tmp_path, monkeypatch):
    path = tmp_path / "a.json"
    storage.write_json(path, {"a": "x" * 1000})
    monkeypatch.setattr(storage, "MAX_DECODED", 100)
    with pytest.raises(ValueError):
        storage.read_json(path)
    (tmp_path / storage.MANIFEST).unlink()
    with pytest.raises(ValueError, match="manifest"):
        storage.read_json(path)


def test_packed_manifest_removal_is_not_accepted_as_legacy(tmp_path):
    storage.save_npz(tmp_path / "a.npz", a=np.zeros(5))
    (tmp_path / storage.MANIFEST).unlink()
    with pytest.raises(ValueError, match="manifest"):
        storage.read_arrays(tmp_path / "a.npz")


def test_lzma_huge_dictionary_is_rejected_before_decoder(tmp_path, monkeypatch):
    import struct

    storage.save_npz(tmp_path / "a.npz", a=np.zeros(5))
    path = tmp_path / "a.npz"
    raw = bytearray(path.read_bytes())
    name_size, extra_size = struct.unpack("<HH", raw[26:30])
    offset = 30 + name_size + extra_size
    raw[offset + 5 : offset + 9] = (2**32 - 1).to_bytes(4, "little")
    path.write_bytes(raw)
    manifest = json.loads((tmp_path / storage.MANIFEST).read_text())
    manifest["files"]["a.npz"]["sha256"] = storage.sha_file(path)
    (tmp_path / storage.MANIFEST).write_text(json.dumps(manifest))

    def forbidden(*a, **kw):
        raise AssertionError("Decoder allocated before dictionary validation")

    monkeypatch.setattr(storage.lzma, "LZMADecompressor", forbidden)
    with pytest.raises(ValueError, match="dictionary"):
        storage.read_arrays(path)
