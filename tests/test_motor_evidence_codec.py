import copy
import json
import lzma
import struct

import numpy as np
import pytest
from test_shared_motor_evidence import proof  # noqa: F401 -- reuse physical fixture

from joxing.evidence_storage import array_descriptor, canonical, read_arrays, read_json
from joxing.motor_codec_independent import independent_decode
from joxing.motor_evidence_codec import (
    MAGIC,
    MAX_HEADER,
    context_from_trace,
    decode,
    encode,
    unxor_planes,
    write_blob,
    xor_planes,
)
from joxing.motor_storage import RawCredit
from joxing.provenance import digest


@pytest.fixture(scope="module")
def corpus(proof):  # noqa: F811 -- pytest fixture injection
    task, folder, _, _, _ = proof
    rows = []
    for name in ("source", "replay"):
        trace = read_json(folder / name / "trace.json")
        arrays = read_arrays(folder / name / "tape.npz")
        context, refs = context_from_trace(
            task.model, task.drivers["A"].task.model, trace
        )
        blob = encode(arrays, context, refs)
        rows.append((arrays, context, refs, blob))
    return rows


def test_physical_source_and_actual_held_replay_all23arrays_exact_in_two_decoders(
    corpus,
):
    for arrays, context, refs, blob in corpus:
        assert len(arrays) == 23
        expected = {k: array_descriptor(a) for k, a in arrays.items()}
        for decoded in (
            decode(blob, expected_context_sha256=digest(context)),
            independent_decode(blob, context, refs, expected),
        ):
            assert set(decoded) == set(arrays)
            assert all(
                array_descriptor(a) == array_descriptor(decoded[k])
                for k, a in arrays.items()
            )


@pytest.mark.parametrize("kind", ["<f8", ">f8", "<f4", "<i8", "|b1"])
@pytest.mark.parametrize("shape", [(), (0,), (3, 0), (1,), (17, 3, 2)])
def test_byte_xor_preserves_scalar_empty_endian_and_ieee_bits(kind, shape):
    a = np.resize(
        np.array([0.0, -0.0, 1.0, -2.0, np.inf, np.nan]),
        max(1, np.prod(shape, dtype=int)),
    )
    if np.dtype(kind).kind in "ib":
        a = np.arange(len(a)) % 2
    a = np.asarray(a[: int(np.prod(shape))] if shape else a[0], dtype=kind).reshape(
        shape
    )
    b = unxor_planes(xor_planes(a), a.dtype, shape)
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize(
    "field",
    ["raw", "normalized", "excitation", "witness_qpos", "witness_qvel", "witness_act"],
)
def test_encoder_never_omits_changed_actual_measurement(corpus, field):
    arrays, context, refs, _ = corpus[0]
    bad = {k: a.copy() for k, a in arrays.items()}
    bad[field].flat[0] += 0.01
    with pytest.raises(ValueError, match="actual derived"):
        encode(bad, context, refs)


def rewrite(blob, edit):
    n = struct.unpack("<I", blob[8:12])[0]
    h = json.loads(blob[12 : 12 + n])
    edit(h)
    header = canonical(h)
    return MAGIC + struct.pack("<I", len(header)) + header + blob[12 + n :]


@pytest.mark.parametrize(
    "fault",
    [
        "normalizer",
        "map",
        "reference",
        "dependency",
        "duplicate",
        "traversal",
        "shape",
        "dtype",
        "aggregate",
        "trailing",
        "truncated",
        "dict",
    ],
)
def test_rehashed_metadata_or_stream_fault_is_rejected(corpus, fault):
    _arrays, c, _refs, blob = corpus[0]
    expected = digest(c)
    assert decode(blob, expected_context_sha256=expected)
    if fault in ("normalizer", "map", "reference"):

        def edit(h):
            if fault == "normalizer":
                h["context"]["normalizer"]["mean"][0] += 1
            elif fault == "map":
                h["context"]["maps"]["A"]["qpos"].reverse()
            else:
                h["context"]["references"]["sha256"] = "0" * 64
            h["context_sha256"] = digest(h["context"])

        bad = rewrite(blob, edit)
    elif fault == "dependency":
        bad = rewrite(blob, lambda h: h.update(dependencies=["self-cycle"]))
    elif fault == "duplicate":
        bad = rewrite(
            blob, lambda h: h["members"].__setitem__(1, copy.deepcopy(h["members"][0]))
        )
    elif fault == "traversal":
        bad = rewrite(blob, lambda h: h["members"][0].update(name="../unsafe"))
    elif fault == "shape":
        bad = rewrite(
            blob, lambda h: h["logical"]["raw"].update(shape=[10**12, 2, 3601])
        )
    elif fault == "dtype":
        bad = rewrite(blob, lambda h: h["logical"]["raw"].update(dtype="|O"))
    elif fault == "aggregate":
        bad = rewrite(blob, lambda h: h.update(decoded_bytes=128 * 1024**2 + 1))
    elif fault == "trailing":
        bad = blob + b"extra"
    elif fault == "truncated":
        bad = blob[:-20]
    else:
        n = struct.unpack("<I", blob[8:12])[0]
        # Oversized XZ dictionary must fail BEFORE decompression allocation.
        bad = blob[: 12 + n] + lzma.compress(
            b"", filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 32 * 1024**2}]
        )
    with pytest.raises(ValueError):
        decode(bad, expected_context_sha256=expected)


def test_header_and_write_bounds_retain_failed_temporary(corpus, tmp_path):
    _, c, _, blob = corpus[0]
    with pytest.raises(ValueError):
        decode(
            MAGIC + struct.pack("<I", MAX_HEADER + 1) + b"x",
            expected_context_sha256=digest(c),
        )
    owner = RawCredit(tmp_path, 3 * 1024**2)
    # Fill ordinary allowance so the existing unbuffered sink fails pre-write.
    (tmp_path / "filler").write_bytes(b"x" * (owner.maximum - owner.reserve - 10))
    with pytest.raises(TimeoutError):
        write_blob(tmp_path / "corpus.jmc", blob)
    assert (tmp_path / "corpus.jmc.tmp").exists() and not (
        tmp_path / "corpus.jmc"
    ).exists()
    with pytest.raises(FileExistsError):
        write_blob(tmp_path / "corpus.jmc", blob)


def test_incomplete_arrays_never_pad_missing_physical_data(corpus):
    arrays, c, refs, _ = corpus[0]
    incomplete = {
        k: arrays[k]
        for k in [
            "decision_ticks",
            "decision_engines",
            "raw",
            "normalized",
            "commands",
            "excitation",
            "ticks",
        ]
    }
    with pytest.raises(ValueError, match="denominator"):
        encode(incomplete, c, refs)


@pytest.mark.parametrize("fault", [False, True])
def test_qualification_registry_shard_and_separate_audit_use_no_new_physics(
    corpus,
    proof,  # noqa: F811 -- pytest fixture injection
    tmp_path,
    monkeypatch,
    fault,
):
    import joxing.motor_codec_qualification as module

    task, folder, _, _, _ = proof
    trace = read_json(folder / "source/trace.json")
    arrays, c, refs, _ = corpus[0]
    monkeypatch.setattr(module, "originals", lambda p: (trace, arrays, c, refs))
    monkeypatch.setattr(module, "verify_files", lambda *a: None)
    monkeypatch.setattr(module, "peak_rss", lambda: 128 * 1024**2)
    monkeypatch.setattr(module, "build_scene", lambda **kw: task.scene)
    monkeypatch.setattr(
        module,
        "context_from_trace",
        lambda model, canonical, t: context_from_trace(
            task.model, task.drivers["A"].task.model, t
        ),
    )
    partial = (
        tmp_path
        / "dependency/runs/v0.0.11/shared-motor-proof-01/source-71001/tape.npz.tmp"
    )
    partial.parent.mkdir(parents=True)
    with partial.open("wb") as f:
        np.savez_compressed(
            f,
            **{
                k: arrays[k]
                for k in [
                    "decision_ticks",
                    "decision_engines",
                    "raw",
                    "normalized",
                    "commands",
                    "excitation",
                    "ticks",
                ]
            },
        )
    if fault:

        def broken(*a):
            raise ValueError("fixture codec failure")

        monkeypatch.setattr(module, "encode", broken)
    p = {
        "raw_bytes": 4 * 1024**2,
        "maximum_rss_bytes": 512 * 1024**2,
        "proof_root": str(tmp_path / "dependency"),
    }
    physics_before = task.physics_total
    result = module.qualify(p, tmp_path, tmp_path / "out", "fixture", "fixture")
    assert result["new_physics"] == 0 and task.physics_total == physics_before
    if fault:
        assert result["status"] == "halted"
        assert [r["status"] for r in result["rows"]] == ["failed", "not_run", "not_run"]
        assert (tmp_path / "out/fault.json").exists()
    else:
        assert result["status"] == "passed"
        assert all(r["status"] == "passed" for r in result["rows"])
        assert (
            module.audit_qualification(
                p, tmp_path, tmp_path / "out", "fixture", "fixture"
            )["status"]
            == "passed"
        )
