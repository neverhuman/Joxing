import json
import shutil
import struct
import zipfile

import numpy as np
import pytest

from joxing import native_tape_qualification as qualification
from joxing.motor_storage import CREDITS


@pytest.fixture
def packet(tmp_path, monkeypatch):
    original_root = qualification.ROOT
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    relative = "experiments/v0.0.11/native_source_partial_display_01.py"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    shutil.copyfile(original_root / relative, target)
    arrays = {
        "commands": np.zeros((2, 2, 3), np.float32),
        "observations": np.arange(32, dtype=np.float32).reshape(2, 2, 8),
        "decision_engines": np.arange(8, dtype=np.float64).reshape(2, 4),
        "contact_offsets": np.array([0, 1, 2], np.int64),
        "contact_geoms": np.arange(4, dtype=np.int64).reshape(2, 2),
    }
    complete = tmp_path / "complete-prefix.zip"
    with zipfile.ZipFile(complete, "w", compression=zipfile.ZIP_LZMA) as z:
        for key, a in arrays.items():
            with z.open(key + ".npy", "w") as f:
                np.lib.format.write_array(f, a, allow_pickle=False)
        start_dir = z.start_dir
    # Close writes the actual directory after the complete local members.
    prefix = tmp_path / "prefix.tmp"
    name = b"contacts.npy"
    header = struct.pack(
        "<4s5H3I2H", b"PK\x03\x04", 63, 2, 14, 0, 0, 0, 0, 0, len(name), 0
    )
    prefix.write_bytes(complete.read_bytes()[:start_dir] + header + name)
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, raw=np.arange(40, dtype=np.float64), empty=np.zeros((0, 2)))
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runs/v0.0.11/native-strike-proof-01/physics.jsonl": {"bytes": 200},
                "runs/v0.0.11/native-strike-proof-01/source/trace.json.gz": {
                    "bytes": 300
                },
            }
        )
    )
    p = {
        "raw_bytes": 4 * 1024**2,
        # rss_bytes() is the PEAK RSS of the whole pytest process
        # (getrusage ru_maxrss); earlier tests in the same run (MuJoCo scenes)
        # already exceed a fixed 512 MiB, which halted this qualification in the
        # full suite and in CI. Give it 512 MiB of headroom above that peak.
        "maximum_rss_bytes": qualification.rss_bytes() + 512 * 1024**2,
        "stage_wall_limits_s": {"proof": 20, "audit": 20},
        "maximum_ephemeral_lzma_bytes": 128 * 1024,
        "corpora": {
            "native-prefix": {
                "kind": "incomplete-native-prefix",
                "path": str(prefix),
                "members": 5,
                "payload_bytes": sum(a.nbytes for a in arrays.values()),
                "minimum_encode_speedup": 0,
            },
            "complete-source": {
                "kind": "complete-legacy-npz",
                "path": str(legacy),
                "members": 2,
                "payload_bytes": 320,
                "minimum_encode_speedup": 0,
            },
        },
        "native_preflight": {
            "raw_manifest": str(manifest),
            "archive_and_receipt_metadata_bound_bytes": 8192,
            "additional_allocations_bytes": {"tail": 1024},
            "whole_raw_cap_bytes": 30000,
            "video_variants": {"original": 2000, "small": 1000},
            "gate_variant": "small",
        },
    }
    yield p
    CREDITS.pop((tmp_path / qualification.RUN).resolve(), None)


def test_fixture_two_rows_and_independent_saved_audit_without_physics(packet):
    result = qualification.prove(packet, "synthetic-source", "synthetic-protocol")
    assert result["status"] == "passed"
    assert result["new_physics"] == 0
    assert not result["native_boxing_proof_passed"]
    assert all(row["integrity_passed"] for row in result["rows"])
    report = qualification.audit(packet, "synthetic-source", "synthetic-protocol")
    assert report["all_planned_rows_audited"]
    assert not report["candidate_reader_used"]
    assert report["new_physics"] == 0
    stored = qualification.ROOT / qualification.RUN / "native-prefix.npz"
    with np.load(stored, allow_pickle=False) as z:
        assert "contacts" not in z.files
        assert "video_states" not in z.files
        assert len(z.files) == 5


def test_failed_write_preserves_attempt_and_later_not_run(packet, monkeypatch):
    def fault(*args, **kwargs):
        raise OSError("injected qualification save failure")

    monkeypatch.setattr(qualification.storage, "store", fault)
    result = qualification.prove(packet, "synthetic-source", "synthetic-protocol")
    assert result["status"] == "halted"
    assert [r["status"] for r in result["rows"]] == ["failed", "not_run"]
    assert result["new_physics"] == 0
    assert (qualification.ROOT / qualification.RUN / "fault.json").is_file()


def test_independent_audit_rejects_changed_encoded_bytes(packet):
    qualification.prove(packet, "synthetic-source", "synthetic-protocol")
    encoded = qualification.ROOT / qualification.RUN / "native-prefix.npz"
    encoded.write_bytes(encoded.read_bytes()[:-1])
    with pytest.raises(ValueError, match="original/encoded identity"):
        qualification.audit(packet, "synthetic-source", "synthetic-protocol")


def test_original_video_can_fail_size_gate_while_named_smaller_variant_fits(packet):
    arrays = qualification.load_corpus(packet, "native-prefix")
    projected = qualification.native_projection(packet, arrays)
    small_size = projected["variants"]["small"]["whole_raw_upper_bytes"]
    packet["native_preflight"]["whole_raw_cap_bytes"] = small_size
    actual = qualification.native_projection(packet, arrays)
    assert actual["variants"]["small"]["fits"]
    assert not actual["variants"]["original"]["fits"]
    assert actual["missing_signed_contacts_remain_missing"]


def test_lzma_comparison_memory_limit_includes_seek_backpatch():
    buffer = qualification.BoundedMemory(8)
    buffer.write(b"12345678")
    buffer.seek(0)
    buffer.write(b"12")
    assert buffer.getvalue() == b"12345678"
    buffer.seek(8)
    with pytest.raises(ValueError, match="memory output cap"):
        buffer.write(b"9")
