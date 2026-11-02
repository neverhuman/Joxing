import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from receipt_encoding import (
    MODE,
    compact_bytes,
    decode_bootstrap,
    encode_bootstrap,
    save_report,
)
from supervise_upper_task_tests import save
from upper_task_gate_support import partition_usage


def typed(value):
    if type(value) is float:
        return ("float", value.hex())
    if type(value) is dict:
        return ("dict", sorted((k, typed(v)) for k, v in value.items()))
    if type(value) is list:
        return ("list", [typed(v) for v in value])
    return (type(value).__name__, value)


def test_encodings_preserve_all_fields_and_float_bits(tmp_path):
    value = {
        "limits": [0.0, -0.0, 5e-324, 1.7976931348623157e308, -(2**100), True, None],
        "records": [
            {"id": i, "sha": "abc123" * 12, "unicode": "拳击"} for i in range(80)
        ],
    }
    raw = compact_bytes(value)
    blob, meta = encode_bootstrap(value)
    assert blob == encode_bootstrap(value)[0] and decode_bootstrap(blob) == raw
    assert meta == {
        "encoding": MODE,
        "decoded_bytes": len(raw),
        "decoded_sha256": hashlib.sha256(raw).hexdigest(),
        "encoded_bytes": len(blob),
        "encoded_sha256": hashlib.sha256(blob).hexdigest(),
    }
    path = tmp_path / "result.json"
    save_report(path, value, mode=MODE, legacy_save=save)
    assert typed(json.loads(path.read_bytes())) == typed(value)
    reference = tmp_path / "old.json"
    save(reference, value)
    save_report(path, value, legacy_save=save)
    assert path.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize("case", ["unknown", "nonfinite", "oversize", "partial"])
def test_atomic_failures_keep_prior_valid_receipt(tmp_path, monkeypatch, case):
    path = tmp_path / "result.json"
    path.write_text('"previous valid"')
    original = Path.write_bytes

    def write(file, data):
        if file.name.endswith(".tmp") and case == "partial":
            original(file, data[:3])
            raise OSError("declared partial disk fault")
        return original(file, data)

    monkeypatch.setattr(Path, "write_bytes", write)
    value = (
        {"x": float("nan")}
        if case == "nonfinite"
        else {"x": "x" * (600000 if case == "oversize" else 1)}
    )
    with pytest.raises((ValueError, OSError)):
        save_report(
            path, value, mode="unknown" if case == "unknown" else MODE, legacy_save=save
        )
    assert path.read_text() == '"previous valid"'


@pytest.mark.parametrize(
    "case", ["truncated", "corrupt", "decoded_limit", "encode_limit"]
)
def test_bootstrap_corruption_and_bounds_fail(case):
    blob, _ = encode_bootstrap({"text": "repeat" * 1000})
    if case == "truncated":
        blob = blob[:-3]
    if case == "corrupt":
        blob = blob[:20] + bytes([blob[20] ^ 128]) + blob[21:]
    with pytest.raises((ValueError, EOFError, OSError)):
        if case == "encode_limit":
            encode_bootstrap({"x": "large"}, limit=3)
        else:
            decode_bootstrap(blob, limit=10 if case == "decoded_limit" else 1048576)


def test_gzip_bytes_stay_in_receipt_partition(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    blob, _ = encode_bootstrap({"data": [1, 2, 3]})
    (tmp_path / "deployment.json.gz").write_bytes(blob)
    (fixtures / "control.json").write_bytes(b"fixture")
    raw, receipts = partition_usage(tmp_path, fixtures)
    assert (raw, receipts) == (7, len(blob))


@pytest.mark.skipif(
    sys.platform != "linux", reason="Exact pinned prior Linux receipt inputs"
)
@pytest.mark.parametrize("kind", ["bootstrap", "main"])
def test_saved_h21_receipt_fidelity(tmp_path, kind):
    p = json.loads(
        Path("experiments/v0.0.11/four-native-context-protocol.json").read_text()
    )
    path, meta = next(
        (Path(n), v) for n, v in p["receipt_input_files"].items() if v["kind"] == kind
    )
    raw = path.read_bytes()
    assert (
        len(raw) == meta["bytes"] and hashlib.sha256(raw).hexdigest() == meta["sha256"]
    )
    value = json.loads(raw)
    if kind == "bootstrap":
        blob, encoded = encode_bootstrap(value)
        assert decode_bootstrap(blob) == raw
        (tmp_path / "roundtrip.json.gz").write_bytes(blob)
        (tmp_path / "roundtrip-summary.json").write_text(json.dumps(encoded))
    else:
        target = tmp_path / "result.json"
        save_report(target, value, mode=MODE, legacy_save=save)
        assert typed(json.loads(target.read_bytes())) == typed(value)
    assert path.read_bytes() == raw  # Original receipt was read-only.


def test_four_supervisor_dispatch_and_old_routes():
    spec = importlib.util.spec_from_file_location(
        "qualify_four_context", SCRIPTS / "qualify_deferred_trial.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.runner_module("joxing-four-native-context-v1")
        == "joxing.four_native_context"
    )
    assert (
        module.runner_module("joxing-native-context-transfer-v1")
        == "joxing.context_native_teardown"
    )


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Actual new deployment wire and registered sizing preflight",
)
def test_actual_bootstrap_and_size_preflight(tmp_path):
    p = json.loads(
        Path("experiments/v0.0.11/four-native-context-protocol.json").read_text()
    )
    prefix = Path(p["deployment_root"]) / "runs/v0.0.11/four-native-context-deployment"
    blob = prefix.with_suffix(".json.gz").read_bytes()
    metadata = json.loads(prefix.with_suffix(".encoding.json").read_text())
    raw = decode_bootstrap(blob)
    value = json.loads(raw)
    repeated, proof = encode_bootstrap(value)
    assert repeated == blob and proof == metadata
    assert (
        p["receipt_wire_encoding"] == MODE
        and value["authorized_preparation_steps"] == 800
    )
    assert not prefix.with_suffix(".json").exists()
    summary = json.loads(Path(p["receipt_projection_file"]).read_text())
    projected = (
        summary["structural_projection_bytes"]
        - summary["reference_bootstrap_gzip_bytes"]
        + len(blob)
        + summary["headroom_bytes"]
    )
    assert (
        projected <= p["receipt_bytes"] <= 4194304 - p["carry_forward"]["receipt_bytes"]
    )
    (tmp_path / "actual-size-preflight.json").write_text(
        json.dumps(
            {
                "scope": "Synthetic sizing only, not four-world evidence",
                "actual_bootstrap_bytes": len(blob),
                "decoded_sha256": metadata["decoded_sha256"],
                "projected_receipts_with_headroom": projected,
                "first_limit_bytes": p["receipt_bytes"],
                "native_work_in_control": 0,
            }
        )
    )
