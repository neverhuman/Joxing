import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from receipt_encoding import (
    MODE,
    V2,
    decode_bootstrap,
    decode_terminal,
    main_metadata,
    report_path,
    save_report,
    terminal_line,
)
from supervise_upper_task_tests import save


@pytest.mark.parametrize("mode", ["legacy", MODE, V2])
def test_exact_main_terminal_and_old_modes(tmp_path, mode):
    value = {
        "rows": [{"v": -0.0, "large": 2**80, "nested": [True, None, 1e-310]}],
        "error": None,
    }
    path = report_path(tmp_path, mode)
    save_report(path, value, mode=mode, legacy_save=save)
    if mode == V2:
        raw = decode_bootstrap(path.read_bytes())
        assert json.loads(raw) == value
        assert json.loads(raw)["rows"][0]["v"].hex() == (-0.0).hex()
        meta = main_metadata(path)
        assert meta["decoded_bytes"] == len(raw)
        line = terminal_line(value, mode, main=meta)
        assert decode_terminal(line) == (json.dumps(value) + "\n").encode()
        assert not (tmp_path / "result.json").exists()
    else:
        assert terminal_line(value, mode) == json.dumps(value)


@pytest.mark.parametrize("case", ["corrupt", "hash", "oversize", "mode", "nan"])
def test_terminal_capsule_faults_fail(case):
    if case == "nan":
        with pytest.raises(ValueError):
            terminal_line({"v": float("nan")}, V2)
        return
    c = json.loads(terminal_line({"v": 1}, V2))
    if case == "corrupt":
        c["data_base64"] = base64.b64encode(b"not gzip").decode()
    if case == "hash":
        c["decoded_sha256"] = "0" * 64
    if case == "oversize":
        c["data_base64"] = "x" * 1048577
    if case == "mode":
        c["encoding"] = "unknown"
    with pytest.raises(ValueError):
        decode_terminal(json.dumps(c))


@pytest.mark.skipif(
    sys.platform != "linux", reason="Pinned complete Linux receipt inputs"
)
@pytest.mark.parametrize("kind", ["main_h21", "terminal_h21", "main_h22"])
def test_saved_main_terminal_complete_fidelity(tmp_path, kind):
    import hashlib

    from receipt_encoding import compact_bytes

    p = json.loads(
        Path(
            "experiments/v0.0.11/resource-admitted-four-context-protocol.json"
        ).read_text()
    )
    name, meta = next(
        (n, v) for n, v in p["codec_input_files"].items() if v["kind"] == kind
    )
    raw = Path(name).read_bytes()
    assert (
        len(raw) == meta["bytes"] and hashlib.sha256(raw).hexdigest() == meta["sha256"]
    )
    value = json.loads(raw)
    if kind.startswith("terminal"):
        line = terminal_line(value, V2)
        assert decode_terminal(line) == raw
        (tmp_path / "terminal-capsule.json").write_text(line)
    else:
        dest = report_path(tmp_path, V2)
        save_report(dest, value, mode=V2, legacy_save=save)
        assert decode_bootstrap(dest.read_bytes()) == compact_bytes(value)
        assert json.loads(decode_bootstrap(dest.read_bytes())) == value


@pytest.mark.skipif(
    sys.platform != "linux", reason="Actual admitted bootstrap and receipt projection"
)
def test_actual_admitted_bootstrap_and_storage_preflight(tmp_path):
    import hashlib

    from receipt_encoding import encode_bootstrap

    p = json.loads(
        Path(
            "experiments/v0.0.11/resource-admitted-four-context-protocol.json"
        ).read_text()
    )
    prefix = (
        Path(p["deployment_root"])
        / "runs/v0.0.11/resource-admitted-four-context-deployment"
    )
    blob = prefix.with_suffix(".json.gz").read_bytes()
    raw = decode_bootstrap(blob)
    data = json.loads(raw)
    assert encode_bootstrap(data)[0] == blob
    assert data["admission"]["status"] == "admitted"
    from host_admission import assess_pair

    assert (
        assess_pair(
            data["admission"]["samples"],
            p["admission_policy"],
            data["admission"]["assessed_monotonic_s"],
        )["status"]
        == "admitted"
    )
    assert p["receipt_wire_encoding"] == V2 and not prefix.with_suffix(".json").exists()
    projection = json.loads(Path(p["receipt_projection_file"]).read_text())
    total = (
        projection["projected_except_bootstrap_bytes"]
        + len(blob)
        + projection["headroom_bytes"]
    )
    assert total <= p["receipt_bytes"] <= 4194304 - p["carry_forward"]["receipt_bytes"]
    assert (
        projection["control_transcode_except_bootstrap_bytes"]
        + len(blob)
        + projection["headroom_bytes"]
        <= p["receipt_bytes"]
    )
    (tmp_path / "actual-admitted-preflight.json").write_text(
        json.dumps(
            {
                "scope": "Synthetic sizing only; not native outcome",
                "projected_bytes": total,
                "cap": p["receipt_bytes"],
                "bootstrap_sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    )


@pytest.mark.parametrize("case", ["valid", "wrong_hash", "symlink", "partial"])
def test_owned_new_xml_exact_bytes_and_atomic_faults(tmp_path, monkeypatch, case):
    import hashlib

    from receipt_encoding import decode_gzip, pack_owned_control

    path = tmp_path / "controls.xml"
    raw = b'<testsuites><testcase name="exact" time="-0.0"/></testsuites>\n'
    path.write_bytes(raw)
    if case == "symlink":
        actual = tmp_path / "actual"
        path.rename(actual)
        path.symlink_to(actual)
    if case == "partial":
        original = Path.write_bytes

        def fail(p, data):
            if p.name.endswith(".tmp"):
                original(p, data[:3])
                raise OSError("partial declared write")
            return original(p, data)

        monkeypatch.setattr(Path, "write_bytes", fail)
    if case == "valid":
        meta = pack_owned_control(path, hashlib.sha256(raw).hexdigest())
        assert (
            decode_gzip((tmp_path / meta["path"]).read_bytes()) == raw
            and not path.exists()
        )
    else:
        with pytest.raises((ValueError, OSError)):
            pack_owned_control(
                path,
                "wrong" if case == "wrong_hash" else hashlib.sha256(raw).hexdigest(),
            )
        assert path.read_bytes() == raw
