import base64
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from portable_receipt_bundle import hydrate
from source_receipt_audit import audit_junit

from joxing import receipt_fixture_paths as paths


def setup(tmp_path, monkeypatch):
    root = tmp_path / "receipts"
    root.mkdir()
    name = "historical/runs/deployment.json.gz"
    data = b"exact bytes"
    (root / name).parent.mkdir(parents=True)
    (root / name).write_bytes(data)
    manifest = {
        "schema": paths.SCHEMA,
        "scope": "INPUT_ONLY_NOT_EXECUTION",
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        },
        "absent": ["historical/runs/deployment.json"],
        "prefixes": ["historical/runs/deployment"],
    }
    raw = paths.compact(manifest)
    (root / paths.MARKER).write_bytes(raw)
    monkeypatch.setenv(paths.ENV_ROOT, str(root))
    monkeypatch.setenv(paths.ENV_HASH, hashlib.sha256(raw).hexdigest())
    return root, manifest


def test_default_host_paths_unchanged(monkeypatch):
    monkeypatch.delenv(paths.ENV_ROOT, raising=False)
    monkeypatch.delenv(paths.ENV_HASH, raising=False)
    name = Path("/home/ubuntu/joxing/old/runs/value.json")
    assert paths.receipt_path(name) == name


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "traversal",
        "foreign",
        "unknown",
        "missing",
        "corrupt",
        "manifest",
        "missing_hash",
        "symlink",
        "negative",
        "schema",
    ],
)
def test_portable_paths_fail_closed_without_host_fallback(tmp_path, monkeypatch, fault):
    root, manifest = setup(tmp_path, monkeypatch)
    logical = "/home/ubuntu/joxing/historical/runs/deployment"
    if fault == "traversal":
        logical = "/home/ubuntu/joxing/historical/../other"
    elif fault == "foreign":
        logical = "/etc/passwd"
    elif fault == "unknown":
        logical = "/home/ubuntu/joxing/historical/runs/unregistered"
    elif fault == "missing":
        (root / "historical/runs/deployment.json.gz").unlink()
    elif fault == "corrupt":
        (root / "historical/runs/deployment.json.gz").write_bytes(b"other bytes")
    elif fault == "manifest":
        (root / paths.MARKER).write_text("{}")
    elif fault == "missing_hash":
        monkeypatch.delenv(paths.ENV_HASH)
    elif fault == "symlink":
        p = root / "historical/runs/deployment.json.gz"
        p.unlink()
        outside = tmp_path / "outside"
        outside.write_bytes(b"exact bytes")
        p.symlink_to(outside)
    elif fault == "negative":
        (root / "historical/runs/deployment.json").write_text("unexpected twin")
    elif fault == "schema":
        manifest["scope"] = "execution"
        raw = paths.compact(manifest)
        (root / paths.MARKER).write_bytes(raw)
        monkeypatch.setenv(paths.ENV_HASH, hashlib.sha256(raw).hexdigest())
    if fault == "none":
        p = paths.receipt_path(logical)
        assert (
            p.with_suffix(".json.gz").read_bytes() == b"exact bytes"
            and not p.with_suffix(".json").exists()
        )
        assert paths.receipt_path("/home/ubuntu/joxing/historical").is_dir()
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            paths.receipt_path(logical)


def test_complete_original_eight_file_bundle_and_absence(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            repository
            / "artifacts/v0.0.11/development/h29-receipt-fixtures-encoding.json"
        ).read_text()
    )
    target = tmp_path / "owned"
    target.mkdir()
    # Hydrator accepts only outputs inside its owner root; archive is a frozen
    # source input, copied for this isolated source test, not a new outcome.
    (target / "bundle.gz").write_bytes((repository / config["archive"]).read_bytes())
    config["archive"] = "bundle.gz"
    result = hydrate(target, config, target / "receipts")
    assert result["files"] == 8 and not result["host_aliases_created"]
    monkeypatch.setenv(paths.ENV_ROOT, str(target / "receipts"))
    monkeypatch.setenv(paths.ENV_HASH, config["manifest_sha256"])
    for name, meta in config["manifest"]["files"].items():
        raw = paths.receipt_path(paths.BASE / name).read_bytes()
        assert (
            len(raw) == meta["bytes"]
            and hashlib.sha256(raw).hexdigest() == meta["sha256"]
        )
    for name in config["manifest"]["absent"]:
        assert not paths.receipt_path(paths.BASE / name).exists()
    with pytest.raises(ValueError):
        hydrate(target, config, target / "receipts")


@pytest.mark.parametrize("fault", ["none", "archive", "member", "charge", "path"])
def test_hydration_refuses_corruption_or_unowned_output(tmp_path, fault):
    root = tmp_path / "owner"
    root.mkdir()
    data = b"input"
    name = "old/file.json"
    manifest = {
        "schema": paths.SCHEMA,
        "scope": "INPUT_ONLY_NOT_EXECUTION",
        "files": {name: {"bytes": 5, "sha256": hashlib.sha256(data).hexdigest()}},
        "absent": [],
        "prefixes": [],
    }
    raw = paths.compact(
        {"files": {name: {"data_base64": base64.b64encode(data).decode()}}}
    )
    blob = gzip.compress(raw, mtime=0)
    (root / "bundle.gz").write_bytes(blob)
    marker = paths.compact(manifest)
    c = {
        "archive": "bundle.gz",
        "encoded_bytes": len(blob),
        "encoded_sha256": hashlib.sha256(blob).hexdigest(),
        "decoded_bytes": len(raw),
        "decoded_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(marker).hexdigest(),
        "charged_bytes": len(data) + len(marker),
    }
    output = root / "receipts"
    if fault == "archive":
        (root / "bundle.gz").write_bytes(b"bad")
    elif fault == "member":
        manifest["files"]["other/file.json"] = manifest["files"].pop(name)
        c["manifest_sha256"] = hashlib.sha256(paths.compact(manifest)).hexdigest()
    elif fault == "charge":
        c["charged_bytes"] -= 1
    elif fault == "path":
        output = tmp_path / "outside"
    if fault == "none":
        assert hydrate(root, c, output)["files"] == 1
    else:
        with pytest.raises(ValueError):
            hydrate(root, c, output)
        assert not output.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "failed",
        "error",
        "skip",
        "missing",
        "duplicate",
        "malformed",
        "false_rc",
        "gzip",
        "both_bad",
    ],
)
def test_raw_failed_and_gzip_source_audit_preserves_every_case(tmp_path, fault):
    nodes = ["tests/example.py::case_a", "tests/example.py::case_b"]
    a = '<testcase classname="tests.example" name="case_a"/>'
    b = '<testcase classname="tests.example" name="case_b"/>'
    rc = 0
    if fault in ("failed", "error", "skip"):
        tag = {"failed": "failure", "error": "error", "skip": "skipped"}[fault]
        b = f'<testcase classname="tests.example" name="case_b"><{tag}/></testcase>'
        rc = 1
    elif fault == "missing":
        b = ""
    elif fault == "duplicate":
        b = a
    elif fault == "false_rc":
        rc = 2
    raw = ("<testsuite>" + a + b + "</testsuite>").encode()
    if fault == "malformed":
        raw = b"<cutoff"
    if fault == "gzip":
        (tmp_path / "controls.xml.gz").write_bytes(gzip.compress(raw))
    else:
        (tmp_path / "controls.xml").write_bytes(raw)
    if fault == "both_bad":
        (tmp_path / "controls.xml.gz").write_bytes(gzip.compress(b"<different/>"))
    report = audit_junit(tmp_path, nodes, rc)
    assert len(report["rows"]) == 2
    assert report["qualified"] is (fault in ("none", "gzip"))
    if fault in ("failed", "error"):
        assert report["counts"]["failed"] == 1 and report["complete"]
        assert report["rows"][1]["failure_kind"] == (
            "error" if fault == "error" else "failure"
        )
    if fault in ("missing", "duplicate", "malformed", "both_bad"):
        assert not report["complete"] and all(
            r["status"] == "unconfirmed" for r in report["rows"]
        )


@pytest.mark.parametrize(
    "fault", ["none", "schema", "physics", "whole", "workers", "wire"]
)
def test_source_only_entry_rejects_other_profiles_before_work(fault):
    from qualify_linux_fixture_readiness import SCHEMA, validate

    p = {
        "schema": SCHEMA,
        "whole_wall_s": 120,
        "setup_wall_s": 60,
        "control_wall_s": 30,
        "audit_wall_s": 15,
        "physics_limit": 0,
        "simulation_workers": 1,
        "threads_per_process": 1,
        "process_ceiling": 12,
        "owned_rss_bytes": 8 * 1024**3,
        "fixture_bytes": 8 * 1024**2,
        "receipt_bytes": 1024**2,
        "receipt_wire_encoding": "gzip-main-stdout-bootstrap-v2",
    }
    if fault == "schema":
        p["schema"] = "old"
    elif fault == "physics":
        p["physics_limit"] = 1
    elif fault == "whole":
        p["whole_wall_s"] = 121
    elif fault == "workers":
        p["simulation_workers"] = 2
    elif fault == "wire":
        p["receipt_wire_encoding"] = "legacy"
    if fault == "none":
        validate(p)
    else:
        with pytest.raises(ValueError):
            validate(p)
