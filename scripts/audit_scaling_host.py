"""Independently verify saved source-v8 final snapshots and contact tails."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, write_json
from joxing.stance_train import file_hash


def audit(folder):
    result = json.loads((folder / "result.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    assert result["protocol_sha256"] == digest(protocol)
    source = subprocess.check_output(
        ["git", "show", result["code_commit"] + ":scripts/qualify_scaling_host.py"],
        cwd=ROOT,
    )
    assert hashlib.sha256(source).hexdigest() == result["script_sha256"]
    for path, expected in protocol["reference_files"].items():
        assert file_hash(ROOT / path) == expected
    assert (
        result["passed"] and result["status"] == "completed" and result["threads"] == 1
    )
    assert [(r["snapshot"], r["path"]) for r in result["cases"]] == [
        (point, mode)
        for point in ("boundary", "mid_control")
        for mode in ("recorded_actions", "closed_loop")
    ]
    base = ROOT / "artifacts/v0.0.8/physical/native_2ms"
    expected_snapshot = json.loads((base / "replay.json").read_text())["final_snapshot"]
    findings = []
    for row in result["cases"]:
        assert row["status"] == "completed"
        stem = row["snapshot"] + "-" + row["path"]
        final = folder / (stem + "-final.json")
        raw = folder / (stem + ".npz")
        assert file_hash(final) == row["final_snapshot_sha256"]
        assert json.loads(final.read_text()) == expected_snapshot
        assert file_hash(raw) == row["raw_substeps_sha256"]
        with (
            np.load(base / "substeps-74000.npz", allow_pickle=False) as reference,
            np.load(raw, allow_pickle=False) as actual,
        ):
            start = row["first_source_substep"]
            offset = int(reference["contact_offsets"][start])
            assert set(reference.files) == set(actual.files)
            for key in reference.files:
                values = (
                    reference[key][start:] - offset
                    if key == "contact_offsets"
                    else reference[key][start:]
                    if key in {"intervals_s", "loads_N", "tilt_drop"}
                    else reference[key][offset:]
                )
                assert values.dtype == actual[key].dtype and np.array_equal(
                    values, actual[key]
                )
            assert len(actual["intervals_s"]) == row["physics_steps"]
        findings.append(
            {
                "snapshot": row["snapshot"],
                "path": row["path"],
                "physics_steps": row["physics_steps"],
                "snapshot_contact_tail_exact": True,
            }
        )
    assert (
        sum(row["physics_steps"] for row in findings)
        == result["physics_steps"]
        == 10394
    )
    assert result["wall_s"] < protocol["qualification_caps"]["wall_s"]
    return {
        "passed": True,
        "source_commit": result["code_commit"],
        "auditor_sha256": file_hash(Path(__file__)),
        "scope": "Independent saved final-snapshot/contact-tail audit. Frozen executable assertions verify closed-loop source actions; no second actor/physics execution.",
        "completed": 4,
        "invalid": 0,
        "incomplete": 0,
        "not_run": 0,
        "source_physics_steps": 10394,
        "new_audit_physics_steps": 0,
        "findings": findings,
        "result_sha256": file_hash(folder / "result.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    write_json(args.folder / "audit.json", result)
    print(json.dumps(result, indent=2))
