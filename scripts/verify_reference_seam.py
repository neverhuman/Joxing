"""Independently read every R1 array and recompute the permitted correction."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, write_json
from joxing.stance_train import file_hash


def verify(folder, receipts):
    r = json.loads((folder / "result.json").read_text())
    p = json.loads((folder / "protocol.json").read_text())
    assert (
        digest(p)
        == r["protocol_sha256"]
        == "1c6f405215cba2e49f18afd083311e39f0a0ad45974352b59ecec0043b5071f9"
    )
    executable = subprocess.check_output(
        ["git", "show", r["code_commit"] + ":scripts/qualify_reference_seam.py"],
        cwd=ROOT,
    )
    assert hashlib.sha256(executable).hexdigest() == r["script_sha256"]
    for path, sha in p["source_files"].items():
        assert file_hash(ROOT / path) == sha, path
    assert r["status"] == "completed" and r["reference_gate_passed"]
    assert not r["residual_training_allowed"]
    assert r["new_physics_steps"] == r["boxing_games"] == 0
    assert r["wall_s"] < p["caps"]["wall_s"]
    with np.load(ROOT / p["motion"], allow_pickle=False) as source:
        qpos = source["qpos_traj"].astype(np.float32)
        fps = float(source["framerate"])
    count, dt = len(qpos), 1 / fps
    duration = count / fps
    original = np.empty_like(qpos)
    original[:-1] = ((qpos[1:] - qpos[:-1]).astype(np.float64) / dt).astype(np.float32)
    original[-1] = ((qpos[0] - qpos[-1]).astype(np.float64) / dt).astype(np.float32)
    following = qpos[0].astype(np.float64)
    following[[0, 2]] += qpos[-1, [0, 2]].astype(np.float64)
    template = (following - qpos[-1].astype(np.float64)) / dt
    corrected = original.copy()
    corrected[-1, [0, 2]] = template[[0, 2]]
    assert (
        np.max(abs(corrected[-1].astype(np.float64) - template))
        <= p["criteria"]["template_velocity_tolerance"]
    )
    expected_schedule = [
        (cycle, mode) for cycle in p["cycles"] for mode in p["interfaces"]
    ]
    assert [(x["cycle"], x["interface"]) for x in r["cases"]] == expected_schedule
    findings = []
    for row in r["cases"]:
        assert row["status"] == "completed" and row["samples"] == count == 117
        path = folder / row["raw_file"]
        assert file_hash(path) == row["raw_sha256"]
        with np.load(path, allow_pickle=False) as a:
            assert set(a.files) == {
                "times",
                "frame_indices",
                "old_qpos",
                "new_qpos",
                "old_xpos",
                "new_xpos",
                "old_qvel",
                "new_qvel",
            }
            assert all(np.isfinite(a[k]).all() for k in a.files)
            times = row["cycle"] * duration + np.arange(count) * dt
            assert np.array_equal(a["times"], times)
            frames = np.clip(
                np.rint(np.mod(times, duration) * fps).astype(int), 0, count - 1
            )
            assert np.array_equal(a["frame_indices"], frames)
            for key in ("qpos", "xpos"):
                assert a["old_" + key].dtype == a["new_" + key].dtype == np.float32
                assert np.array_equal(a["old_" + key], a["new_" + key])
            assert a["old_qpos"].shape == a["new_qpos"].shape == (count, 85)
            assert a["old_qvel"].dtype == a["new_qvel"].dtype == np.float32
            assert np.array_equal(a["old_qvel"], original[frames])
            assert np.array_equal(a["new_qvel"], corrected[frames])
            changed = np.argwhere(a["old_qvel"] != a["new_qvel"])
            assert len(changed) == row["changed_velocity_values"]
            assert all(frames[i] == count - 1 and j in (0, 2) for i, j in changed)
            assert row["all_positions_exact"] and row["all_nonseam_velocities_exact"]
        findings.append(
            {
                "cycle": row["cycle"],
                "interface": row["interface"],
                "samples": count,
                "changed_values": len(changed),
                "raw_verified": True,
            }
        )
    phases = np.arange(int(duration // 0.02)) * 0.02
    assert r["native_reset_grid"]["phases_s"] == phases.tolist()
    frames = np.clip(np.rint(phases * fps).astype(int), 0, count - 1)
    assert len(phases) == 58 and np.array_equal(original[frames], corrected[frames])
    assert r["native_reset_grid"]["all_qpos_xpos_qvel_exact"]
    total = sum(
        f.stat().st_size
        for f in folder.rglob("*")
        if f.is_file() and f.name != "audit.json"
    )
    assert total < p["caps"]["raw_bytes"]
    exit_receipt = json.loads((receipts / "process-exit.json").read_text())
    assert not exit_receipt["owned_processes"] and not exit_receipt["source_diff"]
    assert all(not x["exists"] for x in exit_receipt["recorded_processes"])
    supervisor = json.loads((receipts / "supervisor.json").read_text())
    assert supervisor["status"] == "completed" and all(
        x["returncode"] == 0 for x in supervisor["stages"]
    )
    assert "171 passed" in (receipts / "tests.txt").read_text()
    return {
        "passed": True,
        "source_commit": r["code_commit"],
        "auditor_sha256": file_hash(Path(__file__)),
        "scope": "Every saved time/frame/position/velocity array read; raw motion independently reconstructs all legacy/corrected velocity templates and native reset-grid parity. Position equality/dtypes verified directly; no actor, FK or physics rerun. Frozen executable asserts legacy object immutability.",
        "case_counts": dict(Counter(x["status"] for x in r["cases"])),
        "samples": sum(x["samples"] for x in findings),
        "invalid": 0,
        "incomplete": 0,
        "not_run": 0,
        "raw_bytes": total,
        "new_audit_physics_steps": 0,
        "reference_gate_passed": True,
        "training_authorized_by_this_gate": False,
        "findings": findings,
        "result_sha256": file_hash(folder / "result.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.folder, args.receipts)
    write_json(args.folder / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, indent=2))
