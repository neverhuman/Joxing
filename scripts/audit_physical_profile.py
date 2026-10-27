"""Recalculate all profiler pair, denominator and timing checks from saved data."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, write_json
from joxing.stance_train import file_hash


def audit(folder):
    report = json.loads((folder / "result.json").read_text())
    p = json.loads((folder / "protocol.json").read_text())
    assert p == json.loads(
        (ROOT / "experiments/v0.0.9/profile-protocol.json").read_text()
    )
    assert digest(p) == report["protocol_sha256"]
    source = subprocess.check_output(
        [
            "git",
            "show",
            report["code_commit"] + ":scripts/profile_physical_pipeline.py",
        ],
        cwd=ROOT,
    )
    assert hashlib.sha256(source).hexdigest() == report["script_sha256"]
    assert report["status"] == "completed" and report["boxing_games"] == 0
    assert report["threads"] == 1 and report["ipc_helper_exitcode"] == 0
    assert report["wall_s"] < p["caps"]["wall_s"]
    assert report["physics_steps"] <= p["caps"]["physics_steps"]
    assert len(report["cases"]) == len(p["seeds"]) * len(p["modes"]) == 6
    paired, physics, findings = {}, 0, []
    for row in report["cases"]:
        assert row["status"] == "completed" and row["controls"] == p["control_steps"]
        assert row["case_id"] == f"{row['seed']}:{row['mode']}"
        assert row["seed"] in p["seeds"] and row["mode"] in p["modes"]
        assert row["physics_steps"] == row["controls"] * 10
        physics += row["physics_steps"]
        case = folder / row["case_id"].replace(":", "-")
        trace = json.loads((case / "trace.json").read_text())
        assert file_hash(case / "trace.json") == row["trace_sha256"]
        assert file_hash(case / "trajectory.npz") == row["arrays_sha256"]
        assert digest(trace["initial"]) == row["parity"]["initial"]
        assert digest(trace["final"]) == row["parity"]["final"]
        assert digest(trace["telemetry"]) == row["parity"]["telemetry"]
        assert len(trace["telemetry"]) == row["controls"]
        assert all(np.isfinite(t["reward"]) for t in trace["telemetry"])
        with np.load(case / "trajectory.npz", allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        assert arrays["actions"].shape == (row["controls"], 700)
        assert arrays["observations"].shape == (row["controls"], 3601)
        assert arrays["states"].shape[0] == row["controls"]
        assert all(np.isfinite(a).all() for a in arrays.values())
        if row["seed"] in paired:
            previous, values, old_trace = paired.pop(row["seed"])
            assert previous["mode"] != row["mode"]
            assert previous["parity"] == row["parity"] and trace == old_trace
            assert all(np.array_equal(a, values[k]) for k, a in arrays.items())
            findings.append(
                {
                    "seed": row["seed"],
                    "paired_exact": True,
                    "controls_per_case": row["controls"],
                }
            )
        else:
            paired[row["seed"]] = (row, arrays, trace)
        assert all(
            len(samples) == row["controls"] and min(samples) >= 0
            for samples in row["stage_seconds"].values()
        )
        assert (
            sum(sum(samples) for samples in row["stage_seconds"].values())
            <= row["loop_s"]
        )
        assert len(row["ipc_probe_roundtrip_s"]) == p["ipc_probe_rounds"]
        assert (
            len(row["functions"]) > 0
            if row["mode"] == "profiled"
            else not row["functions"]
        )
    assert not paired and len(findings) == 3 and physics == report["physics_steps"]
    exited = json.loads((folder / "process-exit.json").read_text())
    assert (
        not exited["owned_jobs"]
        and not exited["recorded_pids_present"]
        and not exited["source_diff"]
    )
    assert exited["stages_all_zero"]
    return {
        "passed": True,
        "scope": "Independent saved-array/accounting audit, no new physics.",
        "source_commit": report["code_commit"],
        "protocol_sha256": digest(p),
        "completed": 6,
        "invalid": 0,
        "incomplete": 0,
        "not_run": 0,
        "physics_steps": physics,
        "new_audit_physics_steps": 0,
        "paired_checks": findings,
        "result_sha256": file_hash(folder / "result.json"),
        "unprofiled_median_control_steps_per_s": float(
            np.median(
                [
                    r["control_steps_per_s"]
                    for r in report["cases"]
                    if r["mode"] == "unprofiled"
                ]
            )
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    write_json(args.folder / "audit.json", result)
    print(json.dumps(result, indent=2))
