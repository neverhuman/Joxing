"""Independent D1 saved-array audit; no teacher inference or physics rerun."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, write_json
from joxing.stance_train import file_hash

PROTOCOL_SHA = "136427ced9b436e6ff3ed13b2859e618db4d17f4e85c816480dacc8f914200d1"


def verify(folder, receipts):
    r = json.loads((folder / "result.json").read_text())
    p = json.loads((folder / "protocol.json").read_text())
    assert digest(p) == r["protocol_sha256"] == PROTOCOL_SHA
    executable = subprocess.check_output(
        ["git", "show", r["code_commit"] + ":scripts/qualify_motor_residual.py"],
        cwd=ROOT,
    )
    assert hashlib.sha256(executable).hexdigest() == r["script_sha256"]
    for path, sha in p["source_files"].items():
        assert file_hash(ROOT / path) == sha, path
    assert r["status"] == "completed" and r["interface_gate_passed"]
    assert not r["training_allowed"]
    assert r["optimizer_updates"] == r["boxing_games"] == 0
    assert r["physics_accounting_exact"] and r["threads"] == 1
    assert r["wall_s"] < p["caps"]["wall_s"]
    bundle = r["bundle"]
    assert digest(bundle) == r["bundle_hash"]
    assert bundle["reference_variant"] == p["reference_variant"]
    assert bundle["reward"] == p["reward_id"]
    assert bundle["residual_amplitude"] == p["residual_amplitude"]
    for key in ("actor_hash", "normalizer_hash", "decoder_amplitude"):
        assert (
            bundle["teacher_actor_hash" if key == "actor_hash" else key]
            == p["teacher"][key]
        )
    groups = r["groups"]
    assert len(groups) == 138 and all(groups)
    assert sorted(i for g in groups for i in g) == list(range(700))
    assert digest(groups) == bundle["groups_hash"]
    names = bundle["muscle_names_by_group"]
    assert [len(g) for g in groups] == [len(g) for g in names]
    assert len({name for group in names for name in group}) == 700
    by_name = {name: k for k, group in enumerate(names) for name in group}
    for name in p["independent_arm_pairs"]:
        pair = [by_name[name + "_r"], by_name[name + "_l"]]
        assert pair[0] != pair[1] and pair == r["independent_arm_pairs"][name]
    base = ROOT / "artifacts/v0.0.8/physical/native_2ms"
    source = json.loads((base / "replay.json").read_text())
    old_trace = json.loads((base / "trace-74000.json").read_text())
    with np.load(base / "actions-74000.npz", allow_pickle=False) as a:
        commands = a["actions"].copy()
    schedule = [
        (point, mode)
        for point in ("boundary", "mid_control")
        for mode in ("recorded_actions", "closed_loop")
    ]
    assert [(x["snapshot"], x["path"]) for x in r["cases"]] == schedule
    findings, reference_trace = [], None
    total_physics = 0
    for row in r["cases"]:
        assert row["status"] == "completed"
        stem = row["snapshot"] + "-" + row["path"]
        start = json.loads((folder / (stem + "-start.json")).read_text())
        final = json.loads((folder / (stem + "-final.json")).read_text())
        physical = source[
            "snapshot" if row["snapshot"] == "boundary" else "mid_snapshot"
        ]
        assert start["physical"] == physical
        assert final["physical"] == source["final_snapshot"]
        assert (
            file_hash(folder / (stem + "-final.json")) == row["final_snapshot_sha256"]
        )
        for snapshot in (start, final):
            assert snapshot["schema"] == "joxing-motor-residual-snapshot-v1"
            assert snapshot["bundle_hash"] == r["bundle_hash"]
            assert snapshot["episode_seed"] == 74000
            assert snapshot["reference_dtypes"] == {
                "qpos_ref": "<f4",
                "xpos_ref": "<f4",
                "qpos_ref_future": "<f8",
            }
            for key, dtype in snapshot["reference_dtypes"].items():
                values = snapshot["physical"]["application"][key]
                assert np.array_equal(
                    np.asarray(values, dtype=dtype).astype(np.float64), values
                )
        assert final["held"] is None and final["physical"]["done"]
        control = physical["physical_control"]
        first = control["completed_controls"] * 10 + control["substep"]
        assert first == row["first_source_substep"]
        if control["pending"]:
            held = start["held"]
            assert held["residual"] == [0.0] * 138
            assert held["teacher"] == held["command"] == control["action"]
            assert held["start_s"] == control["control_start_s"]
        else:
            assert start["held"] is None
        with np.load(folder / (stem + "-actions.npz"), allow_pickle=False) as a:
            assert a.files == ["actions"] and a["actions"].dtype == np.float32
            assert np.array_equal(
                a["actions"], commands[source["continuation_start"] :]
            )
        with (
            np.load(base / "substeps-74000.npz", allow_pickle=False) as a,
            np.load(folder / (stem + "-substeps.npz"), allow_pickle=False) as b,
        ):
            assert set(a.files) == set(b.files)
            offset = int(a["contact_offsets"][first])
            for key in a.files:
                expected = (
                    a[key][first:] - offset
                    if key == "contact_offsets"
                    else a[key][first:]
                    if key in {"intervals_s", "loads_N", "tilt_drop"}
                    else a[key][offset:]
                )
                assert expected.dtype == b[key].dtype and np.array_equal(
                    expected, b[key]
                ), key
            physics = len(b["intervals_s"])
        trace = json.loads((folder / (stem + "-trace.json")).read_text())
        assert len(trace) == row["controls"] == row["controls_completed"] == 260
        assert [t["source_control_index"] for t in trace] == list(range(10, 270))
        for t in trace:
            index, telemetry = t["source_control_index"], t["telemetry"]
            for key, expected in old_trace[index].items():
                if key not in {
                    "normalizer_clipped_fraction",
                    "mean_absolute_action_change",
                }:
                    assert telemetry[key] == expected, (stem, index, key)
            begin, end = old_trace[index - 1]["time_s"], telemetry["time_s"]
            failure = telemetry["physical_failure_s"]
            assert failure is None or begin <= failure <= end
            alive = max(0.0, min(end, end if failure is None else failure) - begin)
            assert t["motor_reward"] == alive - float(failure is not None)
            assert t["terminated"] == (failure is not None)
            assert not t["truncated"] and np.isfinite(t["native_reward"])
        if reference_trace is None:
            reference_trace = trace
        else:
            assert (
                trace == reference_trace
            )  # includes new native reward, no invented v8 oracle
        assert physics == row["physics_steps"] == (2600 if first == 100 else 2597)
        total_physics += physics
        findings.append(
            {
                "snapshot": row["snapshot"],
                "path": row["path"],
                "controls": len(trace),
                "physics_steps": physics,
                "raw_verified": True,
            }
        )
    assert (
        total_physics
        == r["physics_steps"]
        == p["denominator"]["expected_physics_steps"]
        == 10394
    )
    assert total_physics < p["caps"]["physics_steps"]
    raw_bytes = sum(
        f.stat().st_size
        for f in folder.rglob("*")
        if f.is_file() and f.name != "audit.json"
    )
    assert raw_bytes < p["caps"]["raw_bytes"]
    exited = json.loads((receipts / "process-exit.json").read_text())
    assert exited["source_commit"] == r["code_commit"]
    assert not exited["source_diff"] and not exited["owned_processes"]
    assert all(not x["exists"] for x in exited["recorded_processes"])
    supervisor = json.loads((receipts / "supervisor.json").read_text())
    assert (
        supervisor["status"] == "completed"
        and supervisor["source_commit"] == r["code_commit"]
    )
    assert [x["stage"] for x in supervisor["stages"]] == ["sync", "tests", "D1"]
    assert all(x["returncode"] == 0 for x in supervisor["stages"])
    assert "178 passed" in (receipts / "tests.txt").read_text()
    return {
        "passed": True,
        "source_commit": r["code_commit"],
        "auditor_sha256": file_hash(Path(__file__)),
        "result_sha256": file_hash(folder / "result.json"),
        "scope": "All four actual typed starts/finals,700-command tapes, every signed contact/load/posture tail and control telemetry independently compared with v8. Motor reward/termination reconstructed; native rewards compared across four paths only. Frozen executable asserts per-control teacher clipping, held no-new-decision and bundle immutability; masks/terminal ownership are unit-tested, not persisted per-control by D1. No actor or physics rerun.",
        "case_counts": dict(Counter(x["status"] for x in r["cases"])),
        "invalid": 0,
        "incomplete": 0,
        "not_run": 0,
        "physics_steps": total_physics,
        "new_audit_physics_steps": 0,
        "raw_bytes": raw_bytes,
        "interface_gate_passed": True,
        "training_authorized_by_this_gate": False,
        "findings": findings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.folder, args.receipts)
    write_json(args.folder / "audit.json", result)
    print(json.dumps(result, indent=2))
