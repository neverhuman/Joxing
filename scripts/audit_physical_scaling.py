"""Independent completed/partial H2 accounting from immutable worker receipts."""

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, write_json
from joxing.stance_train import file_hash

ACTOR = "963d2f56c5887f79e094ddf998316730475f979efb100073237cda81817e272e"
NORMALIZER = "34f38e6e87544d77b87b9a155fe6b9996629fd671f235b773c27f004d6fd5df8"
CORE = "6e2579711ac7806777db42c83971b028fd1046f2f2fffa13792bd978601bc8a7"
STATUSES = {"completed", "invalid", "incomplete", "not_run"}


def trajectory_identity(row):
    return {
        k: row[k]
        for k in (
            "seed",
            "controls",
            "physics_steps",
            "resets",
            "initial_hash",
            "final_hash",
            "step_hashes",
            "action_hash",
        )
    }


def check_case(row, protocol):
    assert row["status"] in STATUSES
    assert row["seed"] in protocol["seeds"]
    assert 0 <= row.get("physics_steps", 0) <= protocol["control_steps"] * 10
    if row["status"] == "completed":
        assert row["controls"] == protocol["control_steps"]
        assert row["physics_steps"] == row["controls"] * 10
        assert len(row["step_hashes"]) == row["controls"]
        for value in row["step_hashes"] + [
            row[k] for k in ("initial_hash", "final_hash", "action_hash")
        ]:
            assert re.fullmatch("[0-9a-f]{64}", value)
        assert row["resets"] == [
            row["seed"] + 1000000 * i for i in range(len(row["resets"]))
        ]
        assert row["resets"] and row["loop_s"] > 0
        assert math.isclose(
            row["control_steps_per_s"], row["controls"] / row["loop_s"], rel_tol=1e-12
        )
    else:
        assert row.get("reason")


def check_registry(report, protocol):
    """Reconstruct schedule independently; preserve qualification and allocation faults."""
    anchors, waves = report["anchors"], report["waves"]
    assert [r["case_id"] for r in anchors] == [f"anchor:{s}" for s in protocol["seeds"]]
    expected = [
        (rep, n)
        for rep in range(protocol["repeats"])
        for n in (protocol["workers"] if rep % 2 == 0 else protocol["workers"][::-1])
    ]
    assert [(w["repeat"], w["workers"]) for w in waves] == expected
    all_cases = anchors + [r for w in waves for r in w["cases"]]
    assert len(all_cases) == len({r["case_id"] for r in all_cases}) == 78
    assert len(anchors) == 16 and sum(len(w["cases"]) for w in waves) == 62
    for row in all_cases:
        check_case(row, protocol)
    references = {r["seed"]: r for r in anchors if r["status"] == "completed"}
    qualified, throughput = 0, []
    for wave in waves:
        assert wave["status"] in STATUSES
        assert wave["wave_id"] == f"{wave['repeat']}:{wave['workers']}"
        assert [r["case_id"] for r in wave["cases"]] == [
            f"{wave['wave_id']}:{s}" for s in protocol["seeds"][: wave["workers"]]
        ]
        for row in wave["cases"]:
            if row["status"] == "completed":
                assert row["seed"] in references
                assert trajectory_identity(row) == trajectory_identity(
                    references[row["seed"]]
                )
                qualified += 1
        if wave["status"] == "completed":
            assert all(
                r["status"] == "completed" and r["serial_anchor_exact"]
                for r in wave["cases"]
            )
            assert wave["controls"] == wave["workers"] * protocol["control_steps"]
            assert 0 < wave["steady_parent_s"] <= wave["startup_inclusive_s"]
            for key, timing in [
                ("steady_control_steps_per_s", "steady_parent_s"),
                ("startup_inclusive_control_steps_per_s", "startup_inclusive_s"),
            ]:
                assert math.isclose(
                    wave[key], wave["controls"] / wave[timing], rel_tol=1e-12
                )
            required = (
                protocol["worker_memory_reservation_bytes"] * wave["workers"]
                + protocol["caps"]["minimum_memory_bytes"]
            )
            assert wave["capacity_before"]["available_memory_bytes"] >= required
            throughput.append(
                {
                    k: wave[k]
                    for k in (
                        "wave_id",
                        "workers",
                        "repeat",
                        "steady_control_steps_per_s",
                        "startup_inclusive_control_steps_per_s",
                    )
                }
            )
        else:
            assert wave.get("reason")
    actual = sum(r.get("physics_steps", 0) for r in all_cases)
    assert (
        actual
        == report["physics_steps"]
        <= report["maximum_allocated_physics_steps"]
        <= protocol["caps"]["physics_steps"]
    )
    assert report["maximum_allocated_physics_steps"] == 93600
    return {
        "case_counts": dict(Counter(r["status"] for r in all_cases)),
        "anchor_counts": dict(Counter(r["status"] for r in anchors)),
        "wave_case_counts": dict(
            Counter(r["status"] for w in waves for r in w["cases"])
        ),
        "wave_counts": dict(Counter(w["status"] for w in waves)),
        "parallel_completed_cases_matching_serial": qualified,
        "physics_steps": actual,
        "samples": throughput,
    }


def audit(folder):
    report = json.loads((folder / "result.json").read_text())
    p = json.loads((folder / "protocol.json").read_text())
    assert p == json.loads(
        (ROOT / "experiments/v0.0.9/scaling-protocol.json").read_text()
    )
    assert digest(p) == report["protocol_sha256"]
    source = subprocess.check_output(
        [
            "git",
            "show",
            report["code_commit"] + ":scripts/benchmark_physical_scaling.py",
        ],
        cwd=ROOT,
    )
    assert hashlib.sha256(source).hexdigest() == report["script_sha256"]
    assert report["source_hash"] == CORE and report["boxing_games"] == 0
    assert report["status"] in {"completed", "halted"}
    assert report["wall_s"] < p["caps"]["wall_s"]
    findings = check_registry(report, p)
    cases = {
        r["case_id"]: r
        for r in report["anchors"] + [r for w in report["waves"] for r in w["cases"]]
    }
    private, seen, exact = [], set(), True
    for path in sorted(folder.glob("*-worker-*.json")):
        value = json.loads(path.read_text())
        exact &= value["phase"] in {"completed", "failed"}
        if "actor_hash" in value:
            assert (
                value["actor_hash"] == ACTOR
                and value["normalizer_stats_hash"] == NORMALIZER
            )
        if value["phase"] == "completed":
            assert value["threads"] == 1 and all(
                r["status"] == "completed" for r in value["cases"]
            )
        assert value["physics_steps"] == sum(
            r.get("physics_steps", 0) for r in value["cases"]
        )
        for row in value["cases"]:
            assert row["case_id"] not in seen
            seen.add(row["case_id"])
            target = cases[row["case_id"]]
            assert target.get("physics_steps", 0) == row.get("physics_steps", 0)
            if row["status"] == "completed":
                assert trajectory_identity(target) == trajectory_identity(row)
        private.append(
            {
                "file": path.name,
                "pid": value["pid"],
                "phase": value["phase"],
                "physics_steps": value["physics_steps"],
                "peak_rss_KiB": value["peak_rss_KiB"],
            }
        )
    assert exact == report["physics_accounting_exact"]
    assert all(r["case_id"] in seen for r in cases.values() if r["status"] != "not_run")
    assert sum(r["physics_steps"] for r in private) == findings["physics_steps"]
    exited = json.loads((folder / "process-exit.json").read_text())
    assert (
        not exited["owned_jobs"]
        and not exited["recorded_pids_present"]
        and not exited["source_diff"]
    )
    full = (folder / "linux-full-tests.txt").read_text()
    test_count = re.search(r"(\d+) passed", full)
    assert test_count and int(test_count.group(1)) >= 146 and "failed" not in full
    summary = []
    for workers in p["workers"]:
        rows = [r for r in findings["samples"] if r["workers"] == workers]
        summary.append(
            {
                "workers": workers,
                "completed_repeats": len(rows),
                "planned_repeats": p["repeats"],
                "median_steady_control_steps_per_s": float(
                    np.median([r["steady_control_steps_per_s"] for r in rows])
                )
                if rows
                else None,
                "median_startup_inclusive_control_steps_per_s": float(
                    np.median(
                        [r["startup_inclusive_control_steps_per_s"] for r in rows]
                    )
                )
                if rows
                else None,
            }
        )
    return {
        "passed": True,
        "scope": "Independent persisted hash/accounting audit; no new physics. Hash equality on unchanged v8 core, not independent raw-tape integration.",
        "source_commit": report["code_commit"],
        "auditor_sha256": file_hash(Path(__file__)),
        "result_sha256": file_hash(folder / "result.json"),
        "protocol_sha256": digest(p),
        "runner_status": report["status"],
        "runner_error": report.get("error"),
        "physics_accounting_exact": exact,
        "linux_tests_passed": int(test_count.group(1)),
        "maximum_allocated_physics_steps": report["maximum_allocated_physics_steps"],
        **findings,
        "summary": summary,
        "worker_receipts": private,
        "all_scaling_cases_completed": report["status"] == "completed",
        "boxing_games": 0,
        "new_audit_physics_steps": 0,
        "limitations": [
            "Short120-control native walking workload with audit hashes; no optimizer or referee.",
            "Capacity before a failed reservation is not retained by the old runner; its required-byte fault is retained.",
            "A single completed repeat is descriptive only; no two-repeat median inference or training-speed claim.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    write_json(args.folder / "audit.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "passed",
                    "runner_status",
                    "case_counts",
                    "wave_counts",
                    "physics_steps",
                    "summary",
                )
            },
            indent=2,
        )
    )
