"""Storage-only continuation: two unfinished cases, eight conditions,18attempts."""

import json
import subprocess
from pathlib import Path

from .boxing_audit import require
from .boxing_evidence import controller_bundles
from .frozen_scene import file_sha
from .provenance import digest
from .sensor_protocol import sensor_cases, verify_inputs

PACKET = "experiments/v0.0.11/sensor-storage-completion-protocol.json"
RUN = "runs/v0.0.11/sensor-storage-completion-01"


def verify_files(root, files, *, exact=False):
    root = Path(root)
    if exact:
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        require(actual == set(files), "Dependency file denominator")
    for name, row in files.items():
        path = root / name
        require(
            path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"],
            "Dependency checksum: " + name,
        )


def validate_packet(p):
    require(p["schema"] == "joxing-sensor-storage-completion-v1", "Completion schema")
    require(p["cases"] == sensor_cases()[6:], "Only the two unfinished cases allowed")
    require(p["bundles"] == controller_bundles(), "Frozen diagnostic controller")
    require(
        p["planned_physics"] == 317 and p["physics_cap"] == 1000,
        "Completion physics cap",
    )
    require(
        p["raw_bytes"] == 24 * 1024**2
        and p["runner_wall_s"] == 60
        and p["post_gate_wall_s"] == 120,
        "Completion raw/time caps",
    )
    require(
        p["delays_ticks"] == {"A": 1, "B": 2}
        and p["formal_simulation_workers"] == p["threads_per_process"] == 1,
        "Completion timing/workers",
    )
    require(
        p["prior_v11_physics"] == 5073
        and p["whole_v11_caps"]["physics"] == 120000
        and p["whole_v11_caps"]["raw_bytes"] == 256 * 1024**2,
        "Whole-version caps",
    )
    require(
        [r["id"] for r in p["fixed_cases"]] == [r["id"] for r in sensor_cases()[:6]],
        "Fixed prior condition selection",
    )
    require(
        p["fixed_source_commit"] == "ef42c872d13bfebb16b347f842ba11f0b8b8e776",
        "Fixed source identity",
    )


def verify_dependencies(p, root):
    verify_inputs(p, root)
    for name, sha in p["unchanged_core"].items():
        require(
            file_sha(Path(root) / name) == sha, "Non-storage semantic change: " + name
        )
    verify_files(p["fixed_run"], p["fixed_files"], exact=True)
    verify_files(p["fixed_source_root"], p["fixed_source_files"])
    verify_files(root, p["receipt_files"])
    # This source must reference the actual retained old protocol and attempts.
    prior = json.loads((Path(p["fixed_run"]) / "result.json").read_text())
    require(
        prior == p["prior_attempts"][1]["result"], "Retained partial result differs"
    )
    require(
        prior["source_commit"] == p["fixed_source_commit"]
        and prior["protocol_sha256"] == p["fixed_protocol_sha256"],
        "Prior source/protocol differs",
    )
    require(
        prior["status"] == "halted" and prior["physics_attempted"] == 2431,
        "Prior negative outcome changed",
    )
    rows = prior["rows"]
    require(
        len(rows) == 8
        and [r["status"] for r in rows]
        == ["completed"] * 6 + ["incomplete", "not_run"],
        "Prior attempt denominator",
    )
    require(
        all(r["physics_attempted"] == 0 for r in rows[6:]),
        "Previously unexecuted cases now have physics",
    )
    require(
        len(p["prior_attempts"][0]["result"]["rows"]) == 8
        and all(
            r["status"] == "not_run" for r in p["prior_attempts"][0]["result"]["rows"]
        ),
        "Original setup failure denominator",
    )
    require(
        sum(a["result"]["physics_attempted"] for a in p["prior_attempts"]) == 2431,
        "Prior attempt physics",
    )


def read_packet(root, expected):
    root = Path(root)
    p = json.loads((root / PACKET).read_text())
    require(digest(p) == expected, "Completion protocol hash")
    validate_packet(p)
    require(
        json.loads(
            subprocess.check_output(
                ["git", "show", "HEAD:" + PACKET], cwd=root, text=True
            )
        )
        == p,
        "Uncommitted completion protocol",
    )
    verify_dependencies(p, root)
    return p


def attempt_rows(p, result):
    rows = []
    for attempt in p["prior_attempts"]:
        rows.extend({"attempt": attempt["id"], **r} for r in attempt["result"]["rows"])
    rows.extend({"attempt": "storage-completion-01", **r} for r in result["rows"])
    require(len(rows) == 18, "All18attempt rows required")
    return rows


def validate_result(p, result):
    validate_packet(p)
    require(
        result["protocol_sha256"] == digest(p) and result["status"] == "completed",
        "Completion result not passed",
    )
    require(len(result["rows"]) == 2, "Completion row denominator")
    for row, case in zip(result["rows"], p["cases"], strict=True):
        require(
            all(row.get(k) == v for k, v in case.items()), "Completion row metadata"
        )
        require(
            row["status"] == "completed"
            and row["physics_attempted"]
            == row["engine_attempts_confirmed"]
            == case["physics"],
            "Incomplete/invalid completion attempt",
        )
    require(
        result["physics_attempted"] == 317
        and result["valid_boxing_games"] == result["optimizer_updates"] == 0,
        "Completion global counters",
    )
    attempt_rows(p, result)
