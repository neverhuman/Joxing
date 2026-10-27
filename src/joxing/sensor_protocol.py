"""Exact finite D1 case schedule and source/input/result checks."""

import json
import subprocess
from pathlib import Path

from .boxing_audit import require
from .boxing_evidence import controller_bundles
from .frozen_scene import file_sha, load_frozen_scene
from .provenance import digest

PACKET = "experiments/v0.0.11/sensor-proof-protocol.json"


def sensor_cases():
    result = []
    for prefix, seed, schedule, end, mid in [
        ("zero", 90101, {"A": [0], "B": [0]}, 1000, 203),
        ("lag", 90201, {"A": [20], "B": [40]}, 200, 83),
        ("reorder", 90202, {"A": [40, 0], "B": [None, 20, 0]}, 200, 83),
        ("drop", 90203, {"A": [None], "B": [40, 0]}, 200, 83),
    ]:
        base = {
            "seed": seed,
            "sensor_schedule": schedule,
            "horizon_ticks": end,
            "end_tick": end,
            "scenario": "native-keyframe-v1",
        }
        source_id = f"{prefix}-{seed}"
        result.append(
            base
            | {"id": source_id, "start_tick": 0, "physics": end, "save_mid_tick": mid}
        )
        result.append(
            base
            | {
                "id": f"{prefix}-mid-{seed}",
                "parent": source_id,
                "checkpoint": "mid",
                "start_tick": mid,
                "physics": end - mid,
            }
        )
    return result


def verify_inputs(p, root):
    root = Path(root)
    for name, sha in {**p["source_files"], **p["upstream_files"]}.items():
        require(file_sha(root / name) == sha, "Frozen source drift: " + name)
    model = p["model"]
    for name, sha in [
        (model["binary"], model["binary_sha256"]),
        (model["report"], model["report_sha256"]),
    ]:
        require(file_sha(root / name) == sha, "Frozen model/report input drift")
    for name, row in p["oracle_files"].items():
        path = Path(p["oracle_root"]) / name
        require(
            path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"],
            "Retained D0 oracle drift: " + name,
        )


def load_scene(p, root):
    m = p["model"]
    return load_frozen_scene(
        Path(root) / m["binary"],
        Path(root) / m["report"],
        binary_sha256=m["binary_sha256"],
        report_sha256=m["report_sha256"],
        scene_hash=m["scene_hash"],
    )


def validate_packet(p):
    require(
        p["schema"] == "joxing-sensor-proof-protocol-v1"
        and p["cases"] == sensor_cases(),
        "Foreign sensor case schedule",
    )
    require(
        p["bundles"] == controller_bundles(), "Diagnostic controller identity drift"
    )
    require(
        p["planned_physics"] == sum(r["physics"] for r in sensor_cases()) == 2748,
        "Sensor physics denominator",
    )
    require(
        p["delays_ticks"] == {"A": 1, "B": 2}
        and p["physics_cap"] == 10000
        and p["raw_bytes"] == 96 * 1024**2,
        "Sensor hard caps/timing changed",
    )


def read_packet(root, expected, packet_path=PACKET):
    root = Path(root)
    p = json.loads((root / packet_path).read_text())
    require(digest(p) == expected, "Sensor protocol identity")
    validate_packet(p)
    require(
        json.loads(
            subprocess.check_output(
                ["git", "show", "HEAD:" + packet_path], cwd=root, text=True
            )
        )
        == p,
        "Sensor protocol is not committed",
    )
    verify_inputs(p, root)
    return p


def validate_rows(p, result):
    validate_packet(p)
    require(result["protocol_sha256"] == digest(p), "Result protocol identity")
    require(
        [r["id"] for r in result["rows"]] == [c["id"] for c in p["cases"]],
        "Missing/reordered/duplicate attempted row",
    )
    require(
        result["status"] == "completed"
        and all(r["status"] == "completed" for r in result["rows"]),
        "Invalid/incomplete attempt cannot pass",
    )
    for row, case in zip(result["rows"], p["cases"], strict=True):
        require(
            all(row.get(k) == v for k, v in case.items())
            and row["physics_attempted"]
            == row["engine_attempts_confirmed"]
            == case["physics"],
            "Row metadata/physics differs",
        )
    require(
        result["physics_attempted"] == p["planned_physics"]
        and result["optimizer_updates"] == result["valid_boxing_games"] == 0,
        "Global physics/game denominator",
    )
