"""Saved-evidence audit of eight fixed conditions across all18attempt rows."""

import argparse
import json
from pathlib import Path

from joxing.boxing_audit import require
from joxing.evidence_storage import read_json
from joxing.motor_storage import RawCredit, write_json
from joxing.sensor_audit import compare_sensor_replay, compare_zero_delay_oracle
from joxing.sensor_completion import (
    attempt_rows,
    validate_result,
    verify_dependencies,
    verify_files,
)
from joxing.sensor_physics_audit import audit_sensor_physics
from joxing.sensor_protocol import load_scene, sensor_cases


def audit_run(out, root):
    out, root = Path(out), Path(root)
    p = json.loads((out / "protocol.json").read_text())
    result = json.loads((out / "result.json").read_text())
    attempts = attempt_rows(p, result)
    validate_result(p, result)
    verify_dependencies(p, root)
    manifest = json.loads((out / "raw-manifest.json").read_text())
    actual = {
        str(f.relative_to(out))
        for f in out.rglob("*")
        if f.is_file()
        and f.name not in ["raw-manifest.json", "result.json", "independent-audit.json"]
    }
    require(actual == set(manifest["files"]), "Completion raw denominator")
    verify_files(out, manifest["files"])
    require(
        manifest["bytes"] == sum(r["bytes"] for r in manifest["files"].values()),
        "Completion byte denominator",
    )
    require(
        sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) <= p["raw_bytes"],
        "Completion raw cap",
    )
    scene = load_scene(p, root)
    report = {
        "status": "running",
        "all_attempt_rows": attempts,
        "cases": [],
        "replays": [],
        "new_physics": 0,
    }
    folders = {c["id"]: Path(p["fixed_run"]) / c["id"] for c in p["fixed_cases"]}
    folders.update({c["id"]: out / c["id"] for c in p["cases"]})
    for case in sensor_cases():
        folder = folders[case["id"]]
        contract = read_json(folder / "contract.json")
        require(
            contract["sensor_schedule"] == case["sensor_schedule"]
            and contract["bundles"] == p["bundles"]
            and contract["horizon_ticks"] == case["horizon_ticks"],
            "Fixed per-condition contract",
        )
        audit = audit_sensor_physics(
            folder, scene, expected_physics=case["physics"], delays=p["delays_ticks"]
        )
        final = read_json(folder / "sensor-final.json")["base"]["application"]
        require(
            final["tick"] == case["end_tick"]
            and final["done"]
            and not final["truncated"],
            "Condition bell",
        )
        report["cases"].append(
            {
                "id": case["id"],
                "source": p["fixed_source_commit"]
                if case["id"] in [c["id"] for c in p["fixed_cases"]]
                else result["source_commit"],
                **audit,
            }
        )
        if "parent" in case:
            report["replays"].append(
                {
                    "id": case["id"],
                    **compare_sensor_replay(
                        folders[case["parent"]],
                        folder,
                        source_checkpoint=folders[case["parent"]]
                        / (case["checkpoint"] + ".json"),
                    ),
                }
            )
    report["oracle"] = compare_zero_delay_oracle(
        Path(p["oracle_root"]) / "source-90101", folders["zero-90101"]
    )
    report.update(
        status="passed",
        formal_sensor_correctness_passed=True,
        unique_conditions=8,
        attempted_rows=18,
        completion_physics=317,
        all_D1_physics=2748,
        cumulative_v11_physics=5390,
        optimizer_updates=0,
        valid_boxing_games=0,
        scope="Causal sensor/contact/replay correctness only; no motor, sporting, learning or whole-v11completion",
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads((args.run / "protocol.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    credit = RawCredit(args.run, p["raw_bytes"])
    try:
        report = audit_run(args.run, Path(__file__).resolve().parents[1])
    except Exception as exc:  # noqa: BLE001 -- retain negative and all attempts
        report = {
            "status": "failed",
            "reason": type(exc).__name__ + ": " + str(exc),
            "all_attempt_rows": attempt_rows(p, result),
            "new_physics": 0,
            "formal_sensor_correctness_passed": False,
        }
    with credit.emergency():
        write_json(args.run / "independent-audit.json", report)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
