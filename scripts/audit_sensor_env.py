"""Independent whole-run sensor proof audit; zero new physics."""

import argparse
import json
from pathlib import Path

from joxing.boxing_audit import require
from joxing.frozen_scene import file_sha
from joxing.motor_storage import RawCredit, write_json
from joxing.sensor_audit import compare_sensor_replay, compare_zero_delay_oracle
from joxing.sensor_physics_audit import audit_sensor_physics
from joxing.sensor_protocol import load_scene, validate_rows, verify_inputs


def audit_run(out):
    out = Path(out)
    p = json.loads((out / "protocol.json").read_text())
    result = json.loads((out / "result.json").read_text())
    validate_rows(p, result)  # Reject hidden/invalid rows before model load.
    root = Path(__file__).resolve().parents[1]
    verify_inputs(p, root)
    manifest = json.loads((out / "raw-manifest.json").read_text())
    actual = {
        str(f.relative_to(out))
        for f in out.rglob("*")
        if f.is_file()
        and f.name not in ["result.json", "raw-manifest.json", "independent-audit.json"]
    }
    require(actual == set(manifest["files"]), "Raw file denominator")
    for n, row in manifest["files"].items():
        require(
            (out / n).stat().st_size == row["bytes"]
            and file_sha(out / n) == row["sha256"],
            "Raw checksum: " + n,
        )
    require(
        manifest["bytes"] == sum(r["bytes"] for r in manifest["files"].values()),
        "Raw byte denominator",
    )
    require(
        sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) <= p["raw_bytes"],
        "Raw cap exceeded",
    )
    scene = load_scene(p, root)
    report = {"status": "running", "cases": [], "replays": [], "new_physics": 0}
    for c in p["cases"]:
        folder = out / c["id"]
        contract = json.loads((folder / "contract.json").read_text())
        require(
            contract["sensor_schedule"] == c["sensor_schedule"]
            and contract["bundles"] == p["bundles"]
            and contract["horizon_ticks"] == c["horizon_ticks"],
            "Per-case contract differs",
        )
        audit = audit_sensor_physics(
            folder, scene, expected_physics=c["physics"], delays=p["delays_ticks"]
        )
        report["cases"].append({"id": c["id"], **audit})
        final = json.loads((folder / "sensor-final.json").read_text())["base"][
            "application"
        ]
        require(
            final["tick"] == c["end_tick"] and final["done"] and not final["truncated"],
            "Registered global bell differs",
        )
        if "parent" in c:
            report["replays"].append(
                {
                    "id": c["id"],
                    **compare_sensor_replay(
                        out / c["parent"],
                        folder,
                        source_checkpoint=out
                        / c["parent"]
                        / (c["checkpoint"] + ".json"),
                    ),
                }
            )
    report["oracle"] = compare_zero_delay_oracle(
        Path(p["oracle_root"]) / "source-90101", out / "zero-90101"
    )
    report.update(
        status="passed",
        all_attempt_rows=8,
        physics_attempted=result["physics_attempted"],
        formal_sensor_correctness_passed=True,
        valid_boxing_games=0,
        optimizer_updates=0,
        scope="Bounded causal-input correctness; no motor, sporting, learning or whole-v11qualification",
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads((args.run / "protocol.json").read_text())
    credit = RawCredit(args.run, p["raw_bytes"])
    try:
        report = audit_run(args.run)
    except Exception as error:  # noqa: BLE001 -- preserve negative audit outcome
        report = {
            "status": "failed",
            "reason": type(error).__name__ + ": " + str(error),
            "new_physics": 0,
        }
    with credit.emergency():
        write_json(args.run / "independent-audit.json", report)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
