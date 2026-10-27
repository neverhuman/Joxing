"""Independent saved-evidence D0 audit; no physics steps or outcome selection."""

import argparse
import hashlib
import json
from pathlib import Path

from joxing.bout_recording import load_recording
from joxing.boxing_audit import audit_case, compare_replay, require
from joxing.boxing_evidence import recording_cases
from joxing.motor_storage import RawCredit, write_json
from joxing.provenance import digest
from joxing.scene import build_scene


def audit_run(out):
    out = Path(out)
    p = json.loads((out / "protocol.json").read_text())
    result = json.loads((out / "result.json").read_text())
    require(result["protocol_sha256"] == digest(p), "Protocol identity")
    require(p["cases"] == recording_cases(), "Case schedule changed")
    require(
        [r["id"] for r in result["rows"]] == [r["id"] for r in p["cases"]],
        "Missing/reordered/duplicate case row",
    )
    require(
        result["status"] == "completed"
        and all(r["status"] == "completed" for r in result["rows"]),
        "Incomplete/invalid cases cannot pass",
    )
    require(
        result["physics_attempted"]
        == sum(r["physics"] for r in p["cases"])
        == p["planned_physics"],
        "Physics denominator",
    )
    require(
        result["optimizer_updates"] == result["valid_boxing_games"] == 0,
        "Diagnostic misclassified as learning/game",
    )
    manifest = json.loads((out / "raw-manifest.json").read_text())
    require(
        set(manifest["files"])
        == {
            str(f.relative_to(out))
            for f in out.rglob("*")
            if f.is_file()
            and f.name
            not in [
                "result.json",
                "raw-manifest.json",
                "independent-audit.json",
                "first-diagnostic.mp4",
                "first-diagnostic.json",
            ]
        },
        "Raw file denominator",
    )
    for name, row in manifest["files"].items():
        raw = (out / name).read_bytes()
        require(
            len(raw) == row["bytes"]
            and hashlib.sha256(raw).hexdigest() == row["sha256"],
            "Raw checksum: " + name,
        )
    require(
        manifest["bytes"] == sum(r["bytes"] for r in manifest["files"].values()),
        "Raw byte denominator",
    )
    scene = build_scene()
    report = {"status": "running", "cases": [], "replays": [], "new_physics": 0}
    for c, r in zip(p["cases"], result["rows"], strict=True):
        contract = json.loads((out / c["id"] / "contract.json").read_text())
        require(
            contract["bundles"] == p["bundles"] and contract["control_ticks"] == 20,
            "Controller/cadence differs from registration",
        )
        require(all(r[k] == v for k, v in c.items()), "Case metadata differs")
        require(r["physics_attempted"] == c["physics"], "Row attempted physics")
        audit = audit_case(
            out / c["id"],
            scene,
            expected_physics=c["physics"],
            delays=p["delays_ticks"],
        )
        report["cases"].append({"id": c["id"], **audit})
        if "parent" in c:
            initial = json.loads((out / c["id"] / "initial.json").read_text())
            checkpoint = json.loads(
                (out / c["parent"] / (c["checkpoint"] + ".json")).read_text()
            )
            require(initial == checkpoint, "Replay did not use actual source snapshot")
            report["replays"].append(
                {"id": c["id"], **compare_replay(out / c["parent"], out / c["id"])}
            )
        final = json.loads((out / c["id"] / "final.json").read_text())
        require(final["application"]["tick"] == c["end_tick"], "Case final tick")
        require(
            final["application"]["truncated"] == (c["id"] == "truncation-90103")
            and final["application"]["done"],
            "Global ending differs",
        )
        if c["id"].startswith("contact-"):
            require(audit["positive_glove_touches"] > 0, "No actual glove force")
    recording, states = load_recording(scene, out / "source-90101/recording")
    require(
        len(states) == 26
        and [r["physics_steps"] for r in recording["frames"]]
        == list(range(0, 1001, 40)),
        "Recording frame schedule",
    )
    report.update(
        status="passed",
        physics_attempted=result["physics_attempted"],
        all_attempt_rows=7,
        recorded_frames=26,
        scope="Untrained shared-world diagnostic; no valid boxing or learning",
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads((args.run / "protocol.json").read_text())
    RawCredit(args.run, p["raw_bytes"])
    try:
        report = audit_run(args.run)
    except Exception as error:  # noqa: BLE001 -- negative audit retained
        report = {
            "status": "failed",
            "reason": type(error).__name__ + ": " + str(error),
            "new_physics": 0,
        }
    write_json(args.run / "independent-audit.json", report)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
