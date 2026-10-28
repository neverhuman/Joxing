"""Bounded frozen-source retrospective scoring audit and rigid adapter controls."""

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from .evidence_storage import read_json, sha_file
from .evidence_storage import write_json as packed_json
from .motor_storage import RawCredit, append_json, write_json
from .provenance import digest
from .score_adapter_fixture import run_fixture
from .score_diagnostic import audit_saved_trace

PACKET = "experiments/v0.0.11/scored-contact-legacy-repair-protocol.json"
RUN = "runs/v0.0.11/scored-contact-proof-01"
ROW_IDS = [
    "saved-native-source",
    "saved-native-guard-control",
    "published-v21-data-gap",
] + [
    f"rigid-{z}-{order}" for z in ("head", "torso") for order in ("forward", "reverse")
]


def verify_files(p, root):
    for group in ("source_files", "upstream_files", "reference_files"):
        for name, expected in p[group].items():
            if sha_file(root / name) != expected:
                raise ValueError("Scoring source/input drift: " + name)
    for name, expected in p["input_files"].items():
        f = root / name
        if f.stat().st_size != expected["bytes"] or sha_file(f) != expected["sha256"]:
            raise ValueError("Saved scoring input drift: " + name)
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD"], cwd=root, check=False
    ).returncode:
        raise ValueError("Scoring tracked source changed")


def packet(root, protocol_sha, source_sha):
    p = json.loads((root / PACKET).read_text())
    if (
        digest(p) != protocol_sha
        or p["rows"] != ROW_IDS
        or p["schema"] != "joxing-score-qualification-v1"
    ):
        raise ValueError("Foreign scoring packet")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != source_sha
    ):
        raise ValueError("Scoring source HEAD")
    if (
        json.loads(subprocess.check_output(["git", "show", "HEAD:" + PACKET], cwd=root))
        != p
    ):
        raise ValueError("Scoring protocol not committed")
    if (
        p["new_native_physics"] != 0
        or p["fixture_physics_limit"] != 1000
        or p["threads_per_process"] != 1
    ):
        raise ValueError("Foreign scoring physical scope")
    if (
        p["prior_postgate_upper_s"] + p["post_gate_wall_s"] > 900
        or p["prior_raw_upper_bytes"] + p["raw_bytes"] > 256 * 1024**2
    ):
        raise ValueError("Scoring original ledger exhausted")
    verify_files(p, root)
    return p


def saved(root, case):
    folder = root / "score-inputs" / case
    initial, final, trace, contract = [
        read_json(folder / (n + ".json"))
        for n in ("initial", "final", "trace", "contract")
    ]
    for snapshot in (initial, final):
        unsigned = {k: v for k, v in snapshot.items() if k != "sha256"}
        if digest(unsigned) != snapshot["sha256"] or snapshot[
            "contract_hash"
        ] != digest(contract):
            raise ValueError("Saved snapshot/contract hash")
        if (
            digest(snapshot["application"]["referee"]["rules"])
            != contract["rules_hash"]
        ):
            raise ValueError("Saved sporting rules hash")
    return audit_saved_trace(
        initial["application"]["referee"],
        final["application"]["referee"],
        trace["intervals"],
        contract["horizon_ticks"],
    )


def peak_rss():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def published_summary(path):
    """Read this legacy report without admitting nonfinite physical evidence.

    Earlier reporting used -Infinity for unmeasurable Elo. Preserve that token as
    a tagged value, never convert it to a finite statistic. The original bytes
    remain pinned. The selected game/points fields must still be finite JSON.
    """
    constants = []

    def legacy(token):
        if token != "-Infinity":
            raise ValueError("Unknown legacy report constant: " + token)
        constants.append(token)
        return {"unmeasurable_legacy_constant": token}

    raw = path.read_bytes()
    if len(raw) > 2 * 1024**2:
        raise ValueError("Legacy summary input size bound")
    document = json.loads(raw, parse_constant=legacy)
    selected = {
        "games": document["primary_vs_v0019"]["games_detail"],
        "points_replay": document["referee_points_check"],
    }

    # Tagged unavailable statistics may exist elsewhere; selected measurements
    # and categorical fields must not contain them or other nonfinite values.
    def finite_selected(value):
        if isinstance(value, dict):
            if "unmeasurable_legacy_constant" in value:
                raise ValueError("Unmeasurable value in selected game evidence")
            for item in value.values():
                finite_selected(item)
        elif isinstance(value, list):
            for item in value:
                finite_selected(item)
        elif isinstance(value, float):
            import math

            if not math.isfinite(value):
                raise ValueError("Nonfinite selected game evidence")

    finite_selected(selected)
    return selected, constants


def qualify(p, root, out, source_sha, protocol_sha):
    out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(out, p["raw_bytes"] - 256 * 1024)
    began = time.monotonic()
    result = {
        "schema": "joxing-score-qualification-result-v1",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "status": "running",
        "rows": [{"id": n, "status": "not_run", "reason": "pending"} for n in ROW_IDS],
        "new_native_physics": 0,
        "fixture_entered": 0,
        "fixture_completed": 0,
        "optimizer_updates": 0,
        "whole_boxing_proof_passed": False,
        "process": {
            "pid": os.getpid(),
            "start_ticks": int(
                Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
            if Path("/proc/self/stat").exists()
            else None,
        },
    }
    write_json(out / "result.json", result)
    write_json(out / "protocol.json", p)
    current = None

    def budget():
        if (
            time.monotonic() - began > p["stage_wall_limits_s"]["proof"]
            or peak_rss() > p["maximum_rss_bytes"]
        ):
            raise TimeoutError("Scoring proof time/RSS cap")

    try:
        for index, current in enumerate(result["rows"]):
            current.update(status="running", reason=None)
            write_json(out / "result.json", result)
            budget()
            if index < 2:
                case = "source-90101" if index == 0 else "contact-source-90104"
                audit = saved(root, case)
                packed_json(out / (current["id"] + ".json"), audit)
                current.update(
                    intervals=audit["intervals"],
                    final_sha256=audit["final_sha256"],
                    rejection_counts=audit["rejections"],
                )
            elif index == 2:
                summary, legacy_constants = published_summary(
                    root
                    / "grokbot/redteam/data/cycle-0029-mirror-opponent-gate-summary.json"
                )
                rows = summary["games"]
                if len(rows) != 48 or any(
                    "touches" in r or "gloves" in r for r in rows
                ):
                    raise ValueError("Published fixed summary extent/schema changed")
                current.update(
                    classification="NOT_EXPLAINABLE_FROM_THIS_SAVED_SUMMARY",
                    primary_rows=len(rows),
                    reported_points_replay=summary["points_replay"],
                    legacy_unmeasurable_constants=legacy_constants,
                    cause="UNDETERMINED",
                    scope="Input completeness finding, not a failed athlete or a scoring success",
                )
            else:
                zone = "head" if index < 5 else "torso"
                reverse = index % 2 == 0

                def receipt(phase, tick, row_id=current["id"]):
                    budget()
                    key = (
                        "fixture_entered" if phase == "entered" else "fixture_completed"
                    )
                    if (
                        result["fixture_entered"] >= p["fixture_physics_limit"]
                        and phase == "entered"
                    ):
                        raise TimeoutError("Scoring fixture physics cap")
                    append_json(
                        out / "physics.jsonl",
                        {"row": row_id, "phase": phase, "tick": tick},
                    )
                    result[key] += 1

                fixture = run_fixture(zone, reverse, step_receipt=receipt)
                packed_json(out / (current["id"] + ".json"), fixture)
                current.update(
                    points=fixture["points"],
                    positive_force_touches=fixture["positive_force_touches"],
                    final_sha256=fixture["independent_audit"]["final_sha256"],
                    classification=fixture["classification"],
                    physics_steps=250,
                )
            current.update(status="passed", reason=None)
            write_json(out / "result.json", result)
            budget()
        verify_files(p, root)
        result["status"] = "passed"
    except Exception as error:  # noqa: BLE001 -- preserve every failed/not_run row
        import traceback

        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        if current is not None and current["status"] == "running":
            current.update(status="failed", reason=result["error"])
        with credit.emergency():
            write_json(
                out / "fault.json",
                {"error": result["error"], "traceback": traceback.format_exc()},
            )
    finally:
        for row in result["rows"]:
            if row["status"] == "not_run":
                row["reason"] = "stopped after earlier fault"
        result.update(wall_s=time.monotonic() - began, max_rss_bytes=peak_rss())
        with credit.emergency():
            write_json(out / "result.json", result)
    return result


def audit(p, root, out, source_sha, protocol_sha):
    result = read_json(out / "result.json")
    if (
        result["status"] != "passed"
        or result["source_commit"] != source_sha
        or result["protocol_sha256"] != protocol_sha
        or [r["id"] for r in result["rows"]] != ROW_IDS
        or any(r["status"] != "passed" for r in result["rows"])
    ):
        raise ValueError("Scoring attempt denominator")
    rows = []
    for i, row in enumerate(result["rows"]):
        if i < 2:
            rebuilt = saved(root, "source-90101" if i == 0 else "contact-source-90104")
            if rebuilt != read_json(out / (row["id"] + ".json")):
                raise ValueError("Saved explanation differs on independent read")
        elif i > 2:
            f = read_json(out / (row["id"] + ".json"))
            rebuilt = audit_saved_trace(
                f["initial_referee"], f["final_referee"], f["intervals"], 250
            )
            if (
                rebuilt != f["independent_audit"]
                or rebuilt["final_sha256"] != row["final_sha256"]
            ):
                raise ValueError("Rigid adapter saved audit mismatch")
        if i == 2:
            selected, constants = published_summary(
                root
                / "grokbot/redteam/data/cycle-0029-mirror-opponent-gate-summary.json"
            )
            if (
                row["classification"] != "NOT_EXPLAINABLE_FROM_THIS_SAVED_SUMMARY"
                or row["primary_rows"] != len(selected["games"])
                or row["reported_points_replay"] != selected["points_replay"]
                or row["legacy_unmeasurable_constants"] != constants
            ):
                raise ValueError("Published summary completeness finding differs")
        rows.append({"id": row["id"], "status": "passed"})
    entries = [json.loads(l) for l in (out / "physics.jsonl").read_text().splitlines()]
    expected = [
        {"row": n, "phase": phase, "tick": tick}
        for n in ROW_IDS[3:]
        for tick in range(250)
        for phase in ("entered", "completed")
    ]
    if (
        entries != expected
        or result["fixture_entered"] != 1000
        or result["fixture_completed"] != 1000
    ):
        raise ValueError("Fixture physical attempt accounting")
    verify_files(p, root)
    if peak_rss() > p["maximum_rss_bytes"]:
        raise ValueError("Scoring read-audit RSS")
    return {
        "schema": "joxing-score-independent-read-audit-v1",
        "status": "passed",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "rows": rows,
        "new_physics": 0,
        "max_rss_bytes": peak_rss(),
        "whole_boxing_proof_passed": False,
    }
