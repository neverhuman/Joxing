"""Fixed native interaction proof, activated only after same-source phase A passes."""

import json
import resource
import subprocess
import time

from .evidence_storage import read_json
from .evidence_storage import write_json as packed_json
from .frozen_group_controller import FrozenGroupController
from .motor_storage import RawCredit, append_json, write_json
from .native_strike_audit import audit_source
from .native_strike_probe import NativeStrikeTape, continue_mid
from .provenance import digest
from .score_qualification import verify_files

PACKET = "experiments/v0.0.11/native-strike-protocol.json"
RUN = "runs/v0.0.11/native-strike-proof-01"


def packet(root, protocol_sha, source_sha):
    from .shared_motor_proof import validate_test_summary

    p = read_json(root / PACKET)
    if digest(p) != protocol_sha or p["schema"] != "joxing-native-strike-protocol-v1":
        raise ValueError("Native strike packet identity")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != source_sha
    ):
        raise ValueError("Native strike source HEAD")
    if (
        json.loads(subprocess.check_output(["git", "show", "HEAD:" + PACKET], cwd=root))
        != p
    ):
        raise ValueError("Native strike packet not committed")
    verify_files(p, root)
    a = read_json(root / "runs/v0.0.11/scored-contact-proof-01/result.json")
    s = read_json(root / "runs/v0.0.11/scored-contact-supervision-01/result.json")
    audited = read_json(
        root / "runs/v0.0.11/scored-contact-proof-01/independent-audit.json"
    )
    if (
        any(
            r["source_commit"] != source_sha
            or r["protocol_sha256"] != p["required_a_protocol_sha256"]
            for r in (a, s, audited)
        )
        or a["status"] != "passed"
        or s["status"] != "completed"
        or audited["status"] != "passed"
    ):
        raise ValueError("Same-source scoring phase A prerequisite missing")
    validate_test_summary(
        p, (root / "runs/v0.0.11/scored-contact-supervision-01/tests.log").read_text()
    )
    if (
        p["prior_postgate_upper_s"] + s["post_gate_wall_s"] + p["post_gate_wall_s"]
        > 900
    ):
        raise ValueError(
            "Original native ledger cannot fund the registered strike probe"
        )
    a_raw = sum(
        f.stat().st_size
        for name in ("scored-contact-proof-01", "scored-contact-supervision-01")
        for f in (root / "runs/v0.0.11" / name).rglob("*")
        if f.is_file()
    )
    if p["prior_raw_upper_bytes"] + a_raw + p["raw_bytes"] > 256 * 1024**2:
        raise ValueError("Original native raw ledger cannot fund probe")
    return p


def controllers(root, p):
    return {
        a: FrozenGroupController.from_summary(
            root / p["policies"][a]["path"],
            expected_sha256=p["policies"][a]["sha256"],
            noise=0.05,
        )
        for a in "AB"
    }


def environment(actors):
    from grokbot.scenarios import RangeBoxingEnv

    from .scene import build_scene

    return RangeBoxingEnv(
        build_scene(),
        separation_m=0.5,
        horizon_ticks=3000,
        bundles={a: c.bundle_sha256 for a, c in actors.items()},
    )


def prove(root, p, source_sha, protocol_sha):
    out = root / RUN
    out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(out, p["raw_bytes"] - 256 * 1024)
    result = {
        "schema": "joxing-native-strike-result-v1",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "status": "running",
        "rows": [
            {"id": "fixed-source-10001-A", "status": "not_run"},
            {"id": "actual-mid-203-continuation", "status": "not_run"},
        ],
        "physics_entered": 0,
        "physics_completed": 0,
        "optimizer_updates": 0,
        "qualified_boxing_games": 0,
    }
    began = time.monotonic()
    write_json(out / "result.json", result)
    write_json(out / "protocol.json", p)
    tape, env, current = None, None, None

    def receipt(phase, tick):
        if (
            time.monotonic() - began >= p["stage_wall_limits_s"]["proof"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            > p["maximum_rss_bytes"]
        ):
            raise TimeoutError("Native source proof wall/RSS cap")
        if phase == "entered" and result["physics_entered"] >= 5797:
            raise TimeoutError("Native source physics cap")
        append_json(
            out / "physics.jsonl", {"row": current["id"], "phase": phase, "tick": tick}
        )
        result["physics_entered" if phase == "entered" else "physics_completed"] += 1

    try:
        current = result["rows"][0]
        current.update(status="running")
        write_json(out / "result.json", result)
        actors = controllers(root, p)
        env = environment(actors)
        write_json(out / "scene.json", env.scene.report)
        write_json(out / "policies.json", {a: c.bundle for a, c in actors.items()})
        tape = NativeStrikeTape()
        source = tape.run(env, actors, seed=10001, mid_tick=203, step_receipt=receipt)
        tape.save(env, out / "source")
        current.update(status="passed", **source)
        write_json(out / "result.json", result)
        current = result["rows"][1]
        current.update(status="running")
        write_json(out / "result.json", result)
        replay = continue_mid(env, controllers(root, p), tape, step_receipt=receipt)
        if replay["final_sha256"] != source["source_final_sha256"]:
            raise ValueError("Native continued final snapshot differs")
        packed_json(out / "continuation.json", replay)
        current.update(
            status="passed",
            new_physics=replay["new_physics"],
            final_sha256=replay["final_sha256"],
            held_ticks_without_actor=replay["held_ticks_without_actor"],
        )
        result.update(
            status="passed", native_legal_points_observed=any(source["points"])
        )
        verify_files(p, root)
    except Exception as error:  # noqa: BLE001 -- retain all attempts/partial evidence
        import traceback

        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        if current is not None:
            current.update(status="failed", reason=result["error"])
        with credit.emergency():
            write_json(
                out / "fault.json",
                {
                    "error": result["error"],
                    "traceback": traceback.format_exc(),
                    "physics_entered": result["physics_entered"],
                    "physics_completed": result["physics_completed"],
                },
            )
            if env is not None and env.initialized:
                write_json(
                    out / "fault-engine.json",
                    {
                        "engine": env.state().tolist(),
                        "tick": env.tick,
                        "last_interval": None
                        if not env.evidence
                        else {
                            "tick": env.evidence[-1]["tick"],
                            "gloves": env.evidence[-1]["gloves"],
                            "events": env.evidence[-1]["events"],
                        },
                    },
                )
            if (
                tape is not None
                and tape.initial is not None
                and not (out / "source").exists()
            ):
                tape.save(env, out / "source", quarantined=True)
    finally:
        for row in result["rows"]:
            if row["status"] == "not_run":
                row["reason"] = "stopped after earlier fault"
        result["environment_engine_entries"] = 0 if env is None else env.physics_total
        result.update(
            wall_s=time.monotonic() - began,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        with credit.emergency():
            write_json(out / "result.json", result)
    return result


def audit(root, p, source_sha, protocol_sha):
    out = root / RUN
    result = read_json(out / "result.json")
    if (
        result["status"] != "passed"
        or any(r["status"] != "passed" for r in result["rows"])
        or result["source_commit"] != source_sha
        or result["protocol_sha256"] != protocol_sha
        or result["physics_entered"] != 5797
        or result["physics_completed"] != 5797
    ):
        raise ValueError("Native source/replay denominator")
    actors = controllers(root, p)
    env = environment(actors)
    audited = audit_source(out / "source", env, actors)
    replay = read_json(out / "continuation.json")
    trace = read_json(out / "source/trace.json")
    if (
        replay["new_physics"] != 2797
        or replay["held_ticks_without_actor"] != 17
        or replay["final_sha256"] != audited["source_final_sha256"]
    ):
        raise ValueError("Native actual continuation extent/identity")
    for row, expected in zip(
        replay["actual_physical_tail"], trace["intervals"][203:], strict=True
    ):
        if (
            row["tick"] != expected["tick"]
            or row["actual_sha256"] != digest(expected)
            or row["expected_sha256"] != digest(expected)
        ):
            raise ValueError("Native actual continued measurement hash")
    entries = [json.loads(s) for s in (out / "physics.jsonl").read_text().splitlines()]
    expected = [
        {"row": name, "phase": phase, "tick": tick}
        for name, start in [
            ("fixed-source-10001-A", 0),
            ("actual-mid-203-continuation", 203),
        ]
        for tick in range(start, 3000)
        for phase in ("entered", "completed")
    ]
    if entries != expected:
        raise ValueError("Native exact engine-entry/completion ledger")
    verify_files(p, root)
    return {
        "schema": "joxing-native-strike-saved-audit-v1",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "status": "passed",
        "new_physics": 0,
        "source_audit": audited,
        "actual_continuation_steps": 2797,
        "whole_boxing_proof_passed": False,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
