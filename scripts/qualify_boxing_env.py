"""Registered bounded D0 shared-world evidence; no competitive learning."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from joxing.bout_recording import BoutRecorder
from joxing.boxing_evidence import BoxingTape, controller_bundles, recording_cases
from joxing.env import BoxingEnv
from joxing.motor_storage import RawCredit, append_json, save_npz, write_json
from joxing.provenance import digest
from joxing.scene import build_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = json.loads((root / "experiments/v0.0.11/recording-protocol.json").read_text())
    if (
        digest(p) != args.protocol_sha256
        or p["cases"] != recording_cases()
        or p["bundles"] != controller_bundles()
    ):
        raise ValueError("Unregistered source/controller/cases")
    for name, sha in {**p["source_files"], **p["upstream_files"]}.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Source drift: " + name)
    frozen = json.loads(
        subprocess.check_output(
            ["git", "show", "HEAD:experiments/v0.0.11/recording-protocol.json"],
            text=True,
        )
    )
    if frozen != p:
        raise ValueError("Protocol not committed at HEAD")
    out = root / "runs/v0.0.11/recording-proof-01"
    out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(out, p["raw_bytes"])
    began = time.monotonic()
    result = {
        "schema": "joxing-shared-recording-proof-v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol_sha256": args.protocol_sha256,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "running",
        "pid": os.getpid(),
        "physics_attempted": 0,
        "optimizer_updates": 0,
        "valid_boxing_games": 0,
        "rows": [
            dict(c, status="not_run", reason="pending", physics_attempted=0)
            for c in p["cases"]
        ],
    }
    write_json(out / "protocol.json", p)
    write_json(out / "result.json", result)
    scene = None
    env = None
    active = None
    tape = None
    case = None

    def guard(count):
        if time.monotonic() - began > p["runner_wall_s"]:
            raise TimeoutError("D0 runner wall cap")
        if result["physics_attempted"] + count > p["physics_cap"]:
            raise TimeoutError("D0 physics cap")
        if shutil.disk_usage(root).free < p["minimum_disk_bytes"]:
            raise TimeoutError("Disk floor")
        if result["physics_attempted"] % 20 == 0:
            available = (
                int(
                    next(
                        x.split()[1]
                        for x in Path("/proc/meminfo").read_text().splitlines()
                        if x.startswith("MemAvailable:")
                    )
                )
                * 1024
            )
            if available < p["minimum_memory_bytes"]:
                raise TimeoutError("Memory floor")
            write_json(
                out / "progress.json",
                {
                    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "wall_s": time.monotonic() - began,
                    "case": active["id"],
                    "tick": env.tick,
                    "physics_attempted": result["physics_attempted"],
                    "hard_crash_physics_lower_bound": result["physics_attempted"],
                    "hard_crash_physics_upper_bound": min(
                        result["physics_attempted"] + 20, p["physics_cap"]
                    ),
                    "available_memory_bytes": available,
                },
            )
        # This is an attempted-step upper bound, advanced before entering the engine.
        result["physics_attempted"] += count
        active["physics_attempted"] += count

    try:
        scene = build_scene()
        write_json(out / "scene.json", scene.report)
        for row in result["rows"]:
            row_began = time.monotonic()
            env = None
            tape = None
            active = row
            row.update(status="running", reason=None)
            write_json(out / "result.json", result)
            append_json(
                out / "events.jsonl",
                {
                    "event": "attempt_started",
                    "id": row["id"],
                    "physics_attempted": result["physics_attempted"],
                    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                },
            )
            case = out / row["id"]
            case.mkdir()
            env = BoxingEnv(
                scene,
                horizon_ticks=row["horizon_ticks"],
                scenario=row["scenario"],
                bundles=p["bundles"],
            )
            if "parent" in row:
                env.restore(
                    json.loads(
                        (
                            out / row["parent"] / (row["checkpoint"] + ".json")
                        ).read_text()
                    )
                )
            else:
                env.reset(seed=row["seed"])
            tape = BoxingTape(env, maximum_ticks=row["physics"])
            write_json(case / "contract.json", env.contract)
            write_json(case / "initial.json", env.snapshot())
            recorder = (
                BoutRecorder(scene, p["bundles"], max_frames=26, max_bytes=8 * 1024**2)
                if row["id"] == "source-90101"
                else None
            )
            if recorder:
                recorder.capture(env.data, physics_steps=0)
            while env.tick < row["end_tick"]:
                if row["id"] == "source-90101" and env.tick == 200:
                    write_json(case / "boundary.json", env.collector_cut())
                if not env.remaining_control:
                    env.begin_control(tape.choose(env), delays=p["delays_ticks"])
                env.advance_substep(guard)
                tape.capture(env)
                if row["id"] == "source-90101" and env.tick == 203:
                    write_json(case / "mid.json", env.snapshot())
                if row["id"] == "contact-source-90104" and env.tick == 1:
                    if not any(
                        t["normal_force_N"] > 0
                        for g in env.evidence[-1]["gloves"].values()
                        for t in g["touches"]
                    ):
                        raise ValueError(
                            "Registered contact reset produced no glove force"
                        )
                    write_json(case / "contact-mid.json", env.snapshot())
                if recorder and env.tick % 40 == 0:
                    peak = max(
                        (
                            c["normal_force_N"]
                            for r in env.evidence[-40:]
                            for c in r["contacts"]
                        ),
                        default=0.0,
                    )
                    recorder.capture(
                        env.data, physics_steps=env.tick, contact_peak_N=peak
                    )
            if row["id"] == "truncation-90103":
                env.truncate("registered D0 external resource cut")
            row.update(tape.save(env, case))
            if (
                row["physics_attempted"] != row["physics"]
                or env.physics_total != row["physics"]
            ):
                raise ValueError(
                    "Attempted physics differs from planned successful row"
                )
            if recorder:
                credit.check(case / "recording/states.npz", 8 * 1024**2)
                recorder.save(case / "recording")
            row.update(status="completed", reason=None)
            row["wall_s"] = time.monotonic() - row_began
            row["diagnostic_physics_per_wall_s"] = row["physics"] / row["wall_s"]
            write_json(out / "result.json", result)
            append_json(
                out / "events.jsonl",
                {
                    "event": "attempt_completed",
                    "id": row["id"],
                    "physics_attempted": result["physics_attempted"],
                    "wall_s": row["wall_s"],
                },
            )
        if result["physics_attempted"] != p["planned_physics"]:
            raise ValueError("Total physics differs")
        result["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- retain every attempted and future row
        result.update(status="halted", reason=type(error).__name__ + ": " + str(error))
        if active:
            active.update(
                status="invalid" if env and env.invalid_reason else "incomplete",
                reason=result["reason"],
            )
        if env is not None:
            active["engine_attempts_confirmed"] = env.physics_total
            active["attempted_lower_bound"] = env.physics_total
            active["attempted_upper_bound"] = active["physics_attempted"]
            if tape is not None:
                try:
                    tape.save(env, case, quarantined=bool(env.invalid_reason))
                except Exception as save_error:  # noqa: BLE001 -- retain loss of partial evidence
                    result["partial_evidence_save_error"] = (
                        type(save_error).__name__ + ": " + str(save_error)
                    )
            with credit.emergency():
                save_npz(out / "fault-engine.npz", state=env.state())
                write_json(
                    out / "fault.json",
                    {
                        "reason": result["reason"],
                        "tick": getattr(env, "tick", None),
                        "physics_total": env.physics_total,
                        "invalid_reason": getattr(env, "invalid_reason", None),
                    },
                )
        for row in result["rows"]:
            if row["status"] == "not_run":
                row["reason"] = "stopped after " + str(
                    active["id"] if active else "initialization"
                )
    finally:
        result["wall_s"] = time.monotonic() - began
        result["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        with credit.emergency():
            append_json(
                out / "events.jsonl",
                {
                    "event": "run_finished",
                    "status": result["status"],
                    "physics_attempted": result["physics_attempted"],
                    "utc": result["finished_utc"],
                },
            )
            files = {
                str(x.relative_to(out)): {
                    "bytes": x.stat().st_size,
                    "sha256": hashlib.sha256(x.read_bytes()).hexdigest(),
                }
                for x in sorted(out.rglob("*"))
                if x.is_file() and x.name not in ["result.json", "raw-manifest.json"]
            }
            write_json(
                out / "raw-manifest.json",
                {"files": files, "bytes": sum(r["bytes"] for r in files.values())},
            )
            write_json(out / "result.json", result)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
