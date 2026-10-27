"""One registered eight-row sensor proof, with no learner or renderer."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from supervise_motor_pilot import identity

from joxing.motor_storage import RawCredit, append_json, save_npz, write_json
from joxing.sensed_env import SensedBoxingEnv
from joxing.sensor_audit import compare_sensor_replay, compare_zero_delay_oracle
from joxing.sensor_evidence import SensorTape
from joxing.sensor_physics_audit import audit_sensor_physics
from joxing.sensor_protocol import load_scene, read_packet


def available_memory():
    return (
        int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = read_packet(root, args.protocol_sha256)
    out = root / "runs/v0.0.11/sensor-proof-01"
    out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(out, p["raw_bytes"])
    began = time.monotonic()
    result = {
        "schema": "joxing-sensor-proof-v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol_sha256": args.protocol_sha256,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "process": identity(os.getpid()),
        "status": "running",
        "physics_attempted": 0,
        "optimizer_updates": 0,
        "valid_boxing_games": 0,
        "rows": [
            dict(
                c,
                status="not_run",
                reason="pending",
                physics_attempted=0,
                engine_attempts_confirmed=0,
            )
            for c in p["cases"]
        ],
    }
    write_json(out / "protocol.json", p)
    write_json(out / "result.json", result)  # Every row durable before MjModel load.
    active = env = tape = case = last_valid = None

    def guard(count):
        if (
            time.monotonic() - began > p["runner_wall_s"]
            or result["physics_attempted"] + count > p["physics_cap"]
        ):
            raise TimeoutError("Sensor wall/physics cap")
        if shutil.disk_usage(root).free < p["minimum_disk_bytes"]:
            raise TimeoutError("Sensor disk floor")
        if result["physics_attempted"] % 20 == 0:
            available = available_memory()
            if available < p["minimum_memory_bytes"]:
                raise TimeoutError("Sensor memory floor")
            write_json(
                out / "progress.json",
                {
                    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "process": result["process"],
                    "case": active["id"],
                    "tick": env.tick,
                    "physics_attempted": result["physics_attempted"],
                    "hard_crash_physics_lower_bound": result["physics_attempted"],
                    "hard_crash_physics_upper_bound": min(
                        result["physics_attempted"] + 20, p["physics_cap"]
                    ),
                    "wall_s": time.monotonic() - began,
                    "available_memory_bytes": available,
                },
            )
        result["physics_attempted"] += (
            count  # Conservative attempt BEFORE engine entry.
        )
        active["physics_attempted"] += count

    try:
        scene = load_scene(p, root)
        write_json(out / "scene.json", scene.report)
        for active in result["rows"]:
            row_began = time.monotonic()
            env = tape = last_valid = None
            active.update(status="running", reason=None)
            write_json(out / "result.json", result)
            append_json(
                out / "events.jsonl",
                {
                    "event": "attempt_started",
                    "id": active["id"],
                    "physics_attempted": result["physics_attempted"],
                },
            )
            case = out / active["id"]
            case.mkdir()
            # Reserve bounded staging headroom before retaining dense arrays.
            credit.check(
                case / "staging-reserve",
                min(32 * 1024**2, 4 * 1024**2 + active["physics"] * 32768),
            )
            env = SensedBoxingEnv(
                scene,
                horizon_ticks=active["horizon_ticks"],
                scenario=active["scenario"],
                bundles=p["bundles"],
                sensor_schedule=active["sensor_schedule"],
            )
            if "parent" in active:
                env.restore(
                    json.loads(
                        (
                            out / active["parent"] / (active["checkpoint"] + ".json")
                        ).read_text()
                    )
                )
            else:
                env.reset(seed=active["seed"])
            if env.tick != active["start_tick"]:
                raise ValueError("Registered initial tick differs")
            tape = SensorTape(env, maximum_ticks=active["physics"])
            write_json(case / "contract.json", env.contract)
            write_json(case / "sensor-initial.json", env.snapshot())
            while env.tick < active["end_tick"]:
                if not env.remaining_control:
                    env.begin_control(tape.choose(env), delays=p["delays_ticks"])
                last_valid = env.state()
                env.advance_substep(guard)
                active["engine_attempts_confirmed"] = env.physics_total
                tape.capture(env)
                if env.tick == active.get("save_mid_tick"):
                    write_json(case / "mid.json", env.snapshot())
            active.update(tape.save(env, case))
            if (
                active["physics_attempted"] != active["physics"]
                or env.physics_total != active["physics"]
            ):
                raise ValueError("Row physics differs")
            # Stop before another row on the first known invariant failure.
            # A separate process repeats all saved-evidence checks after the run.
            active["runtime_physical_audit"] = audit_sensor_physics(
                case,
                scene,
                expected_physics=active["physics"],
                delays=p["delays_ticks"],
            )
            if "parent" in active:
                active["runtime_replay"] = compare_sensor_replay(
                    out / active["parent"],
                    case,
                    source_checkpoint=out
                    / active["parent"]
                    / (active["checkpoint"] + ".json"),
                )
            if active["id"] == "zero-90101":
                active["runtime_oracle"] = compare_zero_delay_oracle(
                    Path(p["oracle_root"]) / "source-90101", case
                )
            guard(0)
            active.update(
                status="completed", reason=None, wall_s=time.monotonic() - row_began
            )
            write_json(out / "result.json", result)
            append_json(
                out / "events.jsonl",
                {
                    "event": "attempt_completed",
                    "id": active["id"],
                    "physics_attempted": result["physics_attempted"],
                },
            )
        if result["physics_attempted"] != p["planned_physics"]:
            raise ValueError("Global physics differs")
        result["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- retain all attempt rows and raw faults
        result.update(status="halted", reason=type(error).__name__ + ": " + str(error))
        if active is not None:
            active.update(
                status="invalid"
                if env is not None and env.invalid_reason
                else "incomplete",
                reason=result["reason"],
            )
        if env is not None:
            active["engine_attempts_confirmed"] = env.physics_total
            if tape is not None:
                try:
                    tape.save(env, case, quarantined=bool(env.invalid_reason))
                except Exception as save_error:  # noqa: BLE001 -- retain partial-save failure
                    result["partial_save_error"] = repr(save_error)
            with credit.emergency():
                save_npz(out / "fault-engine.npz", state=env.state())
                if last_valid is not None:
                    save_npz(out / "last-valid-engine.npz", state=last_valid)
                write_json(
                    out / "fault.json",
                    {
                        "reason": result["reason"],
                        "tick": env.tick,
                        "engine_attempts_confirmed": env.physics_total,
                        "attempted_upper_bound": active["physics_attempted"],
                        "invalid_reason": env.invalid_reason,
                    },
                )
        for row in result["rows"]:
            if row["status"] == "not_run":
                row["reason"] = "stopped after " + str(
                    active["id"] if active else "initialization"
                )
    finally:
        result.update(
            wall_s=time.monotonic() - began,
            finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        with credit.emergency():
            append_json(
                out / "events.jsonl",
                {
                    "event": "run_finished",
                    "status": result["status"],
                    "physics_attempted": result["physics_attempted"],
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
