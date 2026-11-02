"""Owned Linux supervisor for the H15 constructor-only allocation profile."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from joxing_process_ownership import unknown_joxing_processes
from source_shard_support import check_junit
from supervise_motor_pilot import descendants, identity, live
from supervise_upper_task_tests import canonical, save, stop
from upper_task_gate_support import partition_usage

# The supervisor uses only stdlib; the locked runtime is required only by children.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from joxing.trial_memory import (
    profile_denominator,
    read_events,
    sample_owned,
    validate_events,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    p = json.loads((ROOT / args.packet).read_text())
    if (
        p["schema"] != "joxing-trial-memory-profile-v1"
        or canonical(p) != args.protocol_sha256
    ):
        raise ValueError("Profile schema/protocol drift")
    if (
        json.loads(
            subprocess.check_output(["git", "show", "HEAD:" + args.packet], cwd=ROOT)
        )
        != p
    ):
        raise ValueError("Protocol not frozen")

    def verify():
        if (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            != args.source_sha
        ):
            raise ValueError("Source commit drift")
        if (
            subprocess.check_output(
                ["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, text=True
            ).strip()
            != "false"
        ):
            raise ValueError("Complete history required")
        for name, record in p["source_files"].items():
            f = ROOT / name
            if (
                f.stat().st_size != record["bytes"]
                or hashlib.sha256(f.read_bytes()).hexdigest() != record["sha256"]
            ):
                raise ValueError("Source file drift: " + name)
        for name, commit in p["upstream_commits"].items():
            if (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT / "third_party" / name,
                    text=True,
                ).strip()
                != commit
            ):
                raise ValueError("Upstream pin drift")
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=False
        ).returncode:
            raise ValueError("Dirty checkout")

    verify()
    out = ROOT / p["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".trial-memory.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    fixtures = out / "fixtures"
    fixtures.mkdir()
    supervisor = identity(os.getpid())
    ancestors, pid = set(), os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = identity(pid)["parent_pid"]

    def capacity():
        foreign = unknown_joxing_processes(supervisor, ignored_pids=ancestors)
        if foreign:
            raise RuntimeError("Other owned Joxing process: " + str(foreign))
        available = (
            int(
                next(
                    s.split()[1]
                    for s in Path("/proc/meminfo").read_text().splitlines()
                    if s.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        disk = shutil.disk_usage(ROOT).free
        scratch, receipts = partition_usage(out, fixtures)
        if available < p["minimum_memory_bytes"] or disk < p["minimum_disk_bytes"]:
            raise RuntimeError("Shared-host capacity floor")
        if scratch > p["fixture_bytes"] or receipts > p["receipt_bytes"]:
            raise RuntimeError("Profile storage credit")
        return {
            "available_memory_bytes": available,
            "free_disk_bytes": disk,
            "fixture_bytes": scratch,
            "receipt_bytes": receipts,
            "load": os.getloadavg(),
        }

    report = {
        "schema": p["schema"],
        "source_commit": args.source_sha,
        "protocol_sha256": args.protocol_sha256,
        "supervisor": supervisor,
        "status": "running",
        "stages": [],
        "trials": [{"arm": a, "status": "not_run"} for a in ("A", "B")],
        "formal_native_evaluations": 0,
        "formal_training_runs": 0,
        "fixture_physics": "unknown until completed constructor events; partial on forced stop",
        "initial_capacity": capacity(),
    }
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    env.update(
        PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl"
    )
    python = str(ROOT / "experiments/v0.0.11/runtime/.venv/bin/python")
    stages = [
        (
            "sync",
            [
                shutil.which("uv") or "/home/ubuntu/.local/bin/uv",
                "sync",
                "--project",
                "experiments/v0.0.11/runtime",
                "--locked",
            ],
            p["sync_wall_s"],
        ),
        (
            "runtime",
            [
                python,
                "-c",
                "import sys,numpy,mujoco,torch,stable_baselines3; v={'python':sys.version.split()[0],'numpy':numpy.__version__,'mujoco':mujoco.__version__,'torch':torch.__version__,'sb3':stable_baselines3.__version__}; print(v); assert v == "
                + repr(p["runtime_versions"]),
            ],
            p["runtime_wall_s"],
        ),
        (
            "tests",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--junitxml",
                str(out / "tests.xml"),
                "--basetemp",
                str(fixtures / "test-controls"),
                "tests/test_trial_memory.py",
                "tests/test_process_ownership.py",
            ],
            p["tests_wall_s"],
        ),
        (
            "profile",
            [
                python,
                "-m",
                "joxing.trial_memory",
                "--folder",
                str(out),
                "--packet",
                str(ROOT / args.packet),
            ],
            p["profile_wall_s"],
        ),
    ]
    began, seen, active, row = time.monotonic(), {}, None, None
    post_setup_started = None
    planned_stop = False
    try:
        for name, command, limit in stages:
            if name == "tests":
                post_setup_started = time.monotonic()
            row = {
                "stage": name,
                "command": command,
                "status": "running",
                "owned_identities": [],
            }
            report["stages"].append(row)
            started = time.monotonic()
            with (out / (name + ".log")).open("w") as log:
                active = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while True:
                    for item in descendants(active.pid):
                        seen[(item["pid"], item["start_ticks"])] = item
                    living = [i for i in seen.values() if live(i)]
                    samples = sample_owned([supervisor, *living], is_live=live)
                    aggregate = sum(s["rss_bytes"] for s in samples)
                    elapsed = time.monotonic() - started
                    row.update(
                        elapsed_s=elapsed,
                        peak_sampled_rss_bytes=max(
                            row.get("peak_sampled_rss_bytes", 0), aggregate
                        ),
                        capacity=capacity(),
                        owned_identities=list(seen.values()),
                    )
                    if name == "profile":
                        events = read_events(out / "events")
                        roles = validate_events(events, list(seen.values()))
                        roles[f"{supervisor['pid']}:{supervisor['start_ticks']}"] = {
                            "role": "supervisor"
                        }
                        controls_path = out / "control-roles.json"
                        if controls_path.exists():
                            for control in json.loads(controls_path.read_text()):
                                i = control["identity"]
                                if (i["pid"], i["start_ticks"]) not in seen:
                                    # Never read memory for an event-supplied foreign PID.
                                    raise ValueError("Unbound control process role")
                                roles.setdefault(
                                    f"{i['pid']}:{i['start_ticks']}",
                                    {
                                        k: v
                                        for k, v in control.items()
                                        if k != "identity"
                                    },
                                )
                        sample = {
                            "monotonic_s": time.monotonic(),
                            "aggregate_rss_bytes": aggregate,
                            "aggregate_pss_bytes": sum(
                                s["pss_bytes"] or 0 for s in samples
                            ),
                            "processes": samples,
                            "roles": roles,
                            "event_count": len(events),
                        }
                        with (out / "memory.jsonl").open("a") as stream:
                            stream.write(json.dumps(sample) + "\n")
                        ready = {
                            e["trial"] for e in events if e["phase"] == "learn_ready"
                        }
                        if ready == {"A", "B"}:
                            (out / "release").touch()
                        if aggregate >= p["planned_stop_rss_bytes"]:
                            planned_stop = True
                    save(out / "result.json", report)
                    if (
                        aggregate > p["owned_rss_bytes"]
                        or 1 + len(living) > p["process_ceiling"]
                    ):
                        raise RuntimeError(
                            f"Hard profile resource cap: rss={aggregate}, processes={1 + len(living)}"
                        )
                    if (
                        elapsed > limit - 2
                        or time.monotonic() - began > p["whole_wall_s"] - 2
                        or (
                            post_setup_started is not None
                            and time.monotonic() - post_setup_started
                            > p["post_setup_wall_s"] - 2
                        )
                    ):
                        raise TimeoutError("Profile first wall limit")
                    if planned_stop:
                        stop(active, seen)
                        row["status"] = "capacity_limited"
                        break
                    if active.poll() is not None:
                        break
                    time.sleep(p["sample_interval_s"])
                row.update(returncode=active.wait(), wall_s=time.monotonic() - started)
                # Reap asynchronous manager/tracker shutdown without starting another stage.
                until = time.monotonic() + 0.5
                while any(live(i) for i in seen.values()) and time.monotonic() < until:
                    time.sleep(0.05)
                if any(live(i) for i in seen.values()):
                    stop(active, seen)
                if any(live(i) for i in seen.values()):
                    raise RuntimeError("Profile cleanup still live")
                active = None
            if planned_stop:
                break
            if row["returncode"]:
                raise RuntimeError("Profile stage nonzero exit: " + name)
            if name == "tests":
                checked = check_junit(
                    (out / "tests.xml").read_text(), p["test_node_ids"], []
                )
                row["junit_counts"] = checked["counts"]
                if not checked["qualified"]:
                    raise ValueError("Exact profile control denominator")
            row["status"] = "completed"
        verify()
        report["status"] = "capacity_limited" if planned_stop else "completed"
    except BaseException as error:  # noqa: BLE001 -- always stop exclusively owned children
        if active is not None:
            stop(active, seen)
        report.update(status="halted", error=type(error).__name__ + ": " + str(error))
        if row is not None and row["status"] == "running":
            row["status"] = "failed"
    finally:
        report["events"] = read_events(out / "events")
        report["constructor_denominator"] = profile_denominator(report["events"])
        for trial in report["trials"]:
            events = [e for e in report["events"] if e["trial"] == trial["arm"]]
            trial["status"] = (
                "stopped_before_training"
                if any(e["phase"] == "profile_exit" for e in events)
                else "unconfirmed"
                if events
                else "not_run"
            )
            trial["ready"] = any(e["phase"] == "learn_ready" for e in events)
        report.update(
            wall_s=time.monotonic() - began,
            owned_identities=list(seen.values()),
            final_live_recorded=[i for i in seen.values() if live(i)],
            guards_violated=any(
                e["phase"] == "guard_violation" for e in report["events"]
            ),
        )
        if report["status"] == "completed" and any(
            t["status"] != "stopped_before_training" or not t["ready"]
            for t in report["trials"]
        ):
            report.update(
                status="halted",
                error="Missing completed constructor/readiness denominator",
            )
        if (
            report["status"] == "completed"
            and not report["constructor_denominator"]["all_worlds_constructed"]
        ):
            report.update(
                status="halted", error="Missing ten-world constructor denominator"
            )
        save(out / "result.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "wall_s",
                    "trials",
                    "guards_violated",
                    "final_live_recorded",
                )
            }
        )
    )
    if report["status"] == "halted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
