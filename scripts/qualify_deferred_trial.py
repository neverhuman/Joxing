"""Frozen four-slot native parity supervisor; no concurrency or default adoption."""

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
from supervise_upper_task_tests import canonical, rss, save, stop
from upper_task_gate_support import partition_usage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from joxing.physics_credit import PhysicsCredit


def runner_module(schema):
    return {
        "joxing-deferred-native-parity-v1": "joxing.deferred_qualification",
        "joxing-native-teardown-v1": "joxing.native_teardown",
        "joxing-serial-native-teardown-v1": "joxing.serial_native_teardown",
    }[schema]


def main():
    began = time.monotonic()
    parser = argparse.ArgumentParser()
    for key in ("packet", "source-sha", "protocol-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    p = json.loads((ROOT / args.packet).read_text())
    if (
        p["schema"]
        not in (
            "joxing-deferred-native-parity-v1",
            "joxing-native-teardown-v1",
            "joxing-serial-native-teardown-v1",
        )
        or canonical(p) != args.protocol_sha256
    ):
        raise ValueError("Frozen native parity protocol")
    if (
        json.loads(
            subprocess.check_output(["git", "show", "HEAD:" + args.packet], cwd=ROOT)
        )
        != p
    ):
        raise ValueError("Uncommitted native protocol")

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
        for n, d in p["source_files"].items():
            f = ROOT / n
            if (
                f.stat().st_size != d["bytes"]
                or hashlib.sha256(f.read_bytes()).hexdigest() != d["sha256"]
            ):
                raise ValueError("Source drift: " + n)
        for obj, expected in p["required_git_objects"].items():
            if (
                hashlib.sha256(
                    subprocess.check_output(["git", "show", obj], cwd=ROOT)
                ).hexdigest()
                != expected
            ):
                raise ValueError("Historical object drift")
        for n, c in p["upstream_commits"].items():
            if (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT / "third_party" / n,
                    text=True,
                ).strip()
                != c
            ):
                raise ValueError("Upstream drift")
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=False
        ).returncode:
            raise ValueError("Dirty native source")

    def verify_runtime_files():
        for n, d in p.get("runtime_file_hashes", {}).items():
            f = ROOT / n
            if (
                f.stat().st_size != d["bytes"]
                or hashlib.sha256(f.read_bytes()).hexdigest() != d["sha256"]
            ):
                raise ValueError("Installed runtime source drift: " + n)

    verify()
    serial_constructor = p["schema"] == "joxing-serial-native-teardown-v1"
    constructor_only = serial_constructor or p["schema"] == "joxing-native-teardown-v1"
    out = ROOT / p["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".deferred-native.lock").open("a+")
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
            raise RuntimeError("Another owned Joxing job: " + str(foreign))
        memory = (
            int(
                next(
                    x.split()[1]
                    for x in Path("/proc/meminfo").read_text().splitlines()
                    if x.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        disk = shutil.disk_usage(ROOT).free
        raw, receipts = partition_usage(out.parent, fixtures)
        if memory < p["minimum_memory_bytes"] or disk < p["minimum_disk_bytes"]:
            raise RuntimeError("Shared-host capacity floor")
        if raw > p["fixture_bytes"] or receipts > p["receipt_bytes"]:
            raise RuntimeError("Native output first limit")
        return {
            "available_memory_bytes": memory,
            "free_disk_bytes": disk,
            "fixture_bytes": raw,
            "receipt_bytes": receipts,
            "load": os.getloadavg(),
        }

    report = {
        "schema": p["schema"],
        "source_commit": args.source_sha,
        "protocol_sha256": args.protocol_sha256,
        "supervisor": supervisor,
        "status": "running",
        "rows": [dict(s, status="not_run") for s in p["slots"]],
        "stages": [],
        "initial_capacity": capacity(),
        "formal_boxing_games": 0,
        "competitive_training_runs": 0,
    }
    seen = {}
    active = guard = None
    row = None
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
    ledger_path = fixtures / "native-credit.json"
    binding = canonical(
        {
            "source": args.source_sha,
            "protocol": args.protocol_sha256,
            "config": p["trial_config_sha256"],
        }
    )
    ledger = None

    def stage(name, command, limit, slot=None):
        nonlocal active, guard, row
        row = {"stage": name, "command": command, "status": "running"}
        report["stages"].append(row)
        started = time.monotonic()
        log = (out / (name + ".log")).open("w")
        guard_log = None
        try:
            active = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            root = identity(active.pid)
            seen[(root["pid"], root["start_ticks"])] = root
            row["process"] = root
            if slot is not None:
                remaining = (
                    min(limit, p["whole_wall_s"] - (time.monotonic() - began)) - 2
                )
                guard_command = [
                    "python3",
                    "scripts/trial_rss_watchdog.py",
                    "--root-pid",
                    str(root["pid"]),
                    "--root-start",
                    str(root["start_ticks"]),
                    "--supervisor-pid",
                    str(supervisor["pid"]),
                    "--supervisor-start",
                    str(supervisor["start_ticks"]),
                    "--output",
                    str(out / (name + "-watchdog.json")),
                    "--planned-rss",
                    str(p["planned_stop_rss_bytes"]),
                    "--hard-rss",
                    str(p["owned_rss_bytes"]),
                    "--process-cap",
                    str(p["process_ceiling"]),
                    "--wall-s",
                    str(remaining),
                    "--max-gap-s",
                    str(p["max_monitor_gap_s"]),
                    "--rss-reader",
                    p.get("rss_reader", "strict-v2"),
                ]
                guard_log = (out / (name + "-watchdog.log")).open("w")
                guard = subprocess.Popen(
                    guard_command,
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=guard_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                watcher = identity(guard.pid)
                seen[(watcher["pid"], watcher["start_ticks"])] = watcher
                row.update(watchdog_process=watcher, watchdog_command=guard_command)
            while True:
                for i in descendants(active.pid):
                    seen[(i["pid"], i["start_ticks"])] = i
                row.update(elapsed_s=time.monotonic() - started, capacity=capacity())
                if guard is None:
                    living = [i for i in seen.values() if live(i)]
                    total = rss(supervisor) + sum(rss(i) for i in living)
                    row["peak_sampled_rss_bytes"] = max(
                        row.get("peak_sampled_rss_bytes", 0), total
                    )
                    if (
                        total > p["owned_rss_bytes"]
                        or len(living) + 1 > p["process_ceiling"]
                    ):
                        raise RuntimeError("Prerequisite stage resource first limit")
                save(out / "result.json", report)
                if (
                    row["elapsed_s"] > limit - 2
                    or time.monotonic() - began > p["whole_wall_s"] - 2
                ):
                    raise TimeoutError("Native parity first wall limit")
                if (
                    guard is not None
                    and guard.poll() is not None
                    and active.poll() is None
                ):
                    raise RuntimeError("Independent watcher exited before trial")
                if active.poll() is not None:
                    break
                time.sleep(0.2)
            code = active.wait()
            row["returncode"] = code
            if guard is not None:
                guard.wait(timeout=2)
                watch = json.loads((out / (name + "-watchdog.json")).read_text())
                row["watchdog"] = watch
                for i in watch["owned_identities"]:
                    seen[(i["pid"], i["start_ticks"])] = i
                if (
                    guard.returncode
                    or watch["status"] != "exited"
                    or watch["final_live"]
                ):
                    raise RuntimeError(
                        "Native parity watcher stop/fault: " + str(watch.get("reason"))
                    )
                if watch.get(
                    "peak_capacity_charge_bytes", watch["peak_sampled_rss_bytes"]
                ) > p["owned_rss_bytes"] or watch.get("pending_retirements"):
                    raise RuntimeError(
                        "Native watcher capacity/pending retirement fault"
                    )
                guard = None
            if code:
                raise RuntimeError("Native parity stage nonzero: " + name)
            if any(live(i) for i in seen.values()):
                until = time.monotonic() + 0.5
                while any(live(i) for i in seen.values()) and time.monotonic() < until:
                    time.sleep(0.05)
                if any(live(i) for i in seen.values()):
                    raise RuntimeError("Owned stage descendant still live")
            active = None
            row.update(status="completed", wall_s=time.monotonic() - started)
        finally:
            log.close()
            if guard_log is not None:
                guard_log.close()

    try:
        stage(
            "sync",
            [
                shutil.which("uv") or "/home/ubuntu/.local/bin/uv",
                "sync",
                "--project",
                "experiments/v0.0.11/runtime",
                "--locked",
            ],
            180,
        )
        stage(
            "runtime",
            [
                python,
                "-c",
                "import sys,numpy,mujoco,torch,stable_baselines3;v={'python':sys.version.split()[0],'numpy':numpy.__version__,'mujoco':mujoco.__version__,'torch':torch.__version__,'sb3':stable_baselines3.__version__};print(v);assert v=="
                + repr(p["runtime_versions"]),
            ],
            15,
        )
        verify_runtime_files()
        stage(
            "controls",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--junitxml",
                str(out / "controls.xml"),
                "--basetemp",
                str(fixtures / "controls"),
                *p["test_modules"],
            ],
            60,
        )
        checked = check_junit(
            (out / "controls.xml").read_text(), p["test_node_ids"], []
        )
        report["control_counts"] = checked["counts"]
        if not checked["qualified"]:
            raise ValueError("Native prerequisite controls incomplete")
        ledger = PhysicsCredit.create(ledger_path, binding, p["physics_limit"])
        for slot in report["rows"]:
            slot["status"] = "unconfirmed"
            name = slot["id"]
            command = [
                python,
                "-m",
                runner_module(p["schema"]),
                "--folder",
                str(fixtures / name),
                "--slot",
                name,
                "--packet",
                str(ROOT / args.packet),
                "--ledger",
                str(ledger_path),
                "--binding",
                binding,
                "--ready",
                str(out / (name + "-watchdog.ready")),
            ]
            stage(name, command, p["per_slot_wall_s"], slot=slot)
            if constructor_only:
                facts = json.loads(
                    (fixtures / name / "constructor-result.json").read_text()
                )
                workers = facts["workers"] if serial_constructor else [facts["worker"]]
                if len(workers) != (4 if serial_constructor else 1):
                    raise ValueError("Native constructor worker denominator")
                retirements = [
                    row["watchdog"]["retirement_records"].get(
                        f"{worker['pid']}:{worker['start']}"
                    )
                    for worker in workers
                ]
                observed = [r for r in retirements if r is not None]
                if not observed or any(
                    not r["released"]
                    or r["highwater_charge_bytes"] <= 0
                    or r["max_hold_s"] > 0.1
                    for r in observed
                ):
                    raise ValueError("Native worker retirement path not qualified")
                slot.update(status="completed", facts=facts)
                if serial_constructor:
                    slot["retirements"] = observed
                else:
                    slot["retirement"] = observed[0]
                report["physics_credit"] = ledger.snapshot()
                if (
                    report["physics_credit"]["used"]
                    != (800 if serial_constructor else 200)
                    or report["physics_credit"]["pending"]
                ):
                    raise ValueError("Constructor physics denominator")
                continue
            semantic = json.loads(
                (fixtures / name / "semantic-result.json").read_text()
            )
            slot.update(
                status="completed",
                training_transitions=semantic["training_transitions"],
                updates=semantic["updates"],
                initial_actor_hash=semantic["initial_actor_hash"],
                final_actor_hash=semantic["final_actor_hash"],
            )
            report["physics_credit"] = ledger.snapshot()
            if report["physics_credit"]["pending"]:
                raise ValueError("Unknown physics work cannot qualify")
        if constructor_only:
            report["constructor_gate_qualified"] = True
        else:
            from joxing.deferred_qualification import compare_slots

            report["comparison"] = compare_slots(
                {s["id"]: str(fixtures / s["id"]) for s in report["rows"]}
            )
        verify_runtime_files()
        verify()
        report["status"] = "completed"
    except BaseException as error:  # noqa: BLE001 -- retain every attempted/not_run slot
        for trial in report["rows"]:
            if trial["status"] == "unconfirmed":
                trial.update(
                    status="invalid",
                    work_status="partial_or_unconfirmed",
                    reason=type(error).__name__ + ": " + str(error),
                )
        if active is not None:
            stop(
                active,
                {
                    k: v
                    for k, v in seen.items()
                    if guard is None or v["pid"] != guard.pid
                },
            )
        if guard is not None:
            try:
                guard.wait(timeout=2)
            except subprocess.TimeoutExpired:
                stop(guard, {})
        if row is not None and row["status"] == "running":
            row["status"] = "failed"
        report.update(status="halted", error=type(error).__name__ + ": " + str(error))
    finally:
        if ledger is not None:
            report["physics_credit"] = ledger.snapshot()
        report.update(
            wall_s=time.monotonic() - began,
            owned_identities=list(seen.values()),
            final_live_recorded=[i for i in seen.values() if live(i)],
        )
        save(out / "result.json", report)
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "status",
                    "error",
                    "rows",
                    "physics_credit",
                    "wall_s",
                    "final_live_recorded",
                )
            }
        )
    )
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
