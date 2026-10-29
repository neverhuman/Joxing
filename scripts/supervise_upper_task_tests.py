"""Owned full Linux source regression gate; no formal motor pilot activation."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from joxing_process_ownership import unknown_joxing_processes
from supervise_motor_pilot import descendants, identity, live

ROOT = Path(__file__).resolve().parents[1]
PACKET = "experiments/v0.0.11/upper-task-source-gate-protocol.json"
RUN = "runs/v0.0.11/upper-task-source-gate-01"


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def save(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > 512 * 1024:
        raise ValueError("Source gate report credit")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)


def size(root):
    total = 0
    for f in root.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except FileNotFoundError:
            pass
    return total


def rss(row):
    if not live(row):
        return 0
    try:
        return (
            int(
                next(
                    s.split()[1]
                    for s in Path(f"/proc/{row['pid']}/status").read_text().splitlines()
                    if s.startswith("VmRSS:")
                )
            )
            * 1024
        )
    except (OSError, StopIteration):
        return 0


def stop(proc, seen):
    # Signal only the freshly launched group and bound PID/start identities.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for row in seen.values():
        if live(row):
            try:
                os.kill(row["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=1)
    for row in seen.values():
        if live(row):
            try:
                os.kill(row["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    p = json.loads((ROOT / PACKET).read_text())
    if (
        p["schema"] != "joxing-upper-task-source-gate-v1"
        or canonical(p) != args.protocol_sha256
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != args.source_sha
        or json.loads(
            subprocess.check_output(["git", "show", "HEAD:" + PACKET], cwd=ROOT)
        )
        != p
    ):
        raise ValueError("Frozen source-gate source/protocol")

    def verify():
        for n, d in p["source_files"].items():
            f = ROOT / n
            if (
                f.stat().st_size != d["bytes"]
                or hashlib.sha256(f.read_bytes()).hexdigest() != d["sha256"]
            ):
                raise ValueError("Source-gate identity: " + n)
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=False
        ).returncode:
            raise ValueError("Source-gate dirty checkout")

    verify()
    out = ROOT / RUN
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".upper-task-source-gate.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    supervisor = identity(os.getpid())
    ancestors, pid = set(), os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = identity(pid)["parent_pid"]
    fixture = out / "fixtures"

    def capacity():
        other = unknown_joxing_processes(supervisor, ignored_pids=ancestors)
        if other:
            raise RuntimeError("Another owned Joxing process: " + str(other))
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
        if (
            available < p["minimum_available_memory_bytes"]
            or disk < p["minimum_free_disk_bytes"]
        ):
            raise RuntimeError("Source-gate shared-host capacity floor")
        scratch = size(fixture)
        raw = size(out.parent) - scratch
        if scratch > p["fixture_scratch_bytes"] or raw > p["receipt_bytes"]:
            raise RuntimeError("Source-gate fixture/receipt credit")
        return {
            "available_memory_bytes": available,
            "free_disk_bytes": disk,
            "fixture_bytes": scratch,
            "receipt_bytes": raw,
            "load": os.getloadavg(),
        }

    env = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[k] = "1"
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl")
    python = str(ROOT / "experiments/v0.0.11/runtime/.venv/bin/python")
    commands = {
        "sync": [
            shutil.which("uv") or "/home/ubuntu/.local/bin/uv",
            "sync",
            "--project",
            "experiments/v0.0.11/runtime",
            "--locked",
        ],
        "runtime": [
            python,
            "-c",
            "import sys,json,numpy,mujoco; v={'python':sys.version.split()[0],'numpy':numpy.__version__,'mujoco':mujoco.__version__}; print(json.dumps(v)); assert v == "
            + repr(p["runtime_versions"]),
        ],
        "tests": [
            python,
            "-m",
            "pytest",
            "-q",
            "-rx",
            "-p",
            "no:cacheprovider",
            "--durations=15",
            "--basetemp",
            str(fixture),
            "--junitxml",
            str(out / "tests.xml"),
            "tests",
            "grokbot/tests",
        ],
    }
    report = {
        "schema": "joxing-upper-task-source-supervision-v1",
        "source_commit": args.source_sha,
        "protocol_sha256": args.protocol_sha256,
        "process": supervisor,
        "status": "running",
        "formal_native_steps": 0,
        "formal_training_updates": 0,
        "scope": "Full source regression with fixture physics/updates; not a released-checkpoint pilot",
        "initial_capacity": capacity(),
        "stages": [
            {"stage": n, "status": "not_run", "command": cmd}
            for n, cmd in commands.items()
        ],
    }
    active, seen = None, {}
    began = time.monotonic()

    def interrupted(signum, _frame):
        raise TimeoutError("Owned source gate interrupted: " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save(out / "result.json", report)
    try:
        for row in report["stages"]:
            verify()
            row.update(status="running", limit_s=p["stage_wall_limits_s"][row["stage"]])
            started = time.monotonic()
            with (out / (row["stage"] + ".log")).open("w") as log:
                active = subprocess.Popen(
                    row["command"],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                seen = {}
                row["process"] = identity(active.pid)
                row["peak_owned_rss_bytes"] = 0
                save(out / "result.json", report)
                while True:
                    for child in descendants(active.pid):
                        seen[(child["pid"], child["start_ticks"])] = child
                    live_rows = [r for r in seen.values() if live(r)]
                    peak = sum(rss(r) for r in live_rows)
                    row["peak_owned_rss_bytes"] = max(row["peak_owned_rss_bytes"], peak)
                    row["owned_identities"] = list(seen.values())
                    row["latest_capacity"] = capacity()
                    row["elapsed_s"] = time.monotonic() - started
                    if (
                        row["elapsed_s"] >= row["limit_s"] - p["termination_reserve_s"]
                        or time.monotonic() - began
                        >= p["whole_wall_s"] - p["termination_reserve_s"]
                        or peak > p["maximum_owned_rss_bytes"]
                        or len(live_rows) > p["fixture_process_ceiling"]
                    ):
                        raise TimeoutError("Source-gate first wall/RSS/process limit")
                    save(out / "result.json", report)
                    if active.poll() is not None:
                        break
                    time.sleep(1)
                row["returncode"] = active.wait()
                row["wall_s"] = time.monotonic() - started
                if any(live(r) for r in seen.values()):
                    raise RuntimeError("Source-gate surviving owned child")
                active = None
            if row["returncode"]:
                raise RuntimeError("Source-gate stage nonzero exit")
            if row["stage"] == "tests":
                from joxing.shared_motor_proof import validate_test_summary

                validate_test_summary(p, (out / "tests.log").read_text())
            row["status"] = "completed"
            save(out / "result.json", report)
        verify()
        report.update(status="completed", criterion_passed=True)
    except Exception as error:  # noqa: BLE001 -- preserve failed and not_run stages
        if active is not None:
            stop(active, seen)
            row.update(
                returncode=active.poll(),
                wall_s=time.monotonic() - started,
                owned_identities=list(seen.values()),
            )
        if row["status"] == "running":
            row["status"] = "failed"
        report.update(
            status="halted",
            criterion_passed=False,
            error=type(error).__name__ + ": " + str(error),
        )
    finally:
        report["wall_s"] = time.monotonic() - began
        report["final_live_recorded_identities"] = [r for r in seen.values() if live(r)]
        report["fixture_bytes"] = size(fixture)
        report["receipt_bytes"] = size(out.parent) - report["fixture_bytes"]
        save(out / "result.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "stages"}))


if __name__ == "__main__":
    main()
