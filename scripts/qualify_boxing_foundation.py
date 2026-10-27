"""Bounded Linux regression gate only. Never starts a formal rollout or learner."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from supervise_motor_pilot import descendants, identity, live


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Source-host gate requires Linux")
    root = Path.cwd()
    p = json.loads(
        (root / "experiments/v0.0.11/foundation-qualification.json").read_text()
    )
    if (
        canonical(p) != args.protocol_sha256
        or p["schema"] != "joxing-shared-foundation-qualification-v1"
    ):
        raise ValueError("Unregistered qualification packet")
    frozen = json.loads(
        subprocess.check_output(
            ["git", "show", "HEAD:experiments/v0.0.11/foundation-qualification.json"],
            text=True,
        )
    )
    if frozen != p:
        raise ValueError("Protocol not committed at HEAD")
    for name, sha in {**p["source_files"], **p["upstream_files"]}.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen file drift: " + name)
    ancestors = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = identity(pid)["parent_pid"]
    unknown = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) in ancestors:
            continue
        try:
            command = (path / "cmdline").read_bytes()
            if "/joxing/" in str((path / "cwd").resolve()) and (
                b"python" in command or b"/uv " in command
            ):
                unknown.append(int(path.name))
        except OSError:
            pass
    if unknown:
        raise RuntimeError("Another owned Joxing process exists: " + str(unknown))
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
    disk = shutil.disk_usage(root).free
    if memory < 2 * p["minimum_memory_bytes"] or disk < 2 * p["minimum_disk_bytes"]:
        raise RuntimeError("Initial shared-host headroom below registered floors")
    out = root / "runs/v0.0.11/foundation-qualification-01"
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".foundation-qualification.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[key] = "1"
    report = {
        "schema": p["schema"],
        "protocol_sha256": args.protocol_sha256,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "pid": identity(os.getpid()),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "running",
        "initial_available_memory_bytes": memory,
        "initial_disk_free_bytes": disk,
        "stages": [],
        "formal_v11_physics": 0,
        "formal_v11_optimizer_updates": 0,
        "regression_fixtures": "Physics/gradient/process fixtures are separate from formal outcome counters; global16worker limit applies",
    }

    def save():
        tmp = out / "result.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(out / "result.json")

    save()
    uv = "/home/ubuntu/.local/bin/uv"
    runtime = "experiments/v0.0.11/runtime"
    stages = [
        ("sync", [uv, "sync", "--project", runtime, "--locked"], 180),
        ("tests", [uv, "run", "--project", runtime, "--locked", "pytest", "-q"], 420),
    ]
    try:
        for name, command, limit in stages:
            if subprocess.run(
                ["git", "diff", "--quiet", "HEAD"], check=False
            ).returncode:
                raise RuntimeError("Executing source drift")
            began = time.monotonic()
            with (out / (name + ".log")).open("w") as log:
                proc = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                row = {
                    "stage": name,
                    "command": command,
                    "process": identity(proc.pid),
                    "status": "running",
                }
                report["stages"].append(row)
                owned = {}
                while proc.poll() is None and time.monotonic() - began < limit:
                    for child in descendants(proc.pid):
                        owned[(child["pid"], child["start_ticks"])] = child
                    row["owned_processes"] = list(owned.values())
                    save()
                    time.sleep(2)
                if proc.poll() is None:
                    row["timeout"] = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
            code = proc.returncode
            remaining = [v for v in owned.values() if live(v)]
            for _ in range(5):
                if not remaining:
                    break
                time.sleep(1)
                remaining = [v for v in remaining if live(v)]
            row.update(
                returncode=code,
                wall_s=time.monotonic() - began,
                live_recorded_processes=remaining,
            )
            if name == "tests" and (
                code != 0
                or re.findall(r"(\d+) passed", (out / "tests.log").read_text())
                != [str(p["expected_tests"])]
            ):
                code = 86
            row["status"] = "completed" if not code and not remaining else "failed"
            save()
            if row["status"] != "completed":
                raise RuntimeError(name + " qualification failed")
        if subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode:
            raise RuntimeError("Post-test source drift")
        report["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- preserve all qualification failures
        report.update(status="halted", reason=type(error).__name__ + ": " + str(error))
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        save()
        lease.close()


if __name__ == "__main__":
    main()
