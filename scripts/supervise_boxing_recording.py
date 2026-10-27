"""One bounded Linux gate followed by the registered seven-case recording proof."""

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
import time
from pathlib import Path

from supervise_motor_pilot import descendants, identity, live


def canonical(v):
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    file = "experiments/v0.0.11/recording-protocol.json"
    p = json.loads((root / file).read_text())
    if (
        canonical(p) != args.protocol_sha256
        or p["schema"] != "joxing-shared-recording-protocol-v1"
    ):
        raise ValueError("Foreign packet")
    if (
        json.loads(subprocess.check_output(["git", "show", "HEAD:" + file], text=True))
        != p
    ):
        raise ValueError("Protocol not committed")

    def verify_source():
        for name, sha in {**p["source_files"], **p["upstream_files"]}.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
                raise ValueError("Frozen source drift: " + name)
        if subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode:
            raise ValueError("Source diff")

    verify_source()
    ancestors = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = identity(pid)["parent_pid"]
    unknown = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestors:
            continue
        try:
            cmd = (proc / "cmdline").read_bytes()
            if "/joxing/" in str((proc / "cwd").resolve()) and (
                b"python" in cmd or b"/uv " in cmd
            ):
                unknown.append(int(proc.name))
        except OSError:
            pass
    if unknown:
        raise RuntimeError("Another owned process: " + str(unknown))

    def capacity(multiplier=1):
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
        if (
            memory < multiplier * p["minimum_memory_bytes"]
            or disk < multiplier * p["minimum_disk_bytes"]
        ):
            raise RuntimeError("Resource floor")
        run = root / "runs/v0.0.11/recording-proof-01"
        if (
            run.exists()
            and sum(f.stat().st_size for f in run.rglob("*") if f.is_file())
            > p["raw_bytes"]
        ):
            raise RuntimeError("D0raw cap")
        return {
            "available_memory_bytes": memory,
            "disk_free_bytes": disk,
            "load": os.getloadavg(),
        }

    initial = capacity(2)
    out = root / "runs/v0.0.11/recording-supervision-01"
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".recording.lock").open("a+")
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
    os.environ["MUJOCO_GL"] = "egl"
    uv = "/home/ubuntu/.local/bin/uv"
    prefix = [uv, "run", "--project", "experiments/v0.0.11/runtime", "--locked"]
    run = "runs/v0.0.11/recording-proof-01"
    commands = [
        ("sync", [uv, "sync", "--project", "experiments/v0.0.11/runtime", "--locked"]),
        ("tests", prefix + ["pytest", "-q"]),
        (
            "D0",
            prefix
            + [
                "python",
                "scripts/qualify_boxing_env.py",
                "--protocol-sha256",
                args.protocol_sha256,
            ],
        ),
        ("audit", prefix + ["python", "scripts/audit_boxing_env.py", "--run", run]),
        (
            "render",
            prefix
            + [
                "python",
                "scripts/render_bout_recording.py",
                "--recording",
                run + "/source-90101/recording",
                "--out",
                run + "/first-diagnostic.mp4",
                "--raw-budget-bytes",
                str(p["raw_bytes"]),
            ],
        ),
        (
            "decode",
            prefix + ["python", "scripts/validate_bout_render.py", "--run", run],
        ),
    ]
    report = {
        "schema": "joxing-shared-recording-supervisor-v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol_sha256": args.protocol_sha256,
        "process": identity(os.getpid()),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "running",
        "initial_capacity": initial,
        "stages": [
            {"stage": name, "command": cmd, "status": "not_run", "reason": "pending"}
            for name, cmd in commands
        ],
    }

    def save():
        tmp = out / "result.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(out / "result.json")

    post_gate = None
    try:
        for row in report["stages"]:
            verify_source()
            capacity()
            if row["stage"] == "D0":
                post_gate = time.monotonic()
            began = time.monotonic()
            limit = p["stage_wall_limits_s"][row["stage"]]
            if post_gate is not None:
                limit = min(
                    limit, p["post_gate_wall_s"] - (time.monotonic() - post_gate)
                )
            if limit <= 0:
                raise TimeoutError("Post-gate cumulative wall cap")
            row.update(status="running", reason=None)
            failure = None
            owned = {}
            with (out / (row["stage"] + ".log")).open("w") as log:
                proc = subprocess.Popen(
                    row["command"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                row["process"] = identity(proc.pid)
                while proc.poll() is None:
                    for child in descendants(proc.pid):
                        owned[(child["pid"], child["start_ticks"])] = child
                    row["owned_processes"] = list(owned.values())
                    try:
                        row["latest_capacity"] = capacity()
                        if time.monotonic() - began >= limit:
                            raise TimeoutError("Stage/cumulative deadline")
                        if (
                            sum(live(x) for x in owned.values())
                            > p["fixture_process_ceiling"]
                        ):
                            raise RuntimeError("Owned process ceiling")
                    except Exception as error:  # noqa: BLE001 -- stop only owned group
                        failure = type(error).__name__ + ": " + str(error)
                        break
                    save()
                    time.sleep(2)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
            active = [x for x in owned.values() if live(x)]
            for _ in range(5):
                if not active:
                    break
                time.sleep(1)
                active = [x for x in active if live(x)]
            row.update(
                returncode=proc.returncode,
                wall_s=time.monotonic() - began,
                live_recorded_processes=active,
            )
            if row["stage"] == "tests" and re.findall(
                r"(\d+) passed", (out / "tests.log").read_text()
            ) != [str(p["expected_tests"])]:
                failure = "Registered test count not met"
            if proc.returncode or active or failure:
                row.update(
                    status="failed", reason=failure or "Nonzero exit or live child"
                )
                save()
                raise RuntimeError(row["stage"] + " failed: " + str(row["reason"]))
            row["status"] = "completed"
            save()
        verify_source()
        capacity()
        report["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- retain complete stage denominator
        report.update(status="halted", reason=type(error).__name__ + ": " + str(error))
        for row in report["stages"]:
            if row["status"] == "not_run":
                row["reason"] = "stopped after earlier failure"
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        if post_gate is not None:
            report["post_gate_wall_s"] = time.monotonic() - post_gate
        save()
        lease.close()


if __name__ == "__main__":
    main()
