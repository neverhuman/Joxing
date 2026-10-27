"""Linux-only bounded stage ownership. Freeze this source before invocation."""

import argparse
import datetime
import fcntl
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def identity(pid):
    entry = Path("/proc") / str(pid)
    stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "parent_pid": int(stat[1]),
        "start_ticks": int(stat[19]),
        "state": stat[0],
    }


def descendants(pid):
    rows = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                row = identity(int(path.name))
                rows[row["pid"]] = row
            except (OSError, ValueError, IndexError):
                pass
    owned = {pid}
    while True:
        more = {p for p, r in rows.items() if r["parent_pid"] in owned}
        if more <= owned:
            break
        owned |= more
    return [rows[p] for p in sorted(owned) if p in rows]


def live(row):
    try:
        actual = identity(row["pid"])
        return actual["start_ticks"] == row["start_ticks"] and actual["state"] != "Z"
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--expected-tests", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--uv", default="/home/ubuntu/.local/bin/uv")
    args = parser.parse_args()
    root = Path.cwd()
    out = root / "runs/v0.0.10/supervision-01"
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".supervisor.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"
    runtime = [args.uv, "run", "--project", "experiments/v0.0.10/runtime", "--locked"]
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "supervisor": identity(os.getpid()),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "protocol_sha256": args.protocol_sha256,
        "expected_linux_tests": args.expected_tests,
        "status": "running",
        "stages": [],
        "threads": 1,
    }

    def save():
        temporary = out / "supervisor.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out / "supervisor.json")

    stages = [
        (
            "sync",
            [args.uv, "sync", "--project", "experiments/v0.0.10/runtime", "--locked"],
            180,
        ),
        ("tests", runtime + ["pytest", "-q"], 420),
        (
            "D2",
            runtime
            + [
                "python",
                "scripts/train_motor_residual.py",
                "--protocol",
                "experiments/v0.0.10/protocol.json",
                "--protocol-sha256",
                args.protocol_sha256,
                "--inputs",
                str(args.inputs),
                "--out",
                "runs/v0.0.10/pilot-01",
            ],
            2130,
        ),
        (
            "audit",
            runtime
            + [
                "python",
                "scripts/audit_motor_residual.py",
                "--run",
                "runs/v0.0.10/pilot-01",
            ],
            180,
        ),
    ]
    save()
    try:
        for name, command, timeout in stages:
            if subprocess.run(
                ["git", "diff", "--quiet", "HEAD"], check=False
            ).returncode:
                raise RuntimeError("Frozen checkout source drift")
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
                while proc.poll() is None and time.monotonic() - began < timeout:
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
            remaining = [r for r in owned.values() if live(r)]
            # Resource-tracker/zombie exits get a bounded grace period.
            for _ in range(5):
                if not remaining:
                    break
                time.sleep(1)
                remaining = [r for r in remaining if live(r)]
            row.update(
                returncode=code,
                wall_s=time.monotonic() - began,
                live_recorded_processes=remaining,
            )
            if remaining:
                raise RuntimeError("Owned child did not exit; do not launch next stage")
            if name == "tests":
                text = (out / "tests.log").read_text()
                counts = re.findall(r"(\d+) passed", text)
                if code == 0 and counts != [str(args.expected_tests)]:
                    code = 86
                    row["reason"] = "Exact full-suite count gate"
            row["status"] = "completed" if code == 0 else "failed"
            save()
            if code and name != "D2":
                raise RuntimeError(name + " stage failed")
        report["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- retain supervisor failure receipt
        report.update(status="halted", reason=type(error).__name__ + ": " + str(error))
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        save()
        lease.close()


if __name__ == "__main__":
    main()
