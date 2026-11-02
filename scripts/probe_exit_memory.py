"""Observe owned Linux exit memory. Diagnostic only; changes no watchdog policy."""

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

from supervise_motor_pilot import identity, live


def stat_fields(path):
    values = path.read_text().rsplit(")", 1)[1].split()
    return {
        "state": values[0],
        "parent_pid": int(values[1]),
        "flags": int(values[6]),
        "start_ticks": int(values[19]),
    }


def snapshot(bound, proc_root=Path("/proc")):
    folder = proc_root / str(bound["pid"])
    before = stat_fields(folder / "stat")
    if before["start_ticks"] != bound["start_ticks"]:
        raise ValueError("Reused probe PID")
    status = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in (folder / "status").read_text().splitlines()
        if ":" in line
    }
    statm = [int(s) for s in (folder / "statm").read_text().split()]
    if len(statm) != 7:
        raise ValueError("Malformed owned statm")
    after = stat_fields(folder / "stat")
    if after["start_ticks"] != bound["start_ticks"]:
        raise ValueError("Reused probe PID after read")
    rss = int(status["VmRSS"].split()[0]) * 1024 if "VmRSS" in status else None
    threads = []
    if rss is None:
        for task in sorted((folder / "task").iterdir()):
            try:
                threads.append({"tid": int(task.name), **stat_fields(task / "stat")})
            except FileNotFoundError:
                pass
    return {
        "monotonic_s": time.monotonic(),
        "pid": bound["pid"],
        "before": before,
        "after": after,
        "vmrss_bytes": rss,
        "vmsize_present": "VmSize" in status,
        "statm_pages": statm,
        "page_size": os.sysconf("SC_PAGE_SIZE"),
        "threads": threads,
    }


def exiting_empty_mm(row):
    # Hypothesis classification only; NOT a production RSS fallback.
    return (
        row["vmrss_bytes"] is None
        and not row["vmsize_present"]
        and row["statm_pages"][0] == row["statm_pages"][1] == 0
        and bool(row["threads"])
        and all(t["flags"] & 4 for t in row["threads"])
        and bool(row["after"]["flags"] & 4)
    )


def run_case(out, mode, allocation_bytes, wall_s=10):
    out = Path(out)
    out.mkdir(exist_ok=False)
    code = "import os,sys,time; a=bytearray(int(sys.argv[1])); print('ready',flush=True); sys.stdin.buffer.read(1); os._exit(0) if sys.argv[2]=='direct_exit' else sys.exit(0)"
    command = [sys.executable, "-c", code, str(allocation_bytes), mode]
    child = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=False,
        start_new_session=True,
    )
    bound = identity(child.pid)
    began = time.monotonic()
    rows = []
    report = {
        "schema": "joxing-exit-memory-probe-v1",
        "mode": mode,
        "allocation_bytes": allocation_bytes,
        "process": bound,
        "parent": identity(os.getpid()),
        "command": command,
        "kernel": os.uname().release,
        "native_physics_steps": 0,
        "learning_updates": 0,
        "status": "running",
    }
    try:
        if (
            not select.select([child.stdout], [], [], 5)[0]
            or child.stdout.readline() != b"ready\n"
        ):
            raise RuntimeError("Owned allocation readiness")
        child.stdin.write(b"x")
        child.stdin.flush()
        with (out / "samples.jsonl").open("w") as raw:
            while True:
                if time.monotonic() - began > wall_s:
                    raise TimeoutError("Exit probe first wall limit")
                try:
                    row = snapshot(bound)
                except (FileNotFoundError, ProcessLookupError):
                    break
                rows.append(row)
                raw.write(json.dumps(row) + "\n")
                raw.flush()
                if raw.tell() > 512 * 1024:
                    raise RuntimeError("Exit probe raw credit")
                if row["after"]["state"] in ("Z", "X"):
                    break
                time.sleep(0.001)
        child.wait(timeout=1)
        if child.returncode:
            raise RuntimeError("Owned exit returncode")
        missing = [
            r
            for r in rows
            if r["vmrss_bytes"] is None and r["after"]["state"] not in ("Z", "X")
        ]
        report.update(
            status="completed",
            samples=len(rows),
            missing_nonterminal_samples=len(missing),
            missing_nonterminal_classified=sum(exiting_empty_mm(r) for r in missing),
            first_missing=missing[0] if missing else None,
            peak_observed_rss_bytes=max(
                (r["vmrss_bytes"] or 0 for r in rows), default=0
            ),
            hypothesis="supported-synthetic-exit"
            if missing and all(exiting_empty_mm(r) for r in missing)
            else "not_observed"
            if not missing
            else "unclassified_missing_rss",
        )
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        if child.poll() is None:
            if live(bound):
                child.kill()
            child.wait(timeout=2)
        report.update(
            wall_s=time.monotonic() - began,
            returncode=child.returncode,
            process_reaped=True,
        )
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
