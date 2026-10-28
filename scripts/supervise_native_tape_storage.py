"""One owned offline qualification; stages retain failures and never run physics."""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = "experiments/v0.0.11/native-tape-storage-protocol.json"
RUN = "runs/v0.0.11/native-tape-storage-supervision-01"


def identity(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": int(st[19]),
            "parent_pid": int(st[1]),
            "group": int(st[2]),
            "state": st[0],
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def memory(pid):
    try:
        return (
            int(
                next(
                    s.split()[1]
                    for s in Path(f"/proc/{pid}/status").read_text().splitlines()
                    if s.startswith("VmRSS:")
                )
            )
            * 1024
        )
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        return 0


def write(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if len(raw) > 128 * 1024:
        raise ValueError("Offline supervision report bound")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(raw)
    temporary.replace(path)


def raw_used():
    total = 0
    for f in (ROOT / "runs/v0.0.11").rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except FileNotFoundError:
            pass  # A single owned writer may atomically rename its temporary.
    return total


def run(args):
    p = json.loads((ROOT / PACKET).read_text())
    canonical = hashlib.sha256(
        json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        canonical != args.protocol_sha256
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != args.source_sha
    ):
        raise ValueError("Offline supervisor source/protocol identity")
    for n, d in {**p["source_files"], **p["input_files"]}.items():
        f = Path(n) if n.startswith("/") else ROOT / n
        if (
            f.stat().st_size != d["bytes"]
            or hashlib.sha256(f.read_bytes()).hexdigest() != d["sha256"]
        ):
            raise ValueError("Offline source/input hash: " + n)
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
    if (
        available < p["minimum_available_memory_bytes"]
        or shutil.disk_usage(ROOT).free < p["minimum_free_disk_bytes"]
    ):
        raise ValueError("Offline shared-host capacity prerequisite")
    out = ROOT / RUN
    out.mkdir(parents=True, exist_ok=False)
    python = ROOT / "experiments/v0.0.11/runtime/.venv/bin/python"
    uv = shutil.which("uv") or "/home/ubuntu/.local/bin/uv"
    base = [
        str(python),
        "scripts/qualify_native_tape_storage.py",
        "--protocol-sha256",
        canonical,
        "--source-sha",
        args.source_sha,
    ]
    commands = {
        "sync": [uv, "sync", "--project", "experiments/v0.0.11/runtime", "--locked"],
        "tests": [
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests/test_native_tape_storage.py",
            "tests/test_native_tape_qualification.py",
        ],
        "proof": base + ["--stage", "proof"],
        "audit": base + ["--stage", "audit"],
    }
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[k] = "1"
    result = {
        "schema": "joxing-offline-stored-supervision-v1",
        "source_commit": args.source_sha,
        "protocol_sha256": canonical,
        "process": identity(os.getpid()),
        "status": "running",
        "load_before": os.getloadavg(),
        "available_memory_bytes": available,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "new_physics": 0,
        "environment": {"threads": 1, "PYTHONPATH": str(ROOT)},
        "stages": [
            {"stage": k, "status": "not_run", "command": v} for k, v in commands.items()
        ],
        "cases": [{"id": k, "status": "not_run"} for k in p["corpora"]],
    }
    post_started = None
    active = None
    write(out / "result.json", result)
    try:
        for row in result["stages"]:
            stage = row["stage"]
            if stage == "proof":
                post_started = time.monotonic()
            remaining = p["stage_wall_limits_s"][stage]
            if post_started is not None:
                remaining = min(
                    remaining,
                    p["post_gate_wall_s"]
                    - p["termination_reserve_s"]
                    - (time.monotonic() - post_started),
                )
            if remaining <= 0:
                raise TimeoutError("Offline combined postgate cap")
            began = time.monotonic()
            row.update(status="running", wall_limit_s=remaining)
            with (out / (stage + ".log")).open("w") as log:
                proc = subprocess.Popen(
                    commands[stage],
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active = proc
                row["process"] = identity(proc.pid)
                seen = {}
                peak = 0
                fault = None
                write(out / "result.json", result)
                while True:
                    members = []
                    for item in Path("/proc").iterdir():
                        if not item.name.isdigit():
                            continue
                        ident = identity(int(item.name))
                        if ident and ident["group"] == proc.pid:
                            seen[(ident["pid"], ident["start_ticks"])] = ident
                            members.append(ident)
                    peak = max(peak, sum(memory(i["pid"]) for i in members))
                    used = raw_used()
                    if time.monotonic() - began >= remaining:
                        fault = "stage wall cap"
                    elif (
                        stage in ("proof", "audit", "tests")
                        and peak > p["maximum_rss_bytes"]
                    ):
                        fault = "stage process-group RSS cap"
                    elif used > p["raw_bytes"]:
                        fault = "aggregate owned raw cap"
                    elif (out / (stage + ".log")).stat().st_size > 256 * 1024:
                        fault = "stage log cap"
                    if fault:
                        if members:
                            os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        break
                    if proc.poll() is not None:
                        break
                    row["latest_elapsed_s"] = time.monotonic() - began
                    row["latest_group_rss_bytes"] = sum(
                        memory(i["pid"]) for i in members
                    )
                    write(
                        out / "progress.json",
                        {
                            "stage": stage,
                            "elapsed_s": row["latest_elapsed_s"],
                            "group_rss_bytes": row["latest_group_rss_bytes"],
                            "raw_bytes": used,
                            "observed_monotonic": time.monotonic(),
                        },
                    )
                    time.sleep(0.1)
                row.update(
                    returncode=proc.wait(),
                    wall_s=time.monotonic() - began,
                    max_group_rss_bytes=peak,
                    owned_identities=list(seen.values()),
                    error=fault,
                )
                alive = [
                    i
                    for i in seen.values()
                    if (now := identity(i["pid"]))
                    and now["start_ticks"] == i["start_ticks"]
                    and now["state"] != "Z"
                ]
                row["live_recorded_identities"] = alive
                if alive:
                    os.killpg(proc.pid, signal.SIGKILL)
                    raise RuntimeError("Owned offline descendants survived stage")
                active = None
            if fault:
                row["status"] = "failed"
                raise TimeoutError(fault)
            row["status"] = "completed" if row["returncode"] == 0 else "failed"
            write(out / "result.json", result)
            if stage == "tests":
                text = (out / "tests.log").read_text()
                if row["returncode"] or not re.search(
                    r"(?m)^" + str(p["expected_tests"]) + r" passed in [0-9.]+s$", text
                ):
                    raise ValueError("Offline exact targeted source-test gate")
            elif stage == "sync" and row["returncode"]:
                raise ValueError("Offline locked runtime bootstrap")
            elif stage == "proof":
                proof = json.loads(
                    (
                        ROOT / "runs/v0.0.11/native-tape-storage-01/result.json"
                    ).read_text()
                )
                result["cases"] = proof["rows"]
                # Independently audit completed negative measurements too.
                if row["returncode"] not in (0, 1):
                    raise ValueError("Offline proof exited without bounded result")
            elif stage == "audit" and row["returncode"]:
                raise ValueError("Offline independent audit failed")
        proof_root = ROOT / "runs/v0.0.11/native-tape-storage-01"
        proof = json.loads((proof_root / "result.json").read_text())
        audited = json.loads((proof_root / "independent-audit.json").read_text())
        result.update(
            status="completed",
            criterion_passed=proof["status"] == "passed"
            and audited["all_planned_rows_audited"],
        )
    except Exception as error:  # noqa: BLE001 -- every attempted stage remains visible
        result.update(
            status="halted",
            error=type(error).__name__ + ": " + str(error),
            criterion_passed=False,
        )
    finally:
        if active is not None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
                active.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
            except ProcessLookupError:
                pass
        result["post_gate_wall_s"] = (
            0 if post_started is None else time.monotonic() - post_started
        )
        result["raw_bytes"] = raw_used()
        write(out / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    run(parser.parse_args())
