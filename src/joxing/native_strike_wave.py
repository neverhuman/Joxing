"""Bounded dependency chain: scoring qualification, then the fixed native probe."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path


def chain_stages(stages, execute, save):
    rows = [{**s, "status": "not_run", "reason": "pending"} for s in stages]
    for row in rows:
        row.update(status="running", reason=None)
        save(rows)
        evidence = execute(row)
        row.update(evidence)
        if (
            evidence.get("returncode") != 0
            or evidence.get("supervisor_status") != "completed"
            or evidence.get("error") is not None
            or evidence.get("live_recorded_processes")
        ):
            row.update(
                status="failed",
                reason=evidence.get("error") or "Child supervisor did not complete",
            )
            for pending in rows:
                if pending["status"] == "not_run":
                    pending["reason"] = "prerequisite did not complete"
            save(rows)
            return rows
        row.update(status="completed")
        save(rows)
    return rows


def run(root, source_sha, a_sha, b_sha):
    import fcntl

    from joxing_process_ownership import unknown_joxing_processes
    from supervise_motor_pilot import descendants, identity, live

    root = Path(root)
    out = root / "runs/v0.0.11/native-strike-wave-01"
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".native-strike-wave.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    owner = identity(os.getpid())
    ancestors = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = identity(pid)["parent_pid"]
    if unknown_joxing_processes(owner, ignored_pids=ancestors):
        raise RuntimeError("Another Joxing owner is active")
    runtime = str(root / "experiments/v0.0.11/runtime/.venv/bin/python")
    stages = [
        {
            "name": "scoring-repair",
            "command": [
                runtime,
                "scripts/supervise_scored_contacts.py",
                "--protocol-sha256",
                a_sha,
                "--source-sha",
                source_sha,
            ],
            "wall_limit_s": 1140,
            "supervision": "runs/v0.0.11/scored-contact-supervision-01/result.json",
            "proof_result": "runs/v0.0.11/scored-contact-proof-01/result.json",
            "planned_proof_rows": [
                {"id": n, "status": "not_run"}
                for n in json.loads(
                    (
                        root
                        / "experiments/v0.0.11/scored-contact-legacy-repair-protocol.json"
                    ).read_text()
                )["rows"]
            ],
        },
        {
            "name": "native-source-replay-recording",
            "command": [
                runtime,
                "scripts/supervise_native_strike_probe.py",
                "--protocol-sha256",
                b_sha,
                "--source-sha",
                source_sha,
            ],
            "wall_limit_s": 55,
            "supervision": "runs/v0.0.11/native-strike-supervision-01/result.json",
            "proof_result": "runs/v0.0.11/native-strike-proof-01/result.json",
            "planned_proof_rows": [
                {"id": "fixed-source-10001-A", "status": "not_run"},
                {"id": "actual-mid-203-continuation", "status": "not_run"},
            ],
        },
    ]
    report = {
        "schema": "joxing-native-strike-wave-v1",
        "source_commit": source_sha,
        "a_protocol_sha256": a_sha,
        "native_protocol_sha256": b_sha,
        "process": owner,
        "status": "running",
        "stages": [],
    }

    def save(rows):
        report["stages"] = rows
        value = (json.dumps(report, indent=2) + "\n").encode()
        used = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        if used + len(value) > 65536:
            raise TimeoutError("Wave orchestration receipt credit")
        temp = out / "result.tmp"
        temp.write_bytes(value)
        temp.replace(out / "result.json")

    def execute(row):
        began = time.monotonic()
        owned = {}
        error = None
        with (out / (row["name"] + ".log")).open("w") as log:
            proc = subprocess.Popen(
                row["command"],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            row["process"] = identity(proc.pid)
            owned[(row["process"]["pid"], row["process"]["start_ticks"])] = row[
                "process"
            ]
            while proc.poll() is None:
                for child in descendants(proc.pid):
                    owned[(child["pid"], child["start_ticks"])] = child
                row["owned_identities"] = list(owned.values())
                save(report["stages"])
                if time.monotonic() - began >= row["wall_limit_s"]:
                    error = "Wave stage deadline"
                    break
                time.sleep(2)
            if proc.poll() is None:
                # Stop only identities observed under this stage, never a shared
                # host service or a process discovered merely by its name.
                for child in reversed(list(owned.values())):
                    if live(child):
                        try:
                            os.kill(child["pid"], signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                for child in owned.values():
                    if live(child):
                        try:
                            os.kill(child["pid"], signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        remaining = [x for x in owned.values() if live(x)]
        file = root / row["supervision"]
        supervision = json.loads(file.read_text()) if file.exists() else {}
        proof_file = root / row["proof_result"]
        if proof_file.exists():
            row["actual_proof_rows"] = json.loads(proof_file.read_text())["rows"]
        return {
            "returncode": proc.returncode,
            "wall_s": time.monotonic() - began,
            "supervisor_status": supervision.get("status"),
            "live_recorded_processes": remaining,
            "error": error
            or ("Live owned process after child exit" if remaining else None),
        }

    try:
        rows = chain_stages(stages, execute, save)
        report["status"] = (
            "completed" if all(r["status"] == "completed" for r in rows) else "halted"
        )
        save(rows)
    finally:
        lease.close()
    return report
