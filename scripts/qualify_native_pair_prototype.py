"""Bounded local-Mac prototype owner. No services or unrelated processes touched."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(v):
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def processes():
    text = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,rss=,stat=,lstart=,command="], text=True
    )
    rows = {}
    for line in text.splitlines():
        bits = line.split(maxsplit=9)
        if len(bits) == 10:
            rows[int(bits[0])] = {
                "pid": int(bits[0]),
                "parent_pid": int(bits[1]),
                "rss_bytes": int(bits[2]) * 1024,
                "state": bits[3],
                "start": " ".join(bits[4:9]),
                "command": bits[9],
            }
    return rows


def descendants(rows, pid):
    owned = {pid}
    while True:
        more = {i for i, r in rows.items() if r["parent_pid"] in owned}
        if more <= owned:
            break
        owned |= more
    return [rows[i] for i in owned if i in rows]


def admission(p):
    physical = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    cpu = int(subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True))
    pressure = subprocess.check_output(["memory_pressure", "-Q"], text=True)
    free = int(re.search(r"memory free percentage: (\d+)%", pressure)[1])
    load = os.getloadavg()
    v = {
        "monotonic_s": time.monotonic(),
        "physical_memory_bytes": physical,
        "free_percentage": free,
        "available_pressure_estimate_bytes": physical * free // 100,
        "cpu_count": cpu,
        "load": load,
        "free_disk_bytes": __import__("shutil").disk_usage(ROOT).free,
        "definition": "Mac memory_pressure free percentage times physical memory; not Linux MemAvailable/PSI",
    }
    if (
        load[0] > 0.5 * cpu
        or v["available_pressure_estimate_bytes"] < p["minimum_memory_bytes"]
        or v["free_disk_bytes"] < p["minimum_disk_bytes"]
    ):
        raise RuntimeError("Local Mac capacity admission first limit: " + json.dumps(v))
    return v


def main():
    ap = argparse.ArgumentParser()
    for n in ("packet", "source-sha", "protocol-sha256"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--epoch-start-monotonic", required=True, type=float)
    a = ap.parse_args()
    p = json.loads((ROOT / a.packet).read_text())
    epoch = a.epoch_start_monotonic
    if (
        p["schema"] != "joxing-native-pair-mac-prototype-v1"
        or digest(p) != a.protocol_sha256
        or platform.system() != "Darwin"
        or not 0 <= time.monotonic() - epoch < 60
    ):
        raise ValueError("Explicit source/protocol/platform/original epoch")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != a.source_sha
        or subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True
        ).strip()
    ):
        raise ValueError("Frozen source HEAD required")
    for n, m in {
        **p["source_files"],
        **p["input_files"],
        **p["runtime_file_hashes"],
    }.items():
        f = Path(n) if n.startswith("/") else ROOT / n
        if f.stat().st_size != m["bytes"] or sha(f) != m["sha256"]:
            raise ValueError("Frozen input/source/runtime drift:" + n)
    for n, s in p["upstream_commits"].items():
        if (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party" / n, text=True
            ).strip()
            != s
        ):
            raise ValueError("Upstream pin drift")
    versions = {
        "python": sys.version.split()[0],
        **{
            n: importlib.metadata.version(n)
            for n in p["runtime_versions"]
            if n != "python"
        },
    }
    if versions != p["runtime_versions"]:
        raise ValueError("Explicit local runtime profile drift")
    out = ROOT / p["output"]
    out.mkdir(parents=True, exist_ok=False)
    me = processes()[os.getpid()]
    value = {
        "schema": p["schema"],
        "status": "running",
        "source_commit": a.source_sha,
        "protocol_sha256": a.protocol_sha256,
        "owner": me,
        "original_epoch": epoch,
        "stages": [],
        "rows": [
            {"id": "source", "status": "not_run"},
            {"id": "native-ppo", "status": "not_run"},
            {"id": "before-render", "status": "not_run"},
            {"id": "after-render", "status": "not_run"},
            {"id": "audit", "status": "not_run"},
        ],
    }
    active = None
    seen = {}
    env = os.environ | {
        n: "1"
        for n in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    env.update(
        PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT) + ":" + str(ROOT / "scripts"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("MUJOCO_GL", None)

    def save():
        tmp = out / "owner.tmp"
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        tmp.replace(out / "owner.json")

    def usage():
        return sum(f.stat().st_size for f in out.rglob("*") if f.is_file())

    def stage(name, cmd, limit):
        nonlocal active
        row = next(r for r in value["rows"] if r["id"] == name)
        row["status"] = "unconfirmed"
        began = time.monotonic()
        record = {"name": name, "command": cmd, "started_monotonic_s": began}
        value["stages"].append(record)
        with (out / (name + ".log")).open("x") as log:
            active = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                allrows = processes()
                owned = descendants(allrows, active.pid)
                for i in owned:
                    seen[(i["pid"], i["start"])] = i
                charge = allrows[me["pid"]]["rss_bytes"] + sum(
                    i["rss_bytes"] for i in owned
                )
                count = 1 + len(owned)
                record.update(
                    elapsed_s=time.monotonic() - began,
                    peak_sampled_rss_bytes=max(
                        record.get("peak_sampled_rss_bytes", 0), charge
                    ),
                    peak_processes=max(record.get("peak_processes", 0), count),
                    bytes=usage(),
                )
                save()
                if (
                    charge > p["owned_rss_bytes"]
                    or count > p["process_ceiling"]
                    or record["bytes"] > p["raw_bytes"]
                ):
                    raise RuntimeError("Owned RSS/process/raw first limit")
                if (
                    record["elapsed_s"] > limit - 2
                    or time.monotonic() - epoch > p["whole_wall_s"] - 3
                ):
                    raise TimeoutError("Original prototype wall first limit")
                if active.poll() is not None:
                    break
                time.sleep(0.15)
            rc = active.wait()
            record.update(returncode=rc, wall_s=time.monotonic() - began)
            active = None
        if rc:
            raise RuntimeError("Registered stage nonzero:" + name)
        row["status"] = "completed"
        save()

    try:
        value["admission"] = [admission(p)]
        save()
        time.sleep(10)
        value["admission"].append(admission(p))
        save()
        if time.monotonic() - epoch > 60:
            raise TimeoutError("Setup60s first limit")
        stage(
            "source",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--junitxml",
                str(out / "source.xml"),
                "--basetemp",
                str(out / "source-fixtures"),
                *p["test_modules"],
            ],
            p["source_wall_s"],
        )
        sys.path.insert(0, str(ROOT / "scripts"))
        from source_shard_support import check_junit

        source = check_junit((out / "source.xml").read_text(), p["test_node_ids"], [])
        value["source_gate"] = source
        if not source["qualified"]:
            raise ValueError("Complete prototype source gate")
        stage(
            "native-ppo",
            [
                sys.executable,
                "scripts/run_native_pair_prototype.py",
                "--packet",
                a.packet,
                "--source-sha",
                a.source_sha,
                "--protocol-sha256",
                a.protocol_sha256,
            ],
            p["native_wall_s"],
        )
        for label in ("before", "after"):
            stage(
                label + "-render",
                [
                    sys.executable,
                    "scripts/render_native_pair_motor.py",
                    "--folder",
                    str(out / "native" / (label + "-eval")),
                    "--label",
                    label,
                    "--source-sha",
                    a.source_sha,
                ],
                p["render_wall_s"],
            )
        stage(
            "audit",
            [
                sys.executable,
                "scripts/audit_native_pair_prototype.py",
                "--packet",
                a.packet,
            ],
            p["audit_wall_s"],
        )
        value["status"] = "completed"
    except BaseException as error:  # noqa: BLE001 -- preserve actual prefixes/faults
        value.update(status="halted", error=type(error).__name__ + ": " + str(error))
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=2)
        for r in value["rows"]:
            if r["status"] == "unconfirmed":
                r["status"] = "failed"
    finally:
        now = processes()
        live = [
            r
            for key, r in seen.items()
            if r["pid"] in now
            and now[r["pid"]]["start"] == r["start"]
            and not now[r["pid"]]["state"].startswith("Z")
        ]
        value.update(
            original_epoch_elapsed_s=time.monotonic() - epoch,
            owned_observed_identities=list(seen.values()),
            final_live=live,
            final_bytes=usage(),
        )
        if (
            live
            or value["original_epoch_elapsed_s"] > p["whole_wall_s"]
            or value["final_bytes"] > p["raw_bytes"]
        ):
            value["status"] = "halted"
            value["final_limit_error"] = "Exit/wall/raw first limit"
        save()
    print(
        json.dumps(
            {
                k: value.get(k)
                for k in (
                    "status",
                    "error",
                    "original_epoch_elapsed_s",
                    "final_live",
                    "final_bytes",
                )
            }
        )
    )
    if value["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
