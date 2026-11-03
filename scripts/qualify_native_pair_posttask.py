"""Separate bounded replay/media audit; never resumes the invalid training job."""

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import qualify_native_pair_prototype as base
from native_pair_guarded_stage import usage

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--epoch", type=float, required=True)
    a = ap.parse_args()
    p = json.loads((ROOT / a.packet).read_text())
    started = a.epoch
    if (
        p["schema"] != "joxing-native-pilot-posttask-v1"
        or time.monotonic() - started > 60
    ):
        raise ValueError("Fresh posttask epoch")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.source_sha
    ):
        raise ValueError("Posttask frozen source")
    old = Path(p["input_root"])
    original_repo = old.parents[2]
    physical = json.loads(
        (
            original_repo / "experiments/v0.0.11/native-pair-motor-pilot-03.json"
        ).read_text()
    )
    if base.digest(physical) != p["physical_protocol_sha256"]:
        raise ValueError("Original numerical protocol identity")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=original_repo, text=True
        ).strip()
        != p["original_source_commit"]
    ):
        raise ValueError("Original physical source HEAD")
    for name, record in {
        **physical["source_files"],
        **physical["runtime_file_hashes"],
        **physical["input_files"],
    }.items():
        f = Path(name) if name.startswith("/") else original_repo / name
        if base.sha(f) != record["sha256"]:
            raise ValueError("Original physical source/runtime/input drift:" + name)
    out = ROOT / p["output"]
    out.mkdir(parents=True, exist_ok=False)
    for n, r in p["input_files"].items():
        f = old / n
        if f.stat().st_size != r["bytes"] or base.sha(f) != r["sha256"]:
            raise ValueError("Closed original input drift:" + n)
    for n, r in p["source_files"].items():
        f = ROOT / n
        if base.sha(f) != r["sha256"]:
            raise ValueError("Posttask source drift:" + n)
    ids = [
        "audit",
        *[f"{s}-{phase}-render" for s in (761, 983) for phase in ("before", "after")],
    ]
    value = {
        "schema": p["schema"],
        "status": "running",
        "source_commit": a.source_sha,
        "protocol_sha256": base.digest(p),
        "original_attempt_status": "invalid_render_timeout",
        "owner": base.processes()[os.getpid()],
        "rows": [{"id": n, "status": "not_run"} for n in ids],
        "stages": [],
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
        PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT / "scripts"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("MUJOCO_GL", None)

    def save():
        blob = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        raw, text = usage(out)
        if raw + len(blob) > p["raw_bytes"] or text + len(blob) > p["text_bytes"]:
            raise TimeoutError("Posttask publication credit")
        f = out / "owner.json.tmp"
        f.write_bytes(blob)
        f.replace(out / "owner.json")

    def stage(name, command, seconds, filemax):
        nonlocal active
        row = next(r for r in value["rows"] if r["id"] == name)
        row["status"] = "unconfirmed"
        begin = time.monotonic()
        cmd = [
            sys.executable,
            "scripts/native_pair_guarded_stage.py",
            "--root",
            str(out),
            "--raw",
            str(p["raw_bytes"]),
            "--text",
            str(p["text_bytes"]),
            "--file-max",
            str(filemax),
            "--",
            *command,
        ]
        r = {"name": name, "command": cmd, "start": begin}
        value["stages"].append(r)
        save()
        active = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        selector.register(active.stdout, selectors.EVENT_READ)
        with (out / (name + ".log")).open("xb") as log:
            while True:
                for key, _ in selector.select(0.1):
                    blob = os.read(key.fileobj.fileno(), 16384)
                    if blob:
                        raw, text = usage(out)
                        if (
                            raw + len(blob) + 65536 > p["raw_bytes"]
                            or text + len(blob) + 65536 > p["text_bytes"]
                        ):
                            raise TimeoutError("Posttask log credit")
                        log.write(blob)
                        log.flush()
                    else:
                        selector.unregister(key.fileobj)
                rows = base.processes()
                owned = base.descendants(rows, active.pid)
                for item in owned:
                    seen[(item["pid"], item["start"])] = item
                charge = rows[value["owner"]["pid"]]["rss_bytes"] + sum(
                    x["rss_bytes"] for x in owned
                )
                r.update(
                    elapsed_s=time.monotonic() - begin,
                    peak_rss_bytes=max(r.get("peak_rss_bytes", 0), charge),
                    peak_processes=max(r.get("peak_processes", 0), 1 + len(owned)),
                    usage=usage(out),
                )
                save()
                if charge > p["owned_rss_bytes"] or len(owned) + 1 > 12:
                    raise RuntimeError("Posttask resource cap")
                if (
                    r["elapsed_s"] > seconds - 3
                    or time.monotonic() - started > p["whole_wall_s"] - 5
                ):
                    raise TimeoutError("Posttask original stage/whole deadline")
                if active.poll() is not None and not selector.get_map():
                    break
            rc = active.wait()
            active = None
            selector.close()
            r["returncode"] = rc
        if rc:
            raise RuntimeError("Posttask stage failed:" + name)
        row["status"] = "completed"
        save()

    try:
        admission = {
            "minimum_memory_bytes": 16 * 1024**3,
            "minimum_disk_bytes": 10 * 1024**3,
        }
        value["admission"] = [base.admission(admission)]
        save()
        time.sleep(10)
        value["admission"].append(base.admission(admission))
        save()
        stage(
            "audit",
            [
                "scripts/audit_native_pair_pilot.py",
                "--packet",
                str(
                    old.parents[2]
                    / "experiments/v0.0.11/native-pair-motor-pilot-03.json"
                ),
                "--input-root",
                str(old),
                "--report-root",
                str(out / "audit"),
            ],
            p["audit_wall_s"],
            64 * 1024**2,
        )
        for s in (761, 983):
            for phase in ("before", "after"):
                stage(
                    f"{s}-{phase}-render",
                    [
                        "scripts/render_native_pair_motor_compact.py",
                        "--folder",
                        str(old / "native" / f"seed-{s}" / f"{phase}-71000"),
                        "--report",
                        str(out / f"{s}-{phase}"),
                        "--label",
                        phase,
                        "--source-sha",
                        p["original_source_commit"],
                    ],
                    30,
                    4 * 1024**2,
                )
        for n, r in p["input_files"].items():
            if base.sha(old / n) != r["sha256"]:
                raise ValueError("Original input changed during posttask:" + n)
        value["status"] = "completed"
    except BaseException as error:  # noqa: BLE001 -- preserve invalid prefixes/owned cleanup
        value.update(status="halted", error=type(error).__name__ + ": " + str(error))
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=2)
        for row in value["rows"]:
            if row["status"] == "unconfirmed":
                row["status"] = "failed"
    finally:
        now = base.processes()
        live = [
            r
            for (pid, start), r in seen.items()
            if pid in now
            and now[pid]["start"] == start
            and not now[pid]["state"].startswith("Z")
        ]
        value.update(
            wall_s=time.monotonic() - started,
            owned_observed_identities=list(seen.values()),
            final_live=live,
            final_usage=usage(out),
        )
        if live:
            value["status"] = "halted"
        save()
    print(
        json.dumps(
            {k: value.get(k) for k in ("status", "error", "wall_s", "final_live")}
        )
    )
    if value["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
