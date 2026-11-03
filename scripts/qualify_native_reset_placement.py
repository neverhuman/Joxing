"""Bounded H34source/reset/audit/two full fixed recordings; no new training."""

import argparse
import importlib.metadata
import json
import os
import platform
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
    for name in ("packet", "source-sha", "protocol-sha256"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--epoch-start-monotonic", type=float, required=True)
    a = ap.parse_args()
    p = json.loads((ROOT / a.packet).read_text())
    epoch = a.epoch_start_monotonic
    if (
        platform.system() != "Darwin"
        or p["schema"] != "joxing-native-reset-placement-study-v1"
        or base.digest(p) != a.protocol_sha256
        or not 0 <= time.monotonic() - epoch < 60
    ):
        raise ValueError("Frozen original source/protocol/profile/epoch")

    def source_check():
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.source_sha
            or subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD"], text=True
            ).strip()
        ):
            raise ValueError("Source HEAD drift")
        for n, r in {
            **p["source_files"],
            **p["input_files"],
            **p["runtime_file_hashes"],
        }.items():
            path = Path(n) if n.startswith("/") else ROOT / n
            if path.stat().st_size != r["bytes"] or base.sha(path) != r["sha256"]:
                raise ValueError("Source/runtime/input bytes drift:" + n)
        for n, sha in p["upstream_commits"].items():
            if (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT / "third_party" / n,
                    text=True,
                ).strip()
                != sha
            ):
                raise ValueError("Upstream pin drift")

    source_check()
    versions = {
        "python": sys.version.split()[0],
        **{
            n: importlib.metadata.version(n)
            for n in p["runtime_versions"]
            if n != "python"
        },
    }
    if versions != p["runtime_versions"]:
        raise ValueError("Explicit runtime version drift")
    out = ROOT / p["output"]
    out.mkdir(parents=True, exist_ok=False)
    me = base.processes()[os.getpid()]
    active = None
    seen = {}
    ids = [
        "source",
        "physical",
        "audit",
        *[c["id"] + "-render" for c in p["conditions"]],
    ]
    value = {
        "schema": p["schema"],
        "status": "running",
        "source_commit": a.source_sha,
        "protocol_sha256": a.protocol_sha256,
        "owner": me,
        "original_epoch": epoch,
        "rows": [{"id": n, "status": "not_run"} for n in ids],
        "stages": [],
    }
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
        if raw + len(blob) > p["raw_bytes"] or text + len(blob) > p["receipt_bytes"]:
            raise TimeoutError("Owner publication typed first limit")
        temp = out / "owner.json.tmp"
        temp.write_bytes(blob)
        temp.replace(out / "owner.json")

    def stage(name, command, wall, filemax=64 * 1024**2, raw_reserve=0):
        nonlocal active
        raw, text = usage(out)
        if (
            raw + raw_reserve + 65536 > p["raw_bytes"]
            or text + 65536 > p["receipt_bytes"]
        ):
            raise TimeoutError("Stage reserved bytes first limit")
        row = next(r for r in value["rows"] if r["id"] == name)
        row["status"] = "unconfirmed"
        started = time.monotonic()
        cmd = [
            sys.executable,
            "scripts/native_pair_guarded_stage.py",
            "--root",
            str(out),
            "--raw",
            str(p["raw_bytes"]),
            "--text",
            str(p["receipt_bytes"]),
            "--file-max",
            str(filemax),
            "--",
            *command,
        ]
        record = {"name": name, "command": cmd, "started_monotonic_s": started}
        value["stages"].append(record)
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
        log = (out / (name + ".log")).open("xb")
        try:
            while True:
                for key, _ in selector.select(0.1):
                    blob = os.read(key.fileobj.fileno(), 16384)
                    if blob:
                        raw, text = usage(out)
                        if (
                            raw + len(blob) + 65536 > p["raw_bytes"]
                            or text + len(blob) + 65536 > p["receipt_bytes"]
                        ):
                            raise TimeoutError("Captured log typed first limit")
                        log.write(blob)
                        log.flush()
                    else:
                        selector.unregister(key.fileobj)
                rows = base.processes()
                owned = base.descendants(rows, active.pid)
                for r in owned:
                    seen[(r["pid"], r["start"])] = r
                charge = rows[me["pid"]]["rss_bytes"] + sum(
                    r["rss_bytes"] for r in owned
                )
                raw, text = usage(out)
                record.update(
                    elapsed_s=time.monotonic() - started,
                    peak_sampled_rss_bytes=max(
                        record.get("peak_sampled_rss_bytes", 0), charge
                    ),
                    peak_processes=max(record.get("peak_processes", 0), 1 + len(owned)),
                    raw_bytes=raw,
                    text_bytes=text,
                )
                save()
                if (
                    charge > p["owned_rss_bytes"]
                    or len(owned) + 1 > p["process_ceiling"]
                    or raw > p["raw_bytes"]
                    or text > p["receipt_bytes"]
                ):
                    raise RuntimeError("Owned RSS/process/raw/text first limit")
                if (
                    record["elapsed_s"] > wall - 3
                    or time.monotonic() - epoch > p["whole_wall_s"] - 5
                ):
                    raise TimeoutError("Original stage/whole first limit")
                if active.poll() is not None and not selector.get_map():
                    break
            rc = active.wait()
            active = None
            record.update(returncode=rc, wall_s=time.monotonic() - started)
        finally:
            selector.close()
            log.close()
        if rc:
            raise RuntimeError("Registered stage failed:" + name)
        row["status"] = "completed"
        save()

    try:
        # Refuse an overlapping owned native writer; never stop unrelated work.
        competing = [
            r
            for pid, r in base.processes().items()
            if pid != me["pid"]
            and "/joxing/" in r["command"]
            and any(
                n in r["command"]
                for n in (
                    "scripts/qualify_native_",
                    "scripts/run_native_",
                    "scripts/audit_native_",
                    "scripts/render_native_",
                )
            )
            and not r["state"].startswith("Z")
        ]
        if competing:
            value["competing_owned_jobs"] = competing
            raise RuntimeError("Owned native writer already active")
        value["admission"] = [base.admission(p)]
        save()
        time.sleep(10)
        value["admission"].append(base.admission(p))
        save()
        if time.monotonic() - epoch > 60:
            raise TimeoutError("Setup original60s")
        stage(
            "source",
            [
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
            raw_reserve=64 * 1024**2,
        )
        from source_shard_support import check_junit

        value["source_gate"] = check_junit(
            (out / "source.xml").read_text(), p["test_node_ids"], []
        )
        if not value["source_gate"]["qualified"]:
            raise ValueError("Complete registered source prerequisite")
        # Pytest's owned convenience aliases are not raw data. Record before
        # removing them; the strict native evidence root refuses symlinks.
        aliases = []
        for path in (out / "source-fixtures").rglob("*"):
            if path.is_symlink():
                aliases.append(
                    {"path": str(path.relative_to(out)), "target": os.readlink(path)}
                )
                path.unlink()
        value["source_temporary_aliases_removed"] = aliases
        save()
        stage(
            "physical",
            [
                "scripts/run_native_reset_placement.py",
                "--packet",
                a.packet,
                "--stage",
                "physical",
            ],
            p["native_wall_s"],
        )
        stage(
            "audit",
            [
                "scripts/run_native_reset_placement.py",
                "--packet",
                a.packet,
                "--stage",
                "audit",
            ],
            p["audit_wall_s"],
        )
        for c in p["conditions"]:
            stage(
                c["id"] + "-render",
                [
                    "scripts/render_native_reset_placement.py",
                    "--folder",
                    str(out / "physical" / c["id"] / "seed-71000"),
                    "--report",
                    str(out / "recordings" / c["id"]),
                    "--label",
                    c["id"],
                    "--source-sha",
                    a.source_sha,
                ],
                p["render_wall_s"],
                filemax=4 * 1024**2,
                raw_reserve=16 * 1024**2,
            )
        source_check()
        value["status"] = "completed"
    except BaseException as error:  # noqa: BLE001 -- retain prefixes and owned cleanup
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
            original_epoch_elapsed_s=time.monotonic() - epoch,
            owned_observed_identities=list(seen.values()),
            final_live=live,
            final_usage=usage(out),
        )
        if live or value["original_epoch_elapsed_s"] > p["whole_wall_s"]:
            value["status"] = "halted"
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
                    "final_usage",
                )
            }
        )
    )
    if value["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
