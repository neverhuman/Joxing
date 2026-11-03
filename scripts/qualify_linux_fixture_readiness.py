"""H29 source-only detached owner/source phases. No probe or learner dispatch."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from host_admission import AdmissionDeferred, reasons, snapshot
from joxing_process_ownership import unknown_joxing_processes
from portable_receipt_bundle import hydrate
from receipt_encoding import (
    V2,
    main_metadata,
    pack_owned_control,
    save_report,
    terminal_line,
)
from source_shard_support import check_junit
from supervise_motor_pilot import descendants, identity, live
from supervise_upper_task_tests import rss, save, stop
from upper_task_gate_support import partition_usage

from joxing.receipt_fixture_paths import ENV_HASH, ENV_ROOT

SCHEMA = "joxing-portable-linux-source-readiness-v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate(p):
    expected = {
        "schema": SCHEMA,
        "whole_wall_s": 120,
        "setup_wall_s": 60,
        "control_wall_s": 30,
        "audit_wall_s": 15,
        "physics_limit": 0,
        "simulation_workers": 1,
        "threads_per_process": 1,
        "process_ceiling": 12,
        "owned_rss_bytes": 8 * 1024**3,
        "fixture_bytes": 8 * 1024**2,
        "receipt_bytes": 1024**2,
        "receipt_wire_encoding": V2,
    }
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("Explicit bounded source-only H29 protocol required")


def verify(root, p, source, protocol):
    validate(p)
    if (
        digest(p) != protocol
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != source
    ):
        raise ValueError("Source/protocol drift")
    if subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=root, text=True
    ).strip():
        raise ValueError("Modified tracked source")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "--is-shallow-repository"], cwd=root, text=True
        ).strip()
        != "false"
    ):
        raise ValueError("Full Git history required")
    for name, meta in {
        **p["source_files"],
        **p["runtime_file_hashes"],
        **p["input_files"],
    }.items():
        path = Path(name) if name.startswith("/") else root / name
        data = path.read_bytes()
        if (
            len(data) != meta["bytes"]
            or hashlib.sha256(data).hexdigest() != meta["sha256"]
        ):
            raise ValueError("Input/source/runtime drift:" + name)
    for obj, sha in p["required_git_objects"].items():
        if (
            hashlib.sha256(
                subprocess.check_output(["git", "show", obj], cwd=root)
            ).hexdigest()
            != sha
        ):
            raise ValueError("Historical object drift")
    for name, sha in p["upstream_commits"].items():
        if (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root / "third_party" / name, text=True
            ).strip()
            != sha
        ):
            raise ValueError("Upstream pin drift")
    if os.uname().release != p["execution_kernel_release"]:
        raise ValueError("Execution kernel drift")


def save_main(path, value):
    save_report(path, value, mode=V2, legacy_save=save)


def source_phase(args, p):
    epoch = args.epoch_start_monotonic
    if not math.isfinite(epoch) or not 0 <= time.monotonic() - epoch < 60:
        raise ValueError("Original setup epoch required")
    verify(ROOT, p, args.source_sha, args.protocol_sha256)
    run = ROOT / "runs/v0.0.11"
    out = ROOT / p["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (run / ".portable-source.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    fixtures = out / "fixtures"
    fixtures.mkdir()
    supervisor = identity(os.getpid())
    controller = identity(args.controller_pid)
    if (
        controller["start_ticks"] != args.controller_start
        or os.getppid() != controller["pid"]
    ):
        raise ValueError("Bound controller ancestry required")
    ignored = {supervisor["pid"]}
    pid = supervisor["pid"]
    while pid > 1 and pid not in ignored - {supervisor["pid"]}:
        ignored.add(pid)
        pid = identity(pid)["parent_pid"]
    r = {
        "schema": SCHEMA,
        "source_commit": args.source_sha,
        "protocol_sha256": args.protocol_sha256,
        "started_monotonic_s": epoch,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "supervisor": supervisor,
        "controller": controller,
        "status": "running",
        "rows": [{"id": "linux-source", "status": "not_run"}],
        "stages": [],
        "native_physics": 0,
        "native_models": 0,
        "policy_actors": 0,
        "optimizer_updates": 0,
        "formal_probes": 0,
        "owned_identities": [],
    }
    main = out / "result.json.gz"
    seen = {}
    active = None
    env = os.environ | {
        k: "1"
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    env.update(
        PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT / "scripts"),
        PYTHONDONTWRITEBYTECODE="1",
        MUJOCO_GL="egl",
    )

    def capacity():
        if unknown_joxing_processes(supervisor, ignored_pids=ignored):
            raise RuntimeError("Other owned Joxing writer")
        raw, receipts = partition_usage(out.parent, fixtures)
        if raw > p["fixture_bytes"] or receipts + 8192 > p["receipt_bytes"]:
            raise RuntimeError("Source output first limit")
        sample = snapshot(ignored_pids=ignored | {i["pid"] for i in seen.values()})
        if (
            sample["available_memory_bytes"]
            < p["admission_policy"]["minimum_memory_bytes"]
            or sample["disk_free_bytes"] < p["admission_policy"]["minimum_disk_bytes"]
        ):
            raise RuntimeError("Shared host capacity floor")
        return {
            "fixture_bytes": raw,
            "receipt_bytes": receipts,
            "available_memory_bytes": sample["available_memory_bytes"],
            "free_disk_bytes": sample["disk_free_bytes"],
        }

    def stage(name, cmd, limit):
        nonlocal active
        started = time.monotonic()
        row = {"stage": name, "command": cmd, "status": "running"}
        r["stages"].append(row)
        with (out / (name + ".log")).open("xb") as log:
            active = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            bound = identity(active.pid)
            seen[(bound["pid"], bound["start_ticks"])] = bound
            row["process"] = bound
            while True:
                for i in descendants(active.pid):
                    seen[(i["pid"], i["start_ticks"])] = i
                living = [i for i in seen.values() if live(i)]
                charge = rss(supervisor) + rss(controller) + sum(rss(i) for i in living)
                row.update(
                    elapsed_s=time.monotonic() - started,
                    peak_sampled_rss_bytes=max(
                        row.get("peak_sampled_rss_bytes", 0), charge
                    ),
                    peak_processes=max(row.get("peak_processes", 0), 2 + len(living)),
                    capacity=capacity(),
                )
                save_main(main, r)
                if (
                    charge > p["planned_stop_rss_bytes"]
                    or 2 + len(living) > p["process_ceiling"]
                ):
                    raise RuntimeError(
                        "Controller-inclusive source resource first limit"
                    )
                if row["elapsed_s"] > limit - 2 or time.monotonic() - epoch > 101:
                    raise TimeoutError("Source original stage/whole limit")
                if active.poll() is not None:
                    break
                time.sleep(0.1)
            row.update(
                returncode=active.wait(),
                status="completed" if active.returncode == 0 else "failed",
                wall_s=time.monotonic() - started,
            )
        active = None
        if any(live(i) for i in seen.values()):
            raise RuntimeError("Source stage descendant still live")
        return row["returncode"]

    try:
        projection = json.loads((ROOT / p["receipt_projection_file"]).read_text())
        if (
            projection["scope"]
            != "prospective stored-evidence sizing; zero native work"
            or projection["projected_peak_receipt_bytes"] > p["receipt_bytes"]
        ):
            raise ValueError("Prospective whole-study receipt sizing first limit")
        r["receipt_projection"] = projection
        r["fixture_bundle"] = hydrate(ROOT, p["fixture_bundle"], fixtures / "receipts")
        env[ENV_ROOT] = str(fixtures / "receipts")
        env[ENV_HASH] = p["fixture_bundle"]["manifest_sha256"]
        sample = snapshot(ignored_pids=ignored)
        now = time.monotonic()
        bad = reasons(sample, p["admission_policy"], now)
        r["runtime_admission"] = {
            "sample": sample,
            "assessed_monotonic_s": now,
            "reasons": bad,
        }
        if bad:
            raise AdmissionDeferred(str(bad))
        runtime_code = "from pathlib import Path;import runpy,joxing.lean_worker_probe as m;assert Path(m.__file__).resolve().is_relative_to(Path('src').resolve());runpy.run_path('scripts/runtime_import_check.py',run_name='__main__')"
        rc = stage(
            "runtime",
            [
                p["runtime_python"],
                "-c",
                runtime_code,
                "--expected",
                json.dumps(p["runtime_versions"]),
                "--event-wire",
                "gzip-jsonl-v1",
            ],
            15,
        )
        if rc:
            raise RuntimeError("Pinned runtime prerequisite failed")
        if time.monotonic() - epoch > p["setup_wall_s"]:
            raise TimeoutError("Original setup limit")
        r["rows"][0]["status"] = "unconfirmed"
        cmd = [
            p["runtime_python"],
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--junitxml",
            str(out / "controls.xml"),
            "--basetemp",
            str(fixtures / "controls"),
            *p["test_modules"],
        ]
        rc = stage("controls", cmd, p["control_wall_s"])
        raw = (out / "controls.xml").read_bytes()
        checked = check_junit(raw.decode(), p["test_node_ids"], [])
        r["source_gate"] = checked
        if rc or not checked["qualified"]:
            raise RuntimeError("Source checks incomplete/failed")
        r["controls_encoding"] = pack_owned_control(
            out / "controls.xml", hashlib.sha256(raw).hexdigest()
        )
        verify(ROOT, p, args.source_sha, args.protocol_sha256)
        r["rows"][0]["status"] = "completed"
        r["status"] = "completed"
    except BaseException as error:  # noqa: BLE001 -- retain every failed/partial prerequisite
        if active is not None:
            stop(active, seen)
        if r["rows"][0]["status"] == "unconfirmed":
            r["rows"][0]["status"] = "failed"
        r.update(
            status="deferred" if isinstance(error, AdmissionDeferred) else "halted",
            error=type(error).__name__ + ": " + str(error),
        )
    finally:
        r.update(
            wall_s=time.monotonic() - epoch,
            owned_identities=list(seen.values()),
            final_live_recorded=[i for i in seen.values() if live(i)],
        )
        try:
            r["final_usage"] = capacity()
        except Exception as error:  # noqa: BLE001 -- preserve an independently reviewable fault receipt
            r.update(
                status="halted",
                final_capacity_error=type(error).__name__ + ": " + str(error),
            )
        save_main(main, r)
    print(
        terminal_line(
            {
                k: r.get(k)
                for k in (
                    "status",
                    "error",
                    "rows",
                    "physics_credit",
                    "wall_s",
                    "final_live_recorded",
                )
            },
            V2,
            main=main_metadata(main),
        )
    )
    if r["status"] != "completed":
        raise SystemExit(1)


def controller_phase(args, p):
    run = ROOT / "runs/v0.0.11"
    run.mkdir(parents=True, exist_ok=True)
    out = ROOT / p["output"]
    epoch = args.epoch_start_monotonic
    owner = identity(os.getpid())
    path = run / "portable-source-controller.json"
    if path.exists():
        raise FileExistsError("Fresh source-only owner required")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source",
        "--packet",
        args.packet,
        "--source-sha",
        args.source_sha,
        "--protocol-sha256",
        args.protocol_sha256,
        "--epoch-start-monotonic",
        str(epoch),
        "--controller-pid",
        str(owner["pid"]),
        "--controller-start",
        str(owner["start_ticks"]),
    ]
    value = {
        "status": "source_running",
        "controller": owner,
        "source_command": cmd,
        "original_epoch": epoch,
    }
    save(path, value)
    child = audit = None
    seen = {}
    env = os.environ | {
        k: "1"
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    env.update(
        PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT / "scripts"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    try:
        with (run / "portable-source-launch.log").open("x") as log:
            child = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            b = identity(child.pid)
            seen[(b["pid"], b["start_ticks"])] = b
            value["source_process"] = b
            save(path, value)
            while child.poll() is None:
                for i in descendants(child.pid):
                    seen[(i["pid"], i["start_ticks"])] = i
                if time.monotonic() - epoch > 103:
                    raise TimeoutError("Source owner audit reserve")
                time.sleep(0.1)
            child.wait(timeout=2)
        value.update(
            source_returncode=child.returncode,
            status="source_exited",
            owned_identities=list(seen.values()),
        )
        save(path, value)
        if any(live(i) for i in seen.values()):
            raise RuntimeError("Source writer live before audit")
        cmd = [
            p["runtime_python"],
            "scripts/audit_linux_fixture_readiness.py",
            "--packet",
            args.packet,
            "--folder",
            str(out),
            "--source-sha",
            args.source_sha,
            "--ready-file",
            str(out / "independent-audit.ready"),
        ]
        value.update(status="audit_running", audit_command=cmd)
        save(path, value)
        started = time.monotonic()
        with (out / "independent-audit.log").open("x") as log:
            audit = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            b = identity(audit.pid)
            value["audit_process"] = b
            save(path, value)
            (out / "independent-audit.ready").touch(exist_ok=False)
            while audit.poll() is None:
                if time.monotonic() - started > 15 or time.monotonic() - epoch > 118:
                    raise TimeoutError("Original independent audit limit")
                if rss(owner) + rss(b) > p["owned_rss_bytes"]:
                    raise RuntimeError("Audit/controller RSS first limit")
                time.sleep(0.02)
            audit.wait(timeout=1)
        value.update(
            status="audited" if audit.returncode == 0 else "audit_failed",
            audit_returncode=audit.returncode,
        )
    except BaseException as error:  # noqa: BLE001 -- retain source and controller faults
        value.update(
            status="controller_failed", error=type(error).__name__ + ": " + str(error)
        )
        if child is not None and child.poll() is None:
            stop(child, seen)
        if audit is not None and audit.poll() is None:
            stop(audit, {})
    finally:
        value.update(
            original_epoch_elapsed_s=time.monotonic() - epoch,
            owned_identities=list(seen.values()),
            final_live=[i for i in seen.values() if live(i)],
            controller_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        )
        if value["original_epoch_elapsed_s"] > 120:
            value.update(
                status="controller_failed", wall_error="Final original120s limit"
            )
        if out.exists():
            raw, receipts = partition_usage(out.parent, out / "fixtures")
            value["final_usage"] = {
                "fixture_bytes": raw,
                "receipt_bytes_before_final": receipts,
            }
            if raw > p["fixture_bytes"] or receipts + 8192 > p["receipt_bytes"]:
                value.update(
                    status="controller_failed", storage_error="Final bytes first limit"
                )
        save(path, value)
    print(
        json.dumps(
            {
                k: value.get(k)
                for k in (
                    "status",
                    "audit_returncode",
                    "error",
                    "original_epoch_elapsed_s",
                    "final_live",
                )
            }
        )
    )
    if value["status"] != "audited":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k in ("packet", "source-sha", "protocol-sha256"):
        parser.add_argument("--" + k, required=True)
    parser.add_argument("--epoch-start-monotonic", type=float, required=True)
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--controller-pid", type=int)
    parser.add_argument("--controller-start", type=int)
    args = parser.parse_args()
    packet = json.loads((ROOT / args.packet).read_text())
    validate(packet)
    if args.source:
        source_phase(args, packet)
    else:
        controller_phase(args, packet)
