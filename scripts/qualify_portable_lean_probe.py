"""H30 detached finite owner and fake-only paired qualification; no learner route."""

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
from portable_lean_probe_support import (
    SCHEMA,
    check_guard,
    digest,
    guard_limits,
    headroom,
    positive_retirement,
    validate,
)
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
from watchdog_publication import ReportScope

from joxing.receipt_fixture_paths import ENV_HASH, ENV_ROOT


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
    lease = (run / ".portable-lean.lock").open("a+")
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
        "rows": [dict(slot, status="not_run") for slot in p["slots"]],
        "source_status": "not_run",
        "inner_packet_sha256": p["inner_packet_sha256"],
        "stages": [],
        "native_physics": 0,
        "native_models": 0,
        "policy_actors": 0,
        "optimizer_updates": 0,
        "formal_probes_attempted": 0,
        "owned_identities": [],
    }
    main = out / "result.json.gz"
    seen = {}
    active = guard = publication = None
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
        raw += sum(f.stat().st_size for f in out.glob("*-watchdog.*") if f.is_file())
        if publication is not None and not publication.finished:
            publication._validate()
            raw += p["ipc_bytes"]  # Reserved live IPC upper bound, not an RSS reading.
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

    def stage(name, cmd, limit, *, slot=None):
        nonlocal active, guard
        started = time.monotonic()
        row = {
            "stage": name,
            "command": cmd,
            "status": "running",
            "started_monotonic_s": started,
        }
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
            b = identity(active.pid)
            seen[(b["pid"], b["start_ticks"])] = b
            row["process"] = b
            guardlog = None
            try:
                if slot is not None:
                    limits = guard_limits(p)
                    gcmd = [
                        sys.executable,
                        "scripts/lean_probe_watchdog.py",
                        "--root-pid",
                        str(b["pid"]),
                        "--root-start",
                        str(b["start_ticks"]),
                        "--supervisor-pid",
                        str(supervisor["pid"]),
                        "--supervisor-start",
                        str(supervisor["start_ticks"]),
                        "--output",
                        str(publication.path),
                        "--planned-rss",
                        str(limits["planned_rss"]),
                        "--hard-rss",
                        str(limits["hard_rss"]),
                        "--process-cap",
                        str(limits["process_cap"]),
                        "--wall-s",
                        str(min(limit, 270 - (time.monotonic() - epoch)) - 2),
                        "--max-gap-s",
                        str(p["max_monitor_gap_s"]),
                        "--rss-reader",
                        p["rss_reader"],
                        "--phase-timing",
                    ]
                    guardlog = (out / (name + "-watchdog.log")).open("xb")
                    guard = subprocess.Popen(
                        gcmd,
                        cwd=ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=guardlog,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    g = identity(guard.pid)
                    seen[(g["pid"], g["start_ticks"])] = g
                    row.update(watchdog_process=g, watchdog_command=gcmd)
                while True:
                    for i in descendants(active.pid):
                        seen[(i["pid"], i["start_ticks"])] = i
                    living = [i for i in seen.values() if live(i)]
                    controller_rss = rss(controller)
                    charge = (
                        rss(supervisor) + controller_rss + sum(rss(i) for i in living)
                    )
                    row.update(
                        elapsed_s=time.monotonic() - started,
                        peak_sampled_rss_bytes=max(
                            row.get("peak_sampled_rss_bytes", 0), charge
                        ),
                        peak_processes=max(
                            row.get("peak_processes", 0), 2 + len(living)
                        ),
                        controller_peak_sampled_rss=max(
                            row.get("controller_peak_sampled_rss", 0), controller_rss
                        ),
                        capacity=capacity(),
                    )
                    save_main(main, r)
                    if (
                        controller_rss > p["controller_reserve_bytes"]
                        or charge > p["planned_stop_rss_bytes"]
                        or 2 + len(living) > p["process_ceiling"]
                    ):
                        raise RuntimeError("Controller-inclusive resource first limit")
                    if row["elapsed_s"] > limit - 2 or time.monotonic() - epoch > 268:
                        raise TimeoutError(
                            "Original stage/task limit with audit reserve"
                        )
                    if (
                        guard is not None
                        and guard.poll() is not None
                        and active.poll() is None
                    ):
                        raise RuntimeError("Independent guard exited before fake trial")
                    if active.poll() is not None:
                        break
                    time.sleep(0.1)
                rc = active.wait()
                row["returncode"] = rc
                if guard is not None:
                    guard.wait(timeout=2)
                    publication.finish(
                        task_live=any(
                            live(i) for i in seen.values() if i["pid"] != guard.pid
                        ),
                        guard_live=guard.poll() is None,
                    )
                    row["publication_receipt"] = str(
                        publication.receipt.relative_to(ROOT)
                    )
                    watch = json.loads((out / (name + "-watchdog.json")).read_text())
                    row["watchdog"] = watch
                    for i in watch["owned_identities"]:
                        seen[(i["pid"], i["start_ticks"])] = i
                    if guard.returncode:
                        raise RuntimeError("Guard fault: " + str(watch.get("reason")))
                    check_guard(watch, p, complete=True)
                    guard = None
                if any(live(i) for i in seen.values()):
                    raise RuntimeError("Stage descendant still live")
                row.update(
                    status="completed" if rc == 0 else "failed",
                    wall_s=time.monotonic() - started,
                )
                if row["wall_s"] > limit or time.monotonic() - epoch > 270:
                    raise TimeoutError("Publication/task wall first limit")
                active = None
                return rc
            finally:
                if guardlog is not None:
                    guardlog.close()

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
        r["source_status"] = "unconfirmed"
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
        r["source_status"] = "completed"
        r["receipt_preflight"] = headroom(p, capacity(), projection)
        for slot in r["rows"]:
            slot["status"] = "unconfirmed"
            r["formal_probes_attempted"] += 1
            name = slot["id"]
            publication = ReportScope(
                out / (name + "-watchdog.json"), cap=p["ipc_bytes"]
            )
            cmd = [
                p["runtime_python"],
                "scripts/run_portable_lean_probe.py",
                "--packet",
                args.packet,
                "--slot",
                name,
                "--folder",
                str(fixtures / name),
                "--ready",
                str(publication.path.with_suffix(".ready")),
                "--source-sha",
                args.source_sha,
                "--protocol-sha256",
                args.protocol_sha256,
            ]
            rc = stage(name, cmd, p["per_slot_wall_s"], slot=slot)
            if rc:
                raise RuntimeError("Fake-only row nonzero: " + name)
            facts = json.loads((fixtures / name / "probe-result.json").read_text())
            if any(
                facts[k] != 0 for k in ("physics", "actors", "updates", "inference")
            ):
                raise ValueError("Unexpected work in fake-only row")
            positive_retirement(r["stages"][-1]["watchdog"], facts["worker"])
            slot.update(status="completed", facts=facts)
        from joxing.lean_worker_probe import conclude

        r["comparison"] = conclude(
            {s["id"]: str(fixtures / s["id"]) for s in p["slots"]}
        )
        verify(ROOT, p, args.source_sha, args.protocol_sha256)
        r["status"] = "completed"
    except BaseException as error:  # noqa: BLE001 -- retain every failed/partial prerequisite
        if active is not None and active.poll() is None:
            stop(active, seen)
        if guard is not None and guard.poll() is None:
            stop(guard, seen)
        # Bound stragglers only, never another process group or unrelated PID.
        import signal

        for i in seen.values():
            if live(i):
                os.kill(i["pid"], signal.SIGKILL)
        if (
            publication is not None
            and not publication.finished
            and not publication.failed
        ):
            try:
                publication.finish(
                    task_live=any(live(i) for i in seen.values()),
                    guard_live=guard is not None and guard.poll() is None,
                )
            except Exception as fault:
                r["publication_cleanup_error"] = str(fault)  # noqa: BLE001 -- retain failed IPC, never delete unverified evidence
        if r["source_status"] == "unconfirmed":
            r["source_status"] = "failed"
        for slot in r["rows"]:
            if slot["status"] == "unconfirmed":
                slot["status"] = "failed"
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
    path = run / "portable-lean-controller.json"
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
        with (run / "portable-lean-launch.log").open("x") as log:
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
                if time.monotonic() - epoch > 270:
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
            "scripts/audit_portable_lean_probe.py",
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
                if time.monotonic() - started > 30 or time.monotonic() - epoch > 298:
                    raise TimeoutError("Original independent audit limit")
                if (
                    rss(owner) > p["controller_reserve_bytes"]
                    or rss(owner) + rss(b) > p["owned_rss_bytes"]
                ):
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
        if value["controller_peak_rss_bytes"] > p["controller_reserve_bytes"]:
            value.update(
                status="controller_failed",
                rss_error="Controller reserved upper bound exceeded",
            )
        if value["original_epoch_elapsed_s"] > 300:
            value.update(
                status="controller_failed", wall_error="Final original300s limit"
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
    raise SystemExit(
        "H30 DRAFT: independent audit/protocol not registered; formal dispatch disabled. Native motor RL is the priority; see docs/NATIVE_PAIR_RL_PILOT_MISSION.md"
    )
