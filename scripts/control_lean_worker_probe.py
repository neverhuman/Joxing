"""Detached H28 owner: bounded source, then fresh audit in original host epoch."""

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from supervise_motor_pilot import descendants, identity, live
from supervise_upper_task_tests import rss, save, stop
from upper_task_gate_support import partition_usage

ROOT = Path(__file__).resolve().parents[1]


def allowance(epoch, stage_start, stage_cap, *, clock=time.monotonic):
    now = clock()
    if (
        not all(math.isfinite(x) for x in (epoch, stage_start, now))
        or epoch > stage_start
        or stage_start > now
    ):
        raise ValueError("Original monotonic epoch required")
    remaining = min(stage_cap - (now - stage_start), 300 - (now - epoch))
    if remaining <= 0:
        raise TimeoutError("Original whole/stage audit limit")
    return remaining


def source_command(packet, source, protocol, epoch):
    return [
        sys.executable,
        "scripts/qualify_lean_worker_probe.py",
        "--packet",
        packet,
        "--source-sha",
        source,
        "--protocol-sha256",
        protocol,
        "--epoch-start-monotonic",
        str(epoch),
    ]


def main():
    parser = argparse.ArgumentParser()
    for key in ("packet", "source-sha", "protocol-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--epoch-start-monotonic", type=float, required=True)
    args = parser.parse_args()
    p = json.loads((ROOT / args.packet).read_text())
    epoch = args.epoch_start_monotonic
    allowance(epoch, time.monotonic(), 300)
    run = ROOT / "runs/v0.0.11"
    out = ROOT / p["output"]
    controller = identity(os.getpid())
    control_path = run / "lean-worker-probe-controller.json"
    if control_path.exists():
        raise FileExistsError("One fresh controller only")
    status = {
        "status": "source_running",
        "controller": controller,
        "original_epoch": epoch,
        "source_command": source_command(
            args.packet, args.source_sha, args.protocol_sha256, epoch
        ),
        "owned_identities": [],
    }
    save(control_path, status)
    env = os.environ.copy()
    env.update(
        {
            k: "1"
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
    )
    env.update(
        PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT / "scripts"),
        PYTHONDONTWRITEBYTECODE="1",
        MUJOCO_GL="egl",
    )
    source = None
    audit = None
    seen = {}
    peak_control = rss(controller)
    peak_processes = 1
    try:
        with (run / "lean-worker-probe-launch.log").open("x") as log:
            source = subprocess.Popen(
                status["source_command"],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            source_id = identity(source.pid)
            seen[(source_id["pid"], source_id["start_ticks"])] = source_id
            status["source_process"] = source_id
            save(control_path, status)
            while source.poll() is None:
                current = descendants(source.pid)
                for item in current:
                    seen[(item["pid"], item["start_ticks"])] = item
                peak_processes = max(peak_processes, 2 + sum(live(i) for i in current))
                if peak_processes > p["process_ceiling"]:
                    raise RuntimeError("Controller-inclusive process first limit")
                peak_control = max(peak_control, rss(controller))
                if time.monotonic() - epoch > 271:
                    status["source_controller_stop"] = (
                        "Source did not finish by original271s cleanup boundary"
                    )
                    stop(source, seen)
                    break
                time.sleep(0.1)
            source.wait(timeout=2)
        status.update(
            status="source_exited",
            source_returncode=source.returncode,
            owned_identities=list(seen.values()),
        )
        save(control_path, status)
        main_path = out / "result.json.gz"
        if not main_path.exists():
            raise RuntimeError("Source produced no final main receipt")
        import gzip

        result = json.loads(gzip.decompress(main_path.read_bytes()))
        status["source_status"] = result["status"]
        if result["status"] not in ("completed", "halted", "deferred"):
            raise RuntimeError("Stale source running receipt; no false completed audit")
        if any(live(i) for i in seen.values()):
            raise RuntimeError("Owned source writer remains; refuse overlapping audit")
        started = time.monotonic()
        allowance(epoch, started, 30)
        cmd = [
            str(ROOT / "experiments/v0.0.11/runtime/.venv/bin/python"),
            "scripts/audit_lean_worker_probe.py",
            "--packet",
            args.packet,
            "--folder",
            str(out),
            "--source-sha",
            args.source_sha,
            "--ready-file",
            str(out / "independent-audit.ready"),
        ]
        status.update(status="audit_running", audit_command=cmd)
        save(control_path, status)
        audit_peak = 0
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
            audit_id = identity(audit.pid)
            status["audit_process"] = audit_id
            # Last controller write before audit snapshots evidence. No overlapping
            # progress writer while independent reader inventories source bytes.
            save(control_path, status)
            (out / "independent-audit.ready").touch(exist_ok=False)
            while audit.poll() is None:
                allowance(epoch, started, 30)
                if live(audit_id):
                    audit_peak = max(audit_peak, rss(audit_id))
                peak_control = max(peak_control, rss(controller))
                if audit_peak + rss(controller) > p["owned_rss_bytes"]:
                    raise RuntimeError("Owned audit/controller RSS first limit")
                time.sleep(0.02)
            audit.wait(timeout=1)
        status.update(
            status="audited" if audit.returncode == 0 else "audit_failed",
            audit_returncode=audit.returncode,
            audit_peak_rss_bytes=audit_peak,
        )
        if audit.returncode:
            status["audit_fault_log"] = str(out / "independent-audit.log")
    except BaseException as error:  # noqa: BLE001 -- retain controller and original source faults
        status.update(
            status="controller_failed", error=type(error).__name__ + ": " + str(error)
        )
        if source is not None and source.poll() is None:
            stop(source, seen)
        if audit is not None and audit.poll() is None:
            stop(audit, {})
    finally:
        status.update(
            original_epoch_elapsed_s=time.monotonic() - epoch,
            controller_peak_rss_bytes=max(peak_control, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            source_controller_peak_processes=peak_processes,
            owned_identities=list(seen.values()),
            final_live=[i for i in seen.values() if live(i)],
        )
        if out.exists():
            scratch, receipts = partition_usage(out.parent, out / "fixtures")
            scratch += p["mirrored_input_bytes"]
            status["final_usage"] = {
                "fixtures": scratch,
                "receipts_before_controller_final": receipts,
            }
            if scratch > p["fixture_bytes"] or receipts + 8192 > p["receipt_bytes"]:
                status.update(
                    status="controller_failed",
                    storage_error="Final storage first limit including controller publication reserve",
                )
        if time.monotonic() - epoch > 300:
            status.update(status="controller_failed", final_wall_error="Original epoch exceeded through controller publication")
        save(control_path, status)
    print(
        json.dumps(
            {
                k: status.get(k)
                for k in (
                    "status",
                    "source_status",
                    "audit_returncode",
                    "error",
                    "original_epoch_elapsed_s",
                    "final_live",
                )
            }
        )
    )
    if status["status"] != "audited":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
