"""Bounded zero-integration native calibration and independent saved read-audit."""

import argparse
import hashlib
import json
import resource
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np

from .motor_storage import RawCredit, append_json, write_json

ROOT = Path(__file__).resolve().parents[2]
PACKET = "experiments/v0.0.11/native-face-audit-protocol.json"
RUN = "runs/v0.0.11/native-face-audit-01"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )


def packet(protocol_sha, source_sha):
    p = json.loads((ROOT / PACKET).read_text())
    if (
        p["schema"] != "joxing-native-face-offline-protocol-v1"
        or p["new_physics"] != 0
        or p["new_optimizer_updates"] != 0
        or digest(p) != protocol_sha
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != source_sha
        or json.loads(
            subprocess.check_output(["git", "show", "HEAD:" + PACKET], cwd=ROOT)
        )
        != p
    ):
        raise ValueError("Native face frozen source/protocol identity")
    for name, expected in p["upstream_commits"].items():
        row = subprocess.check_output(
            ["git", "ls-tree", "HEAD", "third_party/" + name], cwd=ROOT, text=True
        ).split()
        if len(row) < 3 or row[2] != expected:
            raise ValueError("Native face upstream pin: " + name)
    for n, d in p["source_files"].items():
        f = ROOT / n
        if (
            f.stat().st_size != d["bytes"]
            or hashlib.sha256(f.read_bytes()).hexdigest() != d["sha256"]
        ):
            raise ValueError("Native face source/asset identity: " + n)
    if (
        sys.version.split()[0] != p["python_version"]
        or np.__version__ != p["numpy_version"]
        or mujoco.__version__ != p["mujoco_version"]
    ):
        raise ValueError("Native face pinned runtime identity")
    return p


@contextmanager
def no_integration():
    originals = {
        name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2")
    }

    def forbidden(*args, **kwargs):
        raise RuntimeError(
            "Native face diagnostic attempted forbidden physics integration"
        )

    try:
        for name in originals:
            setattr(mujoco, name, forbidden)
        yield
    finally:
        for name, function in originals.items():
            setattr(mujoco, name, function)


def prove(p, source_sha, protocol_sha):
    from .native_face_audit import audit
    from .scene import build_scene

    out = ROOT / RUN
    out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(out, p["raw_bytes"] - p["supervision_reserve_bytes"])
    began = time.monotonic()
    result = {
        "schema": "joxing-native-face-qualification-v1",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "status": "running",
        "new_physics": 0,
        "rows": [{"id": key, "status": "not_run"} for key in p["case_ids"]],
        "native_boxing_proof_passed": False,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
        },
    }
    lookup = {r["id"]: r for r in result["rows"]}
    entered, completed, current = [], [], None
    write_json(out / "result.json", result)

    def event(phase, key):
        nonlocal current
        if (
            time.monotonic() - began > p["stage_wall_limits_s"]["proof"]
            or rss_bytes() > p["maximum_rss_bytes"]
        ):
            raise TimeoutError("Native face proof wall/RSS cap")
        if phase == "entered":
            if key != p["case_ids"][len(entered)] or current is not None:
                raise ValueError("Native face entered-case order")
            entered.append(key)
            current = key
            lookup[key]["status"] = "running"
        elif phase == "completed":
            if key != current:
                raise ValueError("Native face completed-case ownership")
            completed.append(key)
            lookup[key]["status"] = "passed"
            current = None
        else:
            raise ValueError("Unknown native face case phase")
        append_json(out / "events.jsonl", {"phase": phase, "id": key})
        if len(completed) % 16 == 0:
            write_json(out / "result.json", result)

    try:
        with no_integration():
            scene = build_scene()
            write_json(out / "scene.json", scene.report)
            report = audit(
                scene,
                epsilon=p["epsilon_s"],
                velocity_tolerance=p["velocity_tolerance_mps"],
                geometry_tolerance=p["geometry_tolerance"],
                on_row=lambda row: append_json(out / "rows.jsonl", row),
                on_event=event,
            )
        if entered != p["case_ids"] or completed != p["case_ids"]:
            raise ValueError("Native face exact case denominator")
        write_json(out / "source.json", report)
        result.update(
            status="passed",
            row_count=report["row_count"],
            source_bytes=(out / "source.json").stat().st_size,
            source_sha256=hashlib.sha256(
                (out / "source.json").read_bytes()
            ).hexdigest(),
            maximum_velocity_error_mps=report["maximum_velocity_error_mps"],
            maximum_point_rotation_witness_mps=report[
                "maximum_point_rotation_witness_mps"
            ],
            model_binary_sha256=report["model_binary_sha256"],
        )
    except Exception as error:  # noqa: BLE001 -- retain every attempted/not_run case
        import traceback

        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        if current is not None:
            lookup[current].update(status="failed", reason=result["error"])
        with credit.emergency():
            write_json(
                out / "fault.json",
                {
                    "error": result["error"],
                    "traceback": traceback.format_exc(),
                    "entered": entered,
                    "completed": completed,
                },
            )
    finally:
        result.update(
            cases_entered=len(entered),
            cases_completed=len(completed),
            wall_s=time.monotonic() - began,
            max_rss_bytes=rss_bytes(),
        )
        with credit.emergency():
            write_json(out / "result.json", result)
    return result


def audit(p, source_sha, protocol_sha):
    from .native_face_evidence import audit_saved
    from .scene import build_scene

    began = time.monotonic()
    out = ROOT / RUN
    result = json.loads((out / "result.json").read_text())
    source = out / "source.json"
    if (
        result["status"] != "passed"
        or result["source_commit"] != source_sha
        or result["protocol_sha256"] != protocol_sha
        or source.stat().st_size != result["source_bytes"]
        or hashlib.sha256(source.read_bytes()).hexdigest() != result["source_sha256"]
    ):
        raise ValueError("Native face saved source identity")
    from .evidence_storage import read_json

    report = read_json(source)
    expected_events = [
        {"phase": phase, "id": key}
        for key in p["case_ids"]
        for phase in ("entered", "completed")
    ]
    events = [json.loads(s) for s in (out / "events.jsonl").read_text().splitlines()]
    journal = [json.loads(s) for s in (out / "rows.jsonl").read_text().splitlines()]
    if (
        events != expected_events
        or journal != report["rows"]
        or [r["id"] for r in journal] != p["case_ids"]
        or any(r["status"] != "passed" for r in result["rows"])
    ):
        raise ValueError("Native face persisted attempted/completed case ledger")
    with no_integration():
        audited = audit_saved(
            build_scene(),
            report,
            epsilon=p["epsilon_s"],
            velocity_tolerance=p["velocity_tolerance_mps"],
            geometry_tolerance=p["geometry_tolerance"],
        )
    if (
        time.monotonic() - began > p["stage_wall_limits_s"]["audit"]
        or rss_bytes() > p["maximum_rss_bytes"]
    ):
        raise TimeoutError("Native face independent audit wall/RSS cap")
    audited.update(
        source_commit=source_sha,
        protocol_sha256=protocol_sha,
        all_planned_rows_audited=True,
        wall_s=time.monotonic() - began,
        max_rss_bytes=rss_bytes(),
    )
    RawCredit(out, p["raw_bytes"] - p["supervision_reserve_bytes"])
    write_json(out / "independent-audit.json", audited)
    return audited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("proof", "audit"), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise TimeoutError(
            "Owned native face qualification interrupted: " + str(signum)
        )

    signal.signal(signal.SIGTERM, interrupted)
    p = packet(args.protocol_sha256, args.source_sha)
    result = (prove if args.stage == "proof" else audit)(
        p, args.source_sha, args.protocol_sha256
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
