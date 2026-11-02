"""H24: bounded native Mac source checks, explicitly forbidding physics steps."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "experiments/v0.0.11/mac-source-readiness-protocol.json"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def pytest_sessionstart(session):
    import mujoco

    session._h24_calls = []

    def forbidden(*args, **kwargs):
        session._h24_calls.append("forbidden_physics_step")
        raise AssertionError("H24 permits compile/forward only; no physics steps")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, forbidden)


def pytest_sessionfinish(session, exitstatus):
    import mujoco
    import numpy
    import pytest

    write(Path(os.environ["JOXING_MAC_SOURCE_PROOF"]), {
        "physics_calls": session._h24_calls,
        "exitstatus": int(exitstatus),
        "collected": session.testscollected,
        "versions": {"python": platform.python_version(), "numpy": numpy.__version__,
                     "mujoco": mujoco.__version__, "pytest": pytest.__version__},
        "platform": platform.system(), "machine": platform.machine(),
        "scope": "source/compile/forward only; no actor, training, native dynamics or Linux teardown parity",
    })


def tree(pid):
    raw = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows = [tuple(map(int, line.split())) for line in raw.splitlines()]
    owned = {pid}
    while True:
        more = {p for p, parent, _ in rows if parent in owned}
        if more <= owned:
            break
        owned |= more
    return {"pids": sorted(owned), "rss_bytes": sum(r * 1024 for p, _, r in rows if p in owned)}


def run_stage(name, command, out, cap_s, env, packet, whole_started):
    start = time.monotonic()
    peak = 0
    fault = None
    with (out / (name + ".log")).open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                sample = tree(child.pid)
                peak = max(peak, sample["rss_bytes"])
                elapsed = time.monotonic() - start
                if elapsed > cap_s or time.monotonic() - whole_started > packet["whole_s"]:
                    fault = "wall_limit"
                elif peak > packet["rss_bytes"] or len(sample["pids"]) > packet["processes"]:
                    fault = "owned_resource_limit"
                elif shutil.disk_usage(ROOT).free < packet["minimum_disk_bytes"]:
                    fault = "disk_floor"
                elif sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > packet["raw_bytes"]:
                    fault = "output_limit"
                write(out / "progress.json", {"stage": name, "pid": child.pid,
                      "elapsed_s": elapsed, "owned": sample, "fault": fault})
                if fault:
                    break
                time.sleep(0.2)
        finally:
            # All descendants belong to this newly created group, no unrelated work.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=2)
    return {"stage": name, "command": command, "pid": child.pid,
            "returncode": child.returncode, "fault": fault,
            "wall_s": time.monotonic() - start, "sampled_peak_rss_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    packet = json.loads(PACKET.read_text())
    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == args.source_sha
    assert not subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).strip()
    for name, expected in packet["source_files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/ms-human-700", text=True).strip() == packet["model_gitlink"]
    out = ROOT / packet["output"]
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT),
           "JOXING_MAC_SOURCE_PROOF": str(out / "proof.json"), "UV_NO_PROGRESS": "1"}
    started = time.monotonic()
    report = {"source_commit": args.source_sha,
              "protocol_sha256": hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "status": "incomplete", "stages": [], "attempted_test_cases": 0,
              "expected_test_cases": packet["test_cases"], "physics_steps": 0,
              "model_gitlink": packet["model_gitlink"], "source_files": packet["source_files"]}
    write(out / "result.json", report)
    stages = [
        ("sync", ["uv", "sync", "--frozen", "--extra", "dev", "--python", packet["python"]], packet["sync_s"]),
        ("checks", [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "scripts.qualify_mac_source", *packet["tests"], "--junitxml=" + str(out / "checks.xml")], packet["tests_s"]),
    ]
    for name, command, cap in stages:
        row = run_stage(name, command, out, cap, env, packet, started)
        report["stages"].append(row)
        if (out / "checks.xml").exists():
            report["attempted_test_cases"] = len(list(ET.parse(out / "checks.xml").getroot().iter("testcase")))
        write(out / "result.json", report)
        if row["fault"] or row["returncode"]:
            report["status"] = "failed_prerequisite" if name == "sync" else "failed_checks"
            break
    else:
        proof = json.loads((out / "proof.json").read_text())
        cases = list(ET.parse(out / "checks.xml").getroot().iter("testcase"))
        failed = [c.attrib["name"] for c in cases if list(c)]
        report.update(proof=proof, attempted_test_cases=len(cases), failed_or_skipped_cases=failed)
        report["status"] = "audit_pending"
        write(out / "result.json", report)
        assert len(cases) == packet["test_cases"] == proof["collected"] and not failed
        assert not proof["physics_calls"] and proof["versions"] == packet["versions"]
        report["status"] = "qualified_source_only"
    report["wall_s"] = time.monotonic() - started
    write(out / "result.json", report)
    print(json.dumps({k: report[k] for k in ("status", "source_commit", "wall_s", "attempted_test_cases", "physics_steps")}))


if __name__ == "__main__":
    main()
