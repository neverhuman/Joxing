"""Environment-only guardian for unchanged qualified scientific source2552354."""

import argparse
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def budget(p):
    if (
        p["prior_postgate_upper_s"]
        + p["preflight_wall_limit_s"]
        + p["native_postgate_limit_s"]
        > 900
    ):
        raise ValueError("Original postgate budget exhausted")
    if (
        p["prior_raw_upper_bytes"]
        + p["copied_prerequisite_bytes"]
        + p["operator_raw_limit_bytes"]
        + p["native_raw_limit_bytes"]
        > 256 * 1024**2
    ):
        raise ValueError("Original raw budget exhausted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    p = json.loads(args.manifest.read_text())
    root = args.root.resolve()
    if (
        digest(p) != args.manifest_sha256
        or sha(__file__) != p["operator_sha256"]
        or str(root) != p["root"]
    ):
        raise ValueError("Bootstrap identity drift")
    budget(p)
    out = root / "runs/v0.0.11/native-import-bootstrap-02"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "joxing-native-import-bootstrap-result-v1",
        "attempt_id": p["attempt_id"],
        "source_commit": p["source_commit"],
        "manifest_sha256": args.manifest_sha256,
        "status": "preflight",
        "new_source_changes": 0,
        "scientific_protocol_sha256": p["native_protocol_sha256"],
    }

    def save():
        data = (json.dumps(report, indent=2) + "\n").encode()
        used = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        if used + len(data) > p["operator_raw_limit_bytes"]:
            raise TimeoutError("Bootstrap receipt credit")
        f = out / "result.tmp"
        f.write_bytes(data)
        f.replace(out / "result.json")

    save()
    began = time.monotonic()
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != p["source_commit"]
        or subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=root, check=False
        ).returncode
    ):
        raise ValueError("Qualified source checkout differs")
    scientific = json.loads(
        (root / "experiments/v0.0.11/native-strike-protocol.json").read_text()
    )
    if digest(scientific) != p["native_protocol_sha256"]:
        raise ValueError("Scientific protocol differs")
    checks = []
    for n, h in {
        **scientific["source_files"],
        **scientific["upstream_files"],
        **scientific["reference_files"],
    }.items():
        if sha(root / n) != h:
            raise ValueError("Qualified source dependency: " + n)
        checks.append([n, h])
    for n, d in p["prerequisite_files"].items():
        if (root / n).stat().st_size != d["bytes"] or sha(root / n) != d["sha256"]:
            raise ValueError("Qualified prerequisite copied bytes differ: " + n)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MUJOCO_GL"] = "egl"
    for n in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        env[n] = "1"
    runtime = str(root / "experiments/v0.0.11/runtime/.venv/bin/python")
    probe = "import sys,json,importlib.metadata,importlib.util;print(json.dumps({'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['mujoco','numpy','imageio','imageio-ffmpeg','Pillow','torch']},'grokbot_locations':list(importlib.util.find_spec('grokbot').submodule_search_locations),'joxing_origin':importlib.util.find_spec('joxing').origin}))"
    observed = json.loads(
        subprocess.check_output(
            [runtime, "-c", probe], cwd=root, env=env, text=True, timeout=1
        )
    )
    if (
        observed["python"] != p["runtime"]["python"]
        or observed["packages"] != p["runtime"]["packages"]
        or set(observed["grokbot_locations"]) != {str(root / "grokbot")}
        or observed["joxing_origin"] != str(root / "src/joxing/__init__.py")
    ):
        raise ValueError("Runtime or package resolution differs from qualified source")
    elapsed = time.monotonic() - began
    if elapsed > p["preflight_wall_limit_s"]:
        raise TimeoutError("Bootstrap preflight wall cap")
    sys.path.insert(0, str(root / "scripts"))
    from joxing_process_ownership import unknown_joxing_processes
    from supervise_motor_pilot import descendants, identity, live

    owner = identity(os.getpid())
    ancestors = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = identity(pid)["parent_pid"]
    if unknown_joxing_processes(owner, ignored_pids=ancestors):
        raise RuntimeError("Another owned Joxing job")
    command = [
        runtime,
        "scripts/supervise_native_strike_probe.py",
        "--protocol-sha256",
        p["native_protocol_sha256"],
        "--source-sha",
        p["source_commit"],
    ]
    report.update(
        status="running",
        process=owner,
        preflight_s=elapsed,
        verified_source_files=len(checks),
        verified_source_digest=digest(checks),
        runtime=observed,
        environment={"PYTHONPATH": str(root), "threads": 1, "MUJOCO_GL": "egl"},
        command=command,
    )
    save()
    started = time.monotonic()
    outer_limit = min(
        p["outer_wall_limit_s"], 900 - p["prior_postgate_upper_s"] - elapsed
    )
    owned = {}
    error = None
    logbytes = 0
    proc = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    report["supervisor_process"] = identity(proc.pid)
    owned[(proc.pid, report["supervisor_process"]["start_ticks"])] = report[
        "supervisor_process"
    ]
    os.set_blocking(proc.stdout.fileno(), False)
    with selectors.DefaultSelector() as selector, (out / "child.log").open("wb") as log:
        selector.register(proc.stdout, selectors.EVENT_READ)
        while proc.poll() is None:
            for child in descendants(proc.pid):
                owned[(child["pid"], child["start_ticks"])] = child
            report["owned_identities"] = list(owned.values())
            save()
            for key, _ in selector.select(0.5):
                data = os.read(key.fileobj.fileno(), 8192)
                if logbytes + len(data) > 32768:
                    error = "Bootstrap child output cap"
                    break
                log.write(data)
                log.flush()
                logbytes += len(data)
            if error or time.monotonic() - started > outer_limit:
                error = error or "Bootstrap outer deadline"
                break
        if proc.poll() is None:
            for child in reversed(list(owned.values())):
                if live(child):
                    try:
                        os.kill(child["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
            for child in owned.values():
                if live(child):
                    try:
                        os.kill(child["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    f = root / "runs/v0.0.11/native-strike-supervision-01/result.json"
    result = json.loads(f.read_text()) if f.exists() else {}
    remaining = [x for x in owned.values() if live(x)]
    report.update(
        status="completed"
        if proc.returncode == 0
        and result.get("status") == "completed"
        and not error
        and not remaining
        and result.get("post_gate_wall_s", 999) <= 40
        and p["prior_postgate_upper_s"] + elapsed + time.monotonic() - started <= 900
        else "halted",
        returncode=proc.returncode,
        supervisor_status=result.get("status"),
        error=error,
        live_recorded_processes=remaining,
        outer_wall_s=time.monotonic() - started,
        actual_native_postgate_s=result.get("post_gate_wall_s"),
    )
    save()
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
