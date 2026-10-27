import datetime, json, os, signal, subprocess, time
from pathlib import Path
ROOT = Path("/home/ubuntu/joxing/v010-neutral-20261027")
OUT = ROOT / "runs/v0.0.10"
os.chdir(ROOT)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
UV = "/home/ubuntu/.local/bin/uv"
status = {"pid": os.getpid(), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "running", "stages": [], "formal_physics_allocation": 12000, "formal_physics_steps_source": "neutral-interface-01/result.json", "formal_simulation_workers": 1, "regression_fixture_physics": "Separate full-suite fixtures; not formal D1/learning/game counters", "threads": 1}
def save():
    p = OUT / "supervisor.json"
    temporary = p.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(p)
save()
stages = [
    ("sync", [UV, "sync", "--project", "experiments/v0.0.10/runtime", "--locked"], 180),
    ("tests", [UV, "run", "--project", "experiments/v0.0.10/runtime", "--locked", "pytest", "-q"], 360),
    ("D1", [UV, "run", "--project", "experiments/v0.0.10/runtime", "--locked", "python", "scripts/qualify_motor_residual.py", "--protocol", "experiments/v0.0.10/neutral-interface-protocol.json", "--inputs", "/home/ubuntu/joxing/deployment-cache/inputs", "--out", "runs/v0.0.10/neutral-interface-01"], 70),
]
for name, cmd, timeout in stages:
    began = time.monotonic()
    with (OUT / (name + ".log")).open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        row = {"stage": name, "command": cmd, "pid": proc.pid, "status": "running"}
        status["stages"].append(row)
        save()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
            row["timeout"] = True
        if name == "tests" and code == 0 and "178 passed" not in (OUT / "tests.log").read_text():
            code = 86
            row["reason"] = "Full178-test pass receipt required"
        row.update(returncode=code, status="completed" if code == 0 else "failed", wall_s=time.monotonic() - began)
        save()
    if code != 0:
        status["status"] = "failed"
        break
else:
    status["status"] = "completed"
status["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save()
