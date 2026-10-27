import datetime, json, os, signal, subprocess, time
from pathlib import Path
ROOT = Path("/home/ubuntu/joxing/v010-inputs-20261027")
OUT = ROOT / "runs/v0.0.10"
os.chdir(ROOT)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
UV = "/home/ubuntu/.local/bin/uv"
status = {"pid": os.getpid(), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "running", "stages": [], "new_physics_steps": 0, "simulation_workers": 0, "threads": 1}
def save():
    p = OUT / "supervisor.json"
    temporary = p.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(p)
save()
stages = [
    ("sync", [UV, "sync", "--project", "experiments/v0.0.10/runtime", "--locked"], 180),
    ("tests", [UV, "run", "--project", "experiments/v0.0.10/runtime", "--locked", "pytest", "-q", "tests/test_motor_inputs.py"], 30),
    ("D0", [UV, "run", "--project", "experiments/v0.0.10/runtime", "--locked", "python", "scripts/audit_motor_inputs.py", "--protocol", "experiments/v0.0.10/input-audit-protocol.json", "--h1", "runs/v0.0.10/inputs/H1", "--h3", "runs/v0.0.10/inputs/H3", "--out", "runs/v0.0.10/input-audit-01"], 70),
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
        row.update(returncode=code, status="completed" if code == 0 else "failed", wall_s=time.monotonic() - began)
        save()
    if code != 0:
        status["status"] = "failed"
        break
else:
    status["status"] = "completed"
status["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save()
