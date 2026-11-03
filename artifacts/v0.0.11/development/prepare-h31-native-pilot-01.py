"""Publish-verified fresh dedicated local checkout; one bounded native motor pilot."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

repo = Path.cwd()
packet = "experiments/v0.0.11/native-pair-motor-pilot-01.json"
p = json.loads((repo / packet).read_text())
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
epoch = time.monotonic()
root = Path(p["deployment_root"])
assert not root.exists(), (
    "Fresh dedicated root only; never duplicate/retry an owned task"
)
# Source and prospective protocol must already exist on the public commit.
names = [
    packet,
    "src/joxing/native_pair_motor_rl.py",
    "src/joxing/native_pair_motor_vec.py",
    "src/joxing/native_pair_motor_train.py",
    "src/joxing/native_pair_motor_pilot.py",
    "src/joxing/native_pair_motor_evidence.py",
    "scripts/native_pair_guarded_stage.py",
    "scripts/qualify_native_pair_pilot.py",
    "scripts/run_native_pair_pilot.py",
    "scripts/audit_native_pair_pilot.py",
    "scripts/render_native_pair_motor.py",
    "tests/test_native_pair_motor_rl.py",
    "tests/test_native_pair_motor_vec.py",
]


def public(n):
    data = urllib.request.urlopen(
        "https://raw.githubusercontent.com/neverhuman/Joxing/" + commit + "/" + n,
        timeout=15,
    ).read()
    assert data == (repo / n).read_bytes(), n
    return {"path": n, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    verified = list(pool.map(public, names))
root.parent.mkdir(exist_ok=True)
commands = [
    ["git", "clone", "--no-hardlinks", str(repo), str(root)],
    ["git", "checkout", "--detach", commit],
]
for i, cmd in enumerate(commands):
    subprocess.run(
        cmd,
        cwd=None if i == 0 else root,
        capture_output=True,
        check=True,
        timeout=max(1, 45 - (time.monotonic() - epoch)),
    )
for name, sha in p["upstream_commits"].items():
    target = root / "third_party" / name
    subprocess.run(
        [
            "git",
            "clone",
            "--no-hardlinks",
            str(repo / "third_party" / name),
            str(target),
        ],
        capture_output=True,
        check=True,
        timeout=15,
    )
    subprocess.run(
        ["git", "checkout", "--detach", sha],
        cwd=target,
        capture_output=True,
        check=True,
        timeout=5,
    )
assert time.monotonic() - epoch < 45, "Setup reserve first limit"
run = root / "runs/v0.0.11"
run.mkdir(parents=True, exist_ok=True)
protocol_sha = hashlib.sha256(
    json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
cmd = [
    p["runtime_python"],
    "scripts/qualify_native_pair_pilot.py",
    "--packet",
    packet,
    "--source-sha",
    commit,
    "--protocol-sha256",
    protocol_sha,
    "--epoch-start-monotonic",
    str(epoch),
]
env = os.environ | {
    "PYTHONPATH": str(root / "src") + ":" + str(root / "scripts"),
    "PYTHONDONTWRITEBYTECODE": "1",
}
with (run / "native-pair-pilot-launch.log").open("x") as log:
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
start = subprocess.check_output(
    ["ps", "-o", "lstart=", "-p", str(proc.pid)], text=True
).strip()
report = {
    "schema": p["schema"],
    "status": "launched",
    "source_commit": commit,
    "protocol_sha256": protocol_sha,
    "root": str(root),
    "pid": proc.pid,
    "start": start,
    "original_epoch": epoch,
    "setup_elapsed_s": time.monotonic() - epoch,
    "public_verified": verified,
    "command": cmd,
    "checkout_commands": commands,
    "native_physics": "not_yet_observed",
}
(run / "native-pair-pilot-deployment.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
dest = repo / "artifacts/v0.0.11/development/h31-native-pilot-deployment-01.json"
assert not dest.exists()
dest.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            k: report[k]
            for k in (
                "status",
                "source_commit",
                "protocol_sha256",
                "root",
                "pid",
                "start",
                "setup_elapsed_s",
            )
        }
    )
)
