"""Freeze the regression-only sensor evidence component packet before Linux testing."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "experiments/v0.0.11/sensor-evidence-protocol.json"
    if target.exists():
        raise FileExistsError("Do not overwrite a registered packet")
    paths = [
        *ROOT.glob("src/joxing/**/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("tests/*.py"),
        ROOT / "pyproject.toml",
        ROOT / "experiments/v0.0.11/runtime/pyproject.toml",
        ROOT / "experiments/v0.0.11/runtime/uv.lock",
    ]
    source = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    upstream = {}
    for name in ["ms-human-700", "msgym"]:
        base = ROOT / "third_party" / name
        files = subprocess.check_output(
            ["git", "-C", str(base), "ls-files"], text=True
        ).splitlines()
        for name in files:
            p = base / name
            if p.is_file():
                upstream[str(p.relative_to(ROOT))] = hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
    protocol = {
        "schema": "joxing-sensor-evidence-qualification-v1",
        "scope": "Regression fixtures only for the sensor recorder/transport auditor/comparators; no complete physical referee audit or formal D1rollout, renderer, optimizer or boxing-game outcome. Do not automatically start any D1sensor trajectory.",
        "source_files": source,
        "upstream_files": upstream,
        "expected_tests": 301,
        "stage_wall_limits_s": {"sync": 180, "tests": 420},
        "minimum_memory_bytes": 8 * 1024**3,
        "minimum_disk_bytes": 10 * 1024**3,
        "formal_gate_physics_steps": 0,
        "prior_formal_v11_physics_steps": 2642,
        "fixture_process_ceiling": 16,
        "threads_per_numerical_process": 1,
        "stop": "First source drift, unknown owned job, count/exit failure or stage deadline. Retain logs and every recorded PID/start identity.",
    }
    target.write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        hashlib.sha256(
            json.dumps(
                protocol, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )


if __name__ == "__main__":
    main()
