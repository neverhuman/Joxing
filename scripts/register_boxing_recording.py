"""Register one finite D0 recording source packet; never overwrite outcomes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from joxing.boxing_evidence import controller_bundles, recording_cases
from joxing.provenance import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tests", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / "experiments/v0.0.11/recording-protocol.json"
    if target.exists():
        raise FileExistsError("Do not overwrite a registered experiment")
    paths = [
        *root.glob("src/joxing/**/*.py"),
        *root.glob("scripts/*.py"),
        *root.glob("tests/*.py"),
        root / "pyproject.toml",
        root / "experiments/v0.0.11/runtime/pyproject.toml",
        root / "experiments/v0.0.11/runtime/uv.lock",
    ]
    source = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    upstream = {}
    for module in ["ms-human-700", "msgym"]:
        base = root / "third_party" / module
        for name in subprocess.check_output(
            ["git", "-C", str(base), "ls-files"], text=True
        ).splitlines():
            p = base / name
            if p.is_file():
                upstream[str(p.relative_to(root))] = hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
    protocol = {
        "schema": "joxing-shared-recording-protocol-v1",
        "scope": "Untrained diagnostic shared-world recording/replay. Zero optimizer updates or valid boxing games. No competition/promotion. Not whole v11completion.",
        "source_files": source,
        "upstream_files": upstream,
        "bundles": controller_bundles(),
        "controller": "diagnostic_command in boxing_evidence.py; own/current/prior velocities plus visible opponent velocity and private corner RNG; no learned weights or normalizer",
        "cases": recording_cases(),
        "delays_ticks": {"A": 1, "B": 2},
        "sensing": "ideal zero-delay kinesthetic/visible kinematics, bounded8sample causal history; no sensor jitter claim",
        "planned_physics": 2642,
        "physics_cap": 12000,
        "runner_wall_s": 180,
        "post_gate_wall_s": 300,
        "raw_bytes": 64 * 1024**2,
        "minimum_memory_bytes": 8 * 1024**3,
        "minimum_disk_bytes": 10 * 1024**3,
        "formal_simulation_workers": 1,
        "fixture_process_ceiling": 16,
        "threads_per_process": 1,
        "expected_tests": args.expected_tests,
        "stage_wall_limits_s": {
            "sync": 180,
            "tests": 420,
            "D0": 180,
            "audit": 90,
            "render": 90,
            "decode": 30,
        },
        "recording": {
            "case": "source-90101",
            "ticks": list(range(0, 1001, 40)),
            "fps": 25,
            "encoded_frames": 26,
            "encoded_duration_s": 1.04,
            "simulated_duration_s": 1.0,
            "render_new_physics": 0,
            "classification": "DIAGNOSTIC - NOT A QUALIFIED BOXING BOUT",
        },
        "numeric_audit_tolerances": {
            "normal_unit_abs": 1e-10,
            "signed_force_abs": 1e-8,
            "signed_force_rel": 1e-10,
            "approach_abs": 1e-10,
            "approach_rel": 1e-12,
            "active_speed_abs": 1e-10,
            "face_abs": 1e-12,
            "extension_abs": 1e-12,
            "replay": "exact all saved command/contact/engine-hash/final application and decision/history tails",
        },
        "stop": "First unknown process, source/hash/ownership drift, cap/floor, nonfinite/engine warning, wrong denominator, replay/rules/contact/timing mismatch. Retain every row/fault; do not retry or replace seed. No executing-source edits.",
    }
    target.write_text(json.dumps(protocol, indent=2) + "\n")
    print(digest(protocol))


if __name__ == "__main__":
    main()
