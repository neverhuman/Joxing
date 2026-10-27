"""Register one finite eight-row D1sensor source packet; never overwrite outcomes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from joxing.boxing_evidence import controller_bundles
from joxing.provenance import digest
from joxing.sensor_protocol import sensor_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tests", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / "experiments/v0.0.11/sensor-proof-protocol.json"
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
    oracle_local = root / "runs/v0.0.11/recording-evidence-01/proof"
    oracle_files = {}
    for name in [
        "contract.json",
        "initial.json",
        "final.json",
        "trace.json",
        "tape.npz",
    ]:
        path = oracle_local / "source-90101" / name
        oracle_files["source-90101/" + name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    protocol = {
        "schema": "joxing-sensor-proof-protocol-v1",
        "scope": "Untrained causal delayed/dropped/reordered sensor correctness with exact retained D0oracle and full replay. Zero optimizer updates or valid boxing games; no motor/competition/promotion or whole-v11completion.",
        "model": {
            "binary": "artifacts/generated/v0.0.11/sensor-model/scene.mjb",
            "binary_bytes": 180809620,
            "binary_sha256": "02806d29bba614ea3f110f3729d4cdb7319992f0df41986241a05c9bcfdf03c3",
            "report": "artifacts/generated/v0.0.11/sensor-model/scene.json",
            "report_sha256": "ab6d804b33791df58657a333cb29fc28e52a027158d839e31238a46b25d13d12",
            "scene_hash": "d31ab28092832ba5bdb21cc89c62fd8b6f85286f380dd401fb30723c3d93ed0d",
        },
        "oracle_root": "/home/ubuntu/joxing/v011-recording-20261027/runs/v0.0.11/recording-proof-01",
        "oracle_source_commit": "d2b582ee05041c21129b393945240b42e51153f8",
        "oracle_files": oracle_files,
        "prior_v11_physics": 2642,
        "prior_v11_raw_bytes": 55053851,
        "prior_v11_post_gate_wall_s": 40.476303,
        "whole_v11_caps": {
            "physics": 120000,
            "raw_bytes": 256 * 1024**2,
            "post_gate_wall_s": 900,
            "simulation_workers": 2,
        },
        "source_files": source,
        "upstream_files": upstream,
        "bundles": controller_bundles(),
        "controller": "diagnostic_command in boxing_evidence.py; own/current/prior velocities plus visible opponent velocity and private corner RNG; no learned weights or normalizer",
        "cases": sensor_cases(),
        "delays_ticks": {"A": 1, "B": 2},
        "sensing": "Fixed per-corner tick schedules/drop/reordering; every-tick delivery, sampled actual physical states, bounded8sample history; no random RF/noise robustness claim",
        "planned_physics": 2748,
        "physics_cap": 10000,
        "runner_wall_s": 180,
        "post_gate_wall_s": 300,
        "raw_bytes": 96 * 1024**2,
        "minimum_memory_bytes": 8 * 1024**3,
        "minimum_disk_bytes": 10 * 1024**3,
        "formal_simulation_workers": 1,
        "fixture_process_ceiling": 16,
        "threads_per_process": 1,
        "expected_tests": args.expected_tests,
        "stage_wall_limits_s": {
            "sync": 180,
            "tests": 420,
            "D1": 180,
            "audit": 90,
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
