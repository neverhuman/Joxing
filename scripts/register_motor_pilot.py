"""Materialize the fixed D2 preregistration before any formal outcomes."""

import hashlib
import subprocess

from joxing.motor_pilot import SAC_PARAMETERS
from joxing.motor_protocol import validate
from joxing.provenance import ROOT, digest, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / "experiments/v0.0.10/protocol.json"
    if target.exists():
        raise FileExistsError("Never replace a registered protocol in place")
    paths = [
        *ROOT.glob("src/joxing/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("tests/*.py"),
        ROOT / "pyproject.toml",
        ROOT / "experiments/v0.0.10/runtime/pyproject.toml",
        ROOT / "experiments/v0.0.10/runtime/uv.lock",
        ROOT / "artifacts/v0.0.10/D1/audit.json",
        ROOT / "artifacts/v0.0.10/D1/raw/result.json",
        ROOT / "artifacts/v0.0.10/R1/audit.json",
    ]
    upstream = {}
    for name in ("ms-human-700", "msgym"):
        base = ROOT / "third_party" / name
        names = subprocess.check_output(
            ["git", "-C", str(base), "ls-files"], text=True
        ).splitlines()
        for filename in names:
            path = base / filename
            if path.is_file():
                upstream[str(path.relative_to(ROOT))] = sha(path)
    p = {
        "schema": "joxing-motor-pilot-D2-0.0.10",
        "scope": "Physical motor adaptation, not boxing or self-play. No best-checkpoint selection.",
        "source_files": {
            str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)
        },
        "upstream_files": upstream,
        "inputs": {
            "model.zip": "0a3607c0a9d745a6bb5a5ffceb7539a7405d455943abf283d4f8107f22a47829",
            "normalizer.pkl": "e359b32fa026a02746cffcba6ee3ef9cab4389d37c74b19bd21caa0f3abc0086",
            "config.json": "7d31f757a782ddebf6fe4f2496403492453727a26dfee181c825afa37488b6dd",
        },
        "bundle_hash": "63cd847d4016a880e243603319b5c9b0190251cec8c1350e17c1097a8aa4bce8",
        "training_seeds": [81011, 81029],
        "evaluation_seeds": list(range(83000, 83008)),
        "transitions_per_seed": 8192,
        "sac": SAC_PARAMETERS,
        "reset_stream": "8200000+100000*training_index+worker_index+4*episode_index; 0<=episode_index<25000",
        "acting_schedule": "training_batch4_eval_scalar1_v1; frozen teacher always scalar",
        "qualification": {"linux_tests": 196, "D1_audit_required": True},
        "caps": {
            "physics_steps": 550000,
            "wall_s": 1800,
            "raw_bytes": 1024**3,
            "simulation_workers": 4,
            "threads": 1,
            "minimum_memory_bytes": 8 * 1024**3,
            "minimum_disk_bytes": 10 * 1024**3,
        },
        "audit_caps": {"physics_steps": 50000, "wall_s": 300},
        "per_seed_training_wall_s": 600,
        "raw_credits": {
            "parent_bytes": 192 * 1024**2,
            "worker_bytes": 32 * 1024**2,
            "evaluation_bytes": 12 * 1024**2,
            "replay_bytes": 12 * 1024**2,
        },
        "prior": {
            "physics_steps": 10394,
            "formal_wall_s": 25.066974967718124,
            "raw_charge_bytes": 32 * 1024**2,
        },
        "criteria": {
            "minimum_mean_paired_gain_s_each_seed": 0.5,
            "minimum_final_five_second_survivors_each_seed": 4,
            "evaluations_per_stage_per_seed": 8,
            "required_exact_final_83000_mid_replays": 2,
            "invalid_allowed": 0,
        },
        "stop": "First fault, source/bundle drift, replay/mask/command mismatch, unknown owner or first cap. Preserve every future not_run row; no parameter changes or unchanged larger sweep.",
    }
    result = validate(p, digest(p), source=False)
    write_json(target, p)
    print(result)


if __name__ == "__main__":
    main()
