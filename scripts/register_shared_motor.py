"""Freeze the four shared pretrained sources and one actual held replay once."""

import argparse
import json
from pathlib import Path

from joxing.provenance import digest
from joxing.shared_motor_proof import PACKET, file_sha, planned_rows, validate_packet
from joxing.shared_motor_saved_audit import require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tests", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / PACKET
    if target.exists():
        raise FileExistsError("Never overwrite a frozen physical proof protocol")
    old = json.loads(
        (root / "experiments/v0.0.11/shared-motor-input-protocol.json").read_text()
    )
    inputs = json.loads(
        (root / "artifacts/v0.0.11/motor-input/result.json").read_text()
    )
    require(
        inputs["status"] == "passed"
        and len(inputs["rows"]) == 4
        and all(r["status"] == "passed" for r in inputs["rows"]),
        "Archived input/policy prerequisite failed",
    )
    sources = sorted(
        {
            *root.glob("src/joxing/**/*.py"),
            *root.glob("scripts/*.py"),
            *root.glob("tests/**/*.py"),
            *root.glob("grokbot/**/*.py"),
            *root.glob("grokbot/**/*.json"),
            root / ".github/workflows/ci.yml",
            root / "pyproject.toml",
            root / "experiments/v0.0.11/runtime/pyproject.toml",
            root / "experiments/v0.0.11/runtime/uv.lock",
        }
    )
    refs = [
        "artifacts/v0.0.6/native/evaluations.json",
        "experiments/v0.0.6/protocol.json",
        "experiments/v0.0.11/shared-motor-input-protocol.json",
        "artifacts/v0.0.11/motor-input/result.json",
        "artifacts/v0.0.11/development/shared-motor-input-audit-01.json",
    ]
    native = [
        r
        for r in json.loads((root / refs[0]).read_text())
        if r["mode"] == "scheduled_restored" and 71000 <= r["seed"] <= 71003
    ]
    require(
        len(native) == 4
        and all(
            r["status"] == "completed"
            and r["horizon_reached"]
            and r["control_steps"] == 176
            for r in native
        ),
        "Frozen native horizons",
    )
    p = {
        k: old[k]
        for k in (
            "inputs_root",
            "source_host",
            "actor_sha256",
            "model_sha256",
            "normalizer_sha256",
            "normalizer_stats_sha256",
            "decoder_amplitude",
            "minimum_memory_bytes",
            "minimum_disk_bytes",
            "fixture_process_ceiling",
            "threads_per_process",
            "upstream_files",
        )
    }
    p.update(
        schema="joxing-shared-motor-physical-v1",
        rows=planned_rows(),
        planned_physics=17397,
        physics_cap=60000,
        runner_wall_s=180,
        post_gate_wall_s=300,
        raw_bytes=32 * 1024**2,
        formal_simulation_workers=1,
        prior_v11_physics=5390,
        prior_v11_raw_bytes=202302252,
        prior_v11_post_gate_upper_s=600,
        expected_tests=args.expected_tests,
        expected_xfail_nodes=[
            "grokbot/tests/test_referee_fuzz.py::test_restore_does_not_alias_the_snapshot"
        ],
        prior_post_gate_accounting="Conservative600s upper credit encompasses428.510437s prior registered upper +44.993312s D1 completion +30.964384s independent D1 read audit +6.536687s input oracle, leaving89s supplementary read/audit reserve; tests are fixtures outside formal physics/postgate.",
        stage_wall_limits_s={
            "sync": 180,
            "tests": 660,
            "proof": 180,
            "audit": 60,
            "render": 40,
            "decode": 20,
        },
        source_files={str(f.relative_to(root)): file_sha(f) for f in sources},
        reference_files={name: file_sha(root / name) for name in refs},
        native_dependencies=native,
        environment=json.loads((root / "experiments/v0.0.6/protocol.json").read_text())[
            "environment"
        ],
        recording_source="source-71000",
        variant="pretrained-parallel-walk-1ms-v1",
        acceptance="All4fixed worlds/both athletes reach3520ticks without tracking/physical/numerical failure; exact203->3520replay; all attempts retained. Short locomotion is not guard/boxing. No optimization.",
    )
    p["bundle_sha256"] = digest(
        {
            k: p[k]
            for k in (
                "actor_sha256",
                "model_sha256",
                "normalizer_sha256",
                "normalizer_stats_sha256",
                "decoder_amplitude",
                "variant",
                "environment",
            )
        }
    )
    validate_packet(p)
    target.write_text(json.dumps(p, indent=2) + "\n")
    print(digest(p))


if __name__ == "__main__":
    main()
