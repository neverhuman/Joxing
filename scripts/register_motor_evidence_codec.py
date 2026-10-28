"""Preregister only saved-corpus codec qualification; no new physics."""

import argparse
import json
from pathlib import Path

from joxing.motor_codec_qualification import PACKET, ROW_IDS, sha_file
from joxing.provenance import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tests", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / PACKET
    if target.exists():
        raise FileExistsError("Never overwrite a registered codec qualification")
    physical = json.loads(
        (root / "experiments/v0.0.11/shared-motor-physical-protocol.json").read_text()
    )
    transfer = json.loads(
        (
            root
            / "artifacts/v0.0.11/development/shared-motor-physical-transfer-01.json"
        ).read_text()
    )
    accounting = json.loads(
        (
            root / "artifacts/v0.0.11/shared-motor-physical/version-accounting.json"
        ).read_text()
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
    p = {
        "schema": "joxing-motor-codec-qualification-v1",
        "rows": ROW_IDS,
        "new_physics": 0,
        "raw_bytes": 4 * 1024**2,
        "maximum_rss_bytes": 512 * 1024**2,
        "post_gate_wall_s": 90,
        "stage_wall_limits_s": {"sync": 180, "tests": 900, "codec": 60, "audit": 30},
        "prior_v11_physics": 12430,
        "prior_v11_raw_upper_bytes": accounting["conservative_raw_upper_bytes"],
        "prior_v11_postgate_upper_s": accounting["conservative_postgate_upper_s"],
        "threads_per_process": 1,
        "formal_simulation_workers": 0,
        "fixture_process_ceiling": 16,
        "minimum_memory_bytes": 8 * 1024**3,
        "minimum_disk_bytes": 10 * 1024**3,
        "expected_tests": args.expected_tests,
        "expected_xfail_nodes": [
            "grokbot/tests/test_referee_fuzz.py::test_restore_does_not_alias_the_snapshot"
        ],
        "source_host": "xbabe1",
        "proof_root": transfer["source_root"],
        "input_files": transfer["files"],
        "upstream_files": physical["upstream_files"],
        "reference_files": {
            n: sha_file(root / n)
            for n in [
                "artifacts/v0.0.11/shared-motor-physical/version-accounting.json",
                "artifacts/v0.0.11/development/shared-motor-physical-integrity-exit-01.json",
                "artifacts/v0.0.11/development/shared-motor-physical-transfer-01.json",
            ]
        },
        "source_files": {str(f.relative_to(root)): sha_file(f) for f in sources},
        "budget_scope": "Original native/pretrained proof ledger; separate Grokbot version protocols and externally requested tags remain unchanged",
        "physical_source_commit": "c9a2b9682c2e500d45c28e3bd15f17a1ab46eef6",
        "physical_protocol_sha256": digest(physical),
        "scope": "Exact23-array saved source1 and independent second-process decode; recognize7-member source2incompleteness. No tail references, physics, optimizer or motor promotion.",
        "ephemeral_encoded_corpus": {
            "maximum_bytes": 32 * 1024**2,
            "persisted": False,
            "immutable_original_retained": True,
            "encoded_sha_and_all_logical_descriptors_required": True,
        },
        "persistent_shard": {
            "first_source_intervals": 60,
            "all_original_field_precisions": True,
        },
        "criterion": "All3rows, every logical dtype/shape/Cbyte in both decoders, deterministic regeneration/persisted shard, no padded incomplete data, source/input/ownership/cap gates. Byte savings are evidence-storage only.",
    }
    assert p["prior_v11_postgate_upper_s"] + p["post_gate_wall_s"] <= 900
    assert p["prior_v11_raw_upper_bytes"] + p["raw_bytes"] <= 256 * 1024**2
    target.write_text(json.dumps(p, indent=2) + "\n")
    print(digest(p))


if __name__ == "__main__":
    main()
