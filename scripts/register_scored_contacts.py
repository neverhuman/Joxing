"""Register the fixed scoring audit and four rigid adapter controls."""

import argparse
import json
from pathlib import Path

from joxing.evidence_storage import sha_file
from joxing.provenance import digest
from joxing.score_qualification import PACKET, ROW_IDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tests", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / PACKET
    if target.exists():
        raise FileExistsError("Never replace a registered scoring packet")
    prior = json.loads(
        (
            root / "artifacts/v0.0.11/motor-codec-bindings-repair/conclusion.json"
        ).read_text()
    )
    failed = json.loads(
        (
            root / "artifacts/v0.0.11/scored-contact-failed-01/supervision.json"
        ).read_text()
    )
    failed_transfer = json.loads(
        (
            root
            / "artifacts/v0.0.11/development/scored-contact-failed-transfer-01.json"
        ).read_text()
    )
    physical = json.loads(
        (root / "experiments/v0.0.11/shared-motor-physical-protocol.json").read_text()
    )
    files = sorted(
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
    inputs = {}
    for case in ["source-90101", "contact-source-90104"]:
        for name in ["initial", "final", "trace", "contract"]:
            f = (
                root
                / "runs/v0.0.11/recording-evidence-01/proof"
                / case
                / (name + ".json")
            )
            inputs["score-inputs/" + case + "/" + name + ".json"] = {
                "bytes": f.stat().st_size,
                "sha256": sha_file(f),
                "original_local_path": str(f.relative_to(root)),
            }
    refs = [
        "artifacts/v0.0.11/motor-codec-bindings-repair/conclusion.json",
        "artifacts/v0.0.11/development/scoring-evidence-native-inventory-01.json",
        "artifacts/v0.0.11/development/scoring-evidence-summary-inventory-01.json",
        "artifacts/v0.0.11/development/scored-contact-development-01.json",
        "artifacts/v0.0.11/development/native-strike-development-01.json",
        "artifacts/v0.0.11/scored-contact-failed-01/result.json",
        "artifacts/v0.0.11/scored-contact-failed-01/supervision.json",
        "artifacts/v0.0.11/development/scored-contact-failed-transfer-01.json",
    ]
    p = {
        "schema": "joxing-score-qualification-v1",
        "rows": ROW_IDS,
        "source_host": "xbabe1",
        "new_native_physics": 0,
        "fixture_physics_limit": 1000,
        "fixture_variant": "rigid-adapter-control-v1; not a pruned native model or athlete evaluation",
        "prior_native_proof_physics": 12430,
        "raw_bytes": 4 * 1024**2,
        "maximum_rss_bytes": 512 * 1024**2,
        "post_gate_wall_s": 30,
        "stage_wall_limits_s": {"sync": 180, "tests": 900, "proof": 20, "audit": 10},
        "threads_per_process": 1,
        "formal_simulation_workers": 1,
        "fixture_process_ceiling": 16,
        "minimum_memory_bytes": 8 * 1024**3,
        "minimum_disk_bytes": 10 * 1024**3,
        "expected_tests": args.expected_tests,
        "expected_xfail_nodes": [
            "grokbot/tests/test_referee_fuzz.py::test_restore_does_not_alias_the_snapshot"
        ],
        "prior_postgate_upper_s": prior["conservative_postgate_upper_s"]
        + failed["post_gate_wall_s"],
        "prior_raw_upper_bytes": prior["new_raw_upper_bytes"]
        + failed_transfer["bytes"]
        + 65536,
        "source_files": {str(f.relative_to(root)): sha_file(f) for f in files},
        "upstream_files": physical["upstream_files"],
        "reference_files": {n: sha_file(root / n) for n in refs},
        "input_files": inputs,
        "criterion": "All seven rows and exact saved ledger/final state; guard/no-contact rejection controls; published summary remains not-explainable; all four real rigid adapter controls score head2/torso1 under unchanged rules with reversed geom order; complete1000entered/completed physics receipts; independent saved-data audit and source/ownership/budget gates.",
        "repair_scope": "Preserve failed2pass/1fail/4not_run legacy -Infinity parsing attempt; isolate known unmeasurable statistics from finite selected game data; physical evidence parser stays strict",
        "scope": "Retrospective saved native diagnostics plus repeated rigid physical controls; no held-out athlete or boxing competence, no optimizer or self-play, no scoring/rules/model changes.",
        "non_goals": [
            "native motor continuation",
            "native legal punch feasibility",
            "v21 root-cause inference from absent contact evidence",
            "competitive promotion",
            "external-lane v22 run",
            "release tag",
        ],
    }
    assert p["prior_postgate_upper_s"] + p["post_gate_wall_s"] <= 900
    assert p["prior_raw_upper_bytes"] + p["raw_bytes"] <= 256 * 1024**2
    target.write_text(json.dumps(p, indent=2) + "\n")
    print(digest(p))


if __name__ == "__main__":
    main()
