"""Preregister one fixed native interaction, actual continuation and full recording."""

import argparse
import json
from pathlib import Path

from joxing.evidence_storage import sha_file
from joxing.native_strike_qualification import PACKET
from joxing.provenance import digest
from joxing.score_qualification import PACKET as A_PACKET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tests", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / PACKET
    if target.exists():
        raise FileExistsError("Never replace a native strike protocol")
    a = json.loads((root / A_PACKET).read_text())
    policies = {
        k: {"path": n, "sha256": sha_file(root / n)}
        for k, n in {
            "A": "grokbot/redteam/data/cycle-0029-mirror-opponent-gate-summary.json",
            "B": "grokbot/redteam/data/cycle-0025-hit-stay-gate-summary.json",
        }.items()
    }
    p = {
        "schema": "joxing-native-strike-protocol-v1",
        "scope": "Diagnostic source and actual held continuation, no optimization, qualified bout or seed selection",
        "seed": 10001,
        "candidate_corner": "A",
        "horizon_ticks": 3000,
        "checkpoint_tick": 203,
        "source_physics": 3000,
        "actual_continuation_physics": 2797,
        "physics_limit": 5797,
        "policies": policies,
        "noise": 0.05,
        "rng_labels": {"A": "learner", "B": "opponent"},
        "reset_variant": "unchanged grokbot-range-0.50m-v0, diagnostic only",
        "required_a_protocol_sha256": digest(a),
        "requires_same_frozen_source_full_linux_gate": True,
        "expected_tests": args.expected_tests,
        "expected_xfail_nodes": a["expected_xfail_nodes"],
        "prior_postgate_upper_s": a["prior_postgate_upper_s"],
        "prior_raw_upper_bytes": a["prior_raw_upper_bytes"],
        "activation_condition": "Same-source repaired phase A complete and independently audited, all owned prerequisite processes exited; prior upper + ACTUAL phase A postgate +40 <=900; prior raw upper + ACTUAL phase A outputs +24MiB <=256MiB. Otherwise no launch.",
        "post_gate_wall_s": 40,
        "stage_wall_limits_s": {"proof": 25, "audit": 15, "render": 15},
        "display_wall_limit_s": 15,
        "raw_bytes": 24 * 1024**2,
        "maximum_rss_bytes": 4 * 1024**3,
        "threads_per_process": 1,
        "formal_simulation_workers": 1,
        "fixture_process_ceiling": 16,
        "minimum_memory_bytes": 8 * 1024**3,
        "minimum_disk_bytes": 10 * 1024**3,
        "source_host": "xbabe1",
        "source_files": a["source_files"],
        "upstream_files": a["upstream_files"],
        "reference_files": {
            A_PACKET: sha_file(root / A_PACKET),
            **a["reference_files"],
        },
        "input_files": {},
        "rows": ["fixed-source-10001-A", "actual-mid-203-continuation"],
        "criterion": "Exact complete source/controller/contact/score and2797actual held continuation; independent saved observation/command/RNG/history/score audit; all76fixed video frames decoded and first/middle/last reviewed; every fault/not_run kept. Native points are a measured outcome, not required or manufactured for a pass.",
        "video": {
            "selection": "first fixed source regardless of points",
            "frames": 76,
            "fps": 25,
            "dimensions": [800, 448],
            "maximum_bytes": 2 * 1024**2,
            "new_physics": 0,
        },
        "limitations": [
            "Historical controllers use own feedback, not learned opponent history",
            "Range reset and sporting adjudication remain diagnostic",
            "A replay does not qualify stable motor or native competition",
            "Published earlier zero-point result has no complete input stream; this is a fixed diagnostic, not a claim to reproduce unavailable data",
        ],
    }
    assert p["source_files"] == a["source_files"]
    assert p["prior_raw_upper_bytes"] + a["raw_bytes"] + p["raw_bytes"] <= 256 * 1024**2
    target.write_text(json.dumps(p, indent=2) + "\n")
    print(digest(p))


if __name__ == "__main__":
    main()
