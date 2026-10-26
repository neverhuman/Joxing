"""Independently audit frozen v0.0.2 receipts and replay all exported policies."""

import argparse
import json
from pathlib import Path

import numpy as np

from joxing.provenance import digest, manifest, write_json
from joxing.strike import StrikeConfig, StrikeRig


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.run / "protocol.json")
    result = read(args.run / "result.json")
    check_hash = protocol.copy()
    check_hash.pop("protocol_hash")
    if digest(check_hash) != protocol["protocol_hash"]:
        raise ValueError("Frozen protocol digest mismatch")
    if result["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Run/protocol mismatch")
    config = StrikeConfig(
        **{
            **protocol["config"],
            "target": tuple(protocol["config"]["target"]),
            "phase_ends_s": tuple(protocol["config"]["phase_ends_s"]),
        }
    )
    findings = []
    for seed in protocol["training_seeds"]:
        path = args.run / f"seed-{seed}"
        summary = read(path / "result.json")
        with np.load(path / "checkpoint.npz", allow_pickle=False) as checkpoint:
            params = checkpoint["mean"].copy()
        if digest(params.tolist()) != summary["policy_hash"]:
            raise ValueError("Checkpoint identity mismatch")
        for name in ("initial", "trained"):
            rows = read(path / f"{name}-evaluations.json")
            if [r["scenario_seed"] for r in rows] != protocol["eval_seeds"]:
                raise ValueError(
                    "Missing, duplicated or unexpected evaluation scenario"
                )
            if not all(r["valid"] for r in rows):
                raise ValueError("Invalid evaluation included")
            calculated = float(np.mean([r["strike_withdrawal_success"] for r in rows]))
            if (
                calculated
                != summary["evaluation"][name]["strike_withdrawal_success_rate"]
            ):
                raise ValueError("Evaluation summary disagrees with raw rows")
            if any(r["points"] != sum(e["points"] for e in r["events"]) for r in rows):
                raise ValueError("Points disagree with ledger")
        rig = StrikeRig(config)
        if (
            rig.model_hash != summary["model_hash"]
            or rig.task_hash != summary["task_hash"]
        ):
            raise ValueError("Local model/task differs from proof")
        replay = rig.rollout(params, protocol["diagnostic_scenario_seed"], record=True)
        with np.load(path / "trained-replay.npz", allow_pickle=False) as saved:
            trace_error = float(
                np.max(np.abs(np.asarray(replay["trace"]) - saved["trace"]))
            )
            qpos_error = float(np.max(np.abs(replay["qpos"] - saved["qpos"])))
        original = read(path / "trained-replay.json")
        identities = [
            (
                e["event_id"],
                e["onset_tick"],
                e["finalized_tick"],
                e["points"],
                e["zone"],
            )
            for e in replay["events"]
        ]
        expected = [
            (
                e["event_id"],
                e["onset_tick"],
                e["finalized_tick"],
                e["points"],
                e["zone"],
            )
            for e in original["events"]
        ]
        snapshot = read(path / "contact-snapshot.json")
        rig.restore(snapshot)
        while rig.tick < rig.n_steps:
            rig.advance()
        restored = rig.finish()
        restore_error = float(np.max(np.abs(rig.data.qpos - replay["qpos"][-1])))
        findings.append(
            {
                "training_seed": seed,
                "trace_max_error": trace_error,
                "qpos_max_error": qpos_error,
                "event_identity_match": identities == expected,
                "restored_score": restored["points"],
                "restored_final_qpos_error": restore_error,
                "passed": bool(
                    trace_error < 1e-8
                    and qpos_error < 1e-9
                    and identities == expected
                    and restore_error < 1e-9
                    and restored["points"] == original["points"]
                ),
            }
        )
    report = manifest() | {
        "experiment": "v0.0.2-independent-receipt-and-replay-audit",
        "proof_code_commit": result["code_commit"],
        "protocol_hash": protocol["protocol_hash"],
        "seeds": findings,
        "passed": all(f["passed"] for f in findings),
        "scope": "Controlled short task replay and receipt accounting, not general cross-platform boxing determinism",
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
