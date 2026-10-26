"""Independent raw accounting, selection, actor and replay audit for extension."""

import argparse
import json
from pathlib import Path

import numpy as np
from verify_wave import read, replay_error

from joxing.provenance import digest, manifest, write_json
from joxing.stance_env import StanceEnv
from joxing.stance_train import file_hash
from joxing.wave import ARMS, bootstrap_pair, canonical_actor_hash, select_arm
from joxing.wave_extension import extension_config


def verify(run, original):
    import torch
    from stable_baselines3.sac.policies import Actor

    torch.set_num_threads(1)
    p = read(run / "protocol.json")
    config = extension_config(p)
    provenance = read(run / "provenance.json")
    assert provenance["protocol_hash"] == digest(p)
    assert (
        provenance["parent_protocol_hash"]
        == read(original / "provenance.json")["protocol_hash"]
    )
    result = read(run / "result.json")
    registry = read(run / "registry.json")
    assert registry == result["registry"]
    assert set(registry) == {x["id"] for x in p["experiments"]}
    assert all(
        x["status"] in {"completed", "incomplete", "not_run", "failed"}
        for x in registry.values()
    )
    baseline = read(run / "B0/result.json")
    chosen = baseline["chosen_excitation"]
    assert chosen == read(original / "B0/result.json")["chosen_excitation"]
    assert (
        file_hash(run / "B0/reused-diagnostic-evaluations.json")
        == baseline["reused_diagnostic_sha256"]
        == file_hash(original / f"B0/exc-{chosen}-evaluations.json")
    )
    env = StanceEnv(training=False)
    baseline_rows = read(run / "B0/screen-constant-evaluations.json")
    assert [x["scenario_id"] for x in baseline_rows] == p["screen_ids"]
    assert all(not x["invalid"] and x["status"] != "not_run" for x in baseline_rows)
    physics = baseline["physics_substeps"] + baseline["preparation_substeps"]
    attempts = len(baseline_rows)
    if (run / "B0/confirmation-baseline-result.json").exists():
        c = read(run / "B0/confirmation-baseline-result.json")
        physics += c["physics_substeps"] + c["preparation_substeps"]
        rows = read(run / "B0/confirmation-constant-evaluations.json")
        assert [x["scenario_id"] for x in rows] == p["confirmation_ids"]
        assert all(not x["invalid"] and x["status"] != "not_run" for x in rows)
        attempts += len(rows)
    findings = []
    results = {}
    screen = {}
    for key, row in registry.items():
        if key == "B0" or row["status"] == "not_run":
            continue
        assert row["status"] in {"completed", "incomplete"}
        folder = run / key
        r = read(folder / "result.json")
        results[key] = r
        assert r["source_unchanged"]
        assert env.specification() == read(folder / "specification.json")
        before, after = [
            read(folder / f"{n}-evaluations.json") for n in ("initial", "trained")
        ]
        ids = p["screen_ids"] if key.startswith("S-") else p["confirmation_ids"]
        assert (
            [x["scenario_id"] for x in before]
            == [x["scenario_id"] for x in after]
            == ids
        )
        assert bootstrap_pair(before, after) == r["summary"]
        if key.startswith("S-"):
            assert bootstrap_pair(baseline_rows, after)["invalid"] == 0
            screen[key[-1]] = r
        assert all(x["actor_hash"] == r["initial_actor_hash"] for x in before)
        assert all(x["actor_hash"] == r["final_actor_hash"] for x in after)
        assert all(
            x["reset_audit"]["projected_equality_error"] < 1e-8
            and x["reset_audit"]["max_penetration_m"] <= 0.002
            for x in before + after
        )
        assert row["status"] == (
            "completed" if r["scheduled_training_complete"] else "incomplete"
        )
        if r["scheduled_training_complete"]:
            assert r["training_transitions"] == p["steps"]
            assert (
                r["updates"] == (p["steps"] - config["learner"]["learning_starts"]) // 4
            )
        assert np.isclose(
            r["counts"]["scaled_reward_sum"],
            r["counts"]["raw_reward_sum"] * ARMS[r["arm"]]["reward_scale"],
            rtol=1e-12,
        )
        assert r["evaluation_physics_substeps"] == sum(
            x["physics_steps"] for x in before + after
        )
        physics += (
            r["counts"]["physics_substeps"]
            + r["counts"]["preparation_substeps"]
            + r["evaluation_physics_substeps"]
            + r["evaluation_preparation_substeps"]
        )
        attempts += len(before) + len(after)
        for name, checkpoint in (("initial", "initial"), ("trained", "final")):
            path = folder / f"{checkpoint}-actor.pt"
            assert file_hash(path) == r["actor_files"][checkpoint]
            actor = Actor.load(path, device="cpu")
            assert canonical_actor_hash(actor) == r[f"{checkpoint}_actor_hash"]
            record = read(folder / f"{name}-replay.json")
            actual, first = replay_error(env, record, actor=actor)
            _, second = replay_error(env, record, actor=actor)
            actions, _ = replay_error(env, record, actions=record["actions"])
            findings.append(
                {
                    "experiment": key,
                    "policy": name,
                    **actual,
                    "same_host_repeat_error": float(np.max(abs(first - second))),
                    "recorded_action_replay": actions,
                }
            )
    assert set(screen) == set(ARMS)
    assert len({r["initial_actor_hash"] for r in screen.values()}) == 1
    assert select_arm(screen) == result["selection"]
    for seed in p["confirmation_seeds"]:
        pair = [
            results[k]["initial_actor_hash"]
            for k in (f"R-A-{seed}", f"R-W-{seed}")
            if k in results
        ]
        assert len(set(pair)) <= 1
    assert physics == result["accounted_physics_substeps"]
    assert attempts == 32 + 64 * len(results) + (
        32 if result["selection"]["qualified"] else 0
    )
    same = all(x["same_host_repeat_error"] < 1e-9 for x in findings)
    if manifest()["host"] == provenance["host"]:
        same = same and all(
            x["integration_state_max_error"] < 1e-9
            and x["outcome_tick_matches"]
            and x["recorded_action_replay"]["integration_state_max_error"] < 1e-9
            for x in findings
        )
    cross = all(
        x["integration_state_max_error"] < 1e-5 and x["outcome_tick_matches"]
        for x in findings
    )
    return manifest() | {
        "auditor_script_sha256": file_hash(Path(__file__)),
        "source_run_commit": provenance["code_commit"],
        "protocol_hash": provenance["protocol_hash"],
        "accounting_passed": True,
        "evaluation_attempts": attempts,
        "reused_diagnostic_attempts_excluded": 8,
        "invalid_evaluation_attempts": 0,
        "recomputed_physics_substeps": physics,
        "findings": findings,
        "same_host_repeat_passed": same,
        "cross_host_reference_parity_passed": cross,
        "passed": same and cross,
        "tolerances": {"same_host": 1e-9, "cross_host_full_state": 1e-5},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.run, args.original)
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
