"""Recompute decisions from raw rows and audit actors/engine continuation."""

import argparse
import json
from pathlib import Path

import numpy as np

from joxing.provenance import digest, manifest, write_json
from joxing.stance_env import StanceEnv
from joxing.stance_train import file_hash
from joxing.wave import (
    ARMS,
    bootstrap_pair,
    canonical_actor_hash,
    select_arm,
    validate_protocol,
)


def read(path):
    return json.loads(path.read_text())


def replay_error(env, record, actions=None, actor=None):
    env.restore(record["snapshot"])
    if actions is None:
        while not env.done:
            action, _ = actor.predict(env.observe(), deterministic=True)
            env.step(action)
    else:
        for action in actions:
            assert not env.done, "Replay terminated early"
            env.step(action)
    state = np.asarray(env.snapshot()["state"])
    ref = np.asarray(record.get("exported_final_state", record["final_state"]))
    result = record.get("exported_result", record["result"])
    return {
        "integration_state_max_error": float(np.max(abs(state - ref))),
        "qpos_max_error": float(
            np.max(abs(state[1 : 1 + env.model.nq] - ref[1 : 1 + env.model.nq]))
        ),
        "qvel_max_error": float(
            np.max(
                abs(
                    state[1 + env.model.nq : 1 + env.model.nq + env.model.nv]
                    - ref[1 + env.model.nq : 1 + env.model.nq + env.model.nv]
                )
            )
        ),
        "outcome_tick_matches": env.tick == result["physics_steps"]
        and env.result()["reason"] == result["reason"],
    }, state


def verify(run):
    import torch
    from stable_baselines3.sac.policies import Actor

    torch.set_num_threads(1)
    protocol = validate_protocol(read(run / "protocol.json"))
    provenance = read(run / "provenance.json")
    assert digest(protocol) == provenance["protocol_hash"]
    result = read(run / "result.json")
    registry = read(run / "registry.json")
    assert result["registry"] == registry
    assert set(registry) == {r["id"] for r in protocol["experiments"]}
    assert all(
        r["status"] in {"completed", "incomplete", "not_run", "failed"}
        for r in registry.values()
    )
    env = StanceEnv(training=False)
    findings = []
    screen = {}
    attempts = 0
    initial_hashes = {}
    physical = 0
    for key, item in registry.items():
        if item["status"] not in {"completed", "incomplete"}:
            continue
        folder = run / key
        r = read(folder / "result.json")
        if key == "B0":
            for path in folder.glob("*-evaluations.json"):
                rows = read(path)
                assert len({x["scenario_id"] for x in rows}) == len(rows)
                assert all(not x["invalid"] and x["status"] != "not_run" for x in rows)
                attempts += len(rows)
            means = [
                {
                    "excitation": exc,
                    "mean_survival_s": float(
                        np.mean(
                            [
                                x["elapsed_sim_s"]
                                for x in read(folder / f"exc-{exc}-evaluations.json")
                            ]
                        )
                    ),
                }
                for exc in protocol["constants"]
            ]
            assert means == r["diagnostic_means"]
            assert (
                max(means, key=lambda x: x["mean_survival_s"])["excitation"]
                == r["chosen_excitation"]
            )
            physical += r["physics_substeps"] + r["preparation_substeps"]
            if (folder / "confirmation-baseline-result.json").exists():
                extra = read(folder / "confirmation-baseline-result.json")
                physical += extra["physics_substeps"] + extra["preparation_substeps"]
            continue
        if key == "P0":
            assert file_hash(folder / "common-transitions.npz") == r["dataset_sha256"]
            assert r["speedup"] == r["medians"]["optimized"] / r["medians"]["reference"]
            assert r["passed"] == (
                r["speedup"] >= 2
                and all(x["finite"] and x["actor_changed"] for x in r["samples"])
            )
            physical += r["physics_substeps"] + r["preparation_substeps"]
            continue
        if key == "X0":
            physical += r["physics_substeps"] + r["preparation_substeps"]
            for name in ("initial", "trained"):
                record = read(folder / f"{name}-actions.json")
                actual, first = replay_error(env, record, actions=record["actions"])
                _, second = replay_error(env, record, actions=record["actions"])
                findings.append(
                    {
                        "experiment": key,
                        "policy": name,
                        "mode": "recorded_actions",
                        **actual,
                        "same_host_repeat_error": float(np.max(abs(first - second))),
                    }
                )
            continue
        assert env.specification() == read(folder / "specification.json")
        before, after = [
            read(folder / f"{n}-evaluations.json") for n in ("initial", "trained")
        ]
        expected_ids = protocol["splits"][
            "screen" if key.startswith("S-") else "confirmation"
        ]
        assert (
            [x["scenario_id"] for x in before]
            == [x["scenario_id"] for x in after]
            == expected_ids
        )
        assert bootstrap_pair(before, after) == r["summary"]
        assert all(
            x["reset_audit"]["projected_equality_error"] < 1e-8
            and x["reset_audit"]["max_penetration_m"] <= 0.002
            for x in before + after
        )
        attempts += len(before) + len(after)
        assert r["evaluation_physics_substeps"] == sum(
            x["physics_steps"] for x in before + after
        )
        physical += (
            r["counts"]["physics_substeps"]
            + r["counts"]["preparation_substeps"]
            + r["evaluation_physics_substeps"]
            + r["evaluation_preparation_substeps"]
        )
        if key.startswith("S-"):
            screen[key[-1]] = r
        initial_hashes[key] = r["initial_actor_hash"]
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
                    "mode": "closed_loop",
                    **actual,
                    "same_host_repeat_error": float(np.max(abs(first - second))),
                    "recorded_action_replay": actions,
                }
            )
    if screen:
        assert set(screen) == set(ARMS)
        assert len({r["initial_actor_hash"] for r in screen.values()}) == 1
        assert select_arm(screen) == result["selection"]
    for seed in (263, 431):
        keys = [
            f"R-{a}-{seed}" for a in ("A", "W") if f"R-{a}-{seed}" in initial_hashes
        ]
        assert len({initial_hashes[k] for k in keys}) <= 1
    same_host = all(f["same_host_repeat_error"] < 1e-9 for f in findings)
    same_source_host = manifest()["host"] == provenance["host"]
    if same_source_host:
        same_host = same_host and all(
            f["integration_state_max_error"] < 1e-9 and f["outcome_tick_matches"]
            for f in findings
        )
    parity = all(
        f["integration_state_max_error"] < 1e-5 and f["outcome_tick_matches"]
        for f in findings
    )
    assert physical == result["accounted_physics_substeps"]
    expected_attempts = (
        80 + 64 * len(initial_hashes) + (32 if result["selection"]["qualified"] else 0)
    )
    assert attempts == expected_attempts
    return manifest() | {
        "source_run_commit": provenance["code_commit"],
        "protocol_hash": provenance["protocol_hash"],
        "accounting_passed": True,
        "evaluation_attempts": attempts,
        "recomputed_physics_substeps": physical,
        "planned_evaluation_attempts_under_selection": expected_attempts,
        "invalid_evaluation_attempts": 0,
        "findings": findings,
        "same_host_repeat_passed": same_host,
        "cross_host_reference_parity_passed": parity,
        "tolerances": {"same_host_repeat": 1e-9, "cross_host_full_state": 1e-5},
        "passed": same_host and parity,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.run)
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
