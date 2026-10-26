"""Meaningful invariants for the controlled experiment and its decision rule."""

import copy
import json

import numpy as np
import pytest

from joxing.provenance import ROOT, digest
from joxing.stance_env import StanceEnv
from joxing.wave import (
    ARMS,
    ConstantPolicy,
    TrainingContract,
    bootstrap_pair,
    evaluate,
    select_arm,
    validate_protocol,
)


def test_protocol_is_strict_and_complete():
    protocol = json.loads((ROOT / "experiments/v0.0.5/protocol.json").read_text())
    assert validate_protocol(protocol) == protocol
    for key in ("arms", "seeds", "learner", "caps", "splits", "selection", "profile"):
        changed = copy.deepcopy(protocol)
        changed[key] = {}
        with pytest.raises(ValueError):
            validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["experiments"].append(changed["experiments"][0])
    with pytest.raises(ValueError):
        validate_protocol(changed)


def test_reward_wrapper_preserves_physics_and_terminal_state():
    raw = StanceEnv()
    scaled = TrainingContract(StanceEnv(), 1000, 149)
    raw.reset(options={"scenario_id": 62000})
    scaled.reset(options={"scenario_id": 62000})
    policy = ConstantPolicy(0.02)
    while not raw.done:
        action, _ = policy.predict(None)
        a = raw.step(action)
        b = scaled.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert b[1] == a[1] * 1000
        assert a[2:4] == b[2:4]
    assert digest(raw.snapshot()["state"]) == digest(
        scaled.unwrapped.snapshot()["state"]
    )
    assert raw.result() == scaled.unwrapped.result()


def test_training_resets_are_seeded_and_separate():
    a = TrainingContract(StanceEnv(), 1, 149)
    b = TrainingContract(StanceEnv(), 1000, 149)
    ids = []
    for _ in range(3):
        a.reset()
        b.reset()
        assert a.unwrapped.scenario_id == b.unwrapped.scenario_id
        ids.append(a.unwrapped.scenario_id)
    assert all(110000 <= s <= 110063 for s in ids)


def row(scenario, seconds):
    return {
        "scenario_id": scenario,
        "initial_state_hash": str(scenario),
        "status": "physical_failure",
        "invalid": False,
        "elapsed_sim_s": seconds,
    }


def test_pairs_reject_missing_duplicate_invalid_and_different_starts():
    before = [row(1, 1), row(2, 1)]
    after = [row(1, 1.5), row(2, 2)]
    assert bootstrap_pair(before, after)["paired_survival_gain_s"] == 0.75
    for bad in (
        [],
        [after[0]],
        [after[0], after[0]],
        [after[0], row(3, 2)],
        [after[0], after[1] | {"invalid": True}],
        [after[0], after[1] | {"initial_state_hash": "wrong"}],
    ):
        with pytest.raises(ValueError):
            bootstrap_pair(before, bad)


def test_selection_is_complete_thresholded_and_stable_on_ties():
    def cell(mean=1.0, gain=0.0):
        return {
            "scheduled_training_complete": True,
            "summary": {
                "invalid": 0,
                "trained_mean_survival_s": mean,
                "paired_survival_gain_s": gain,
            },
        }

    results = {a: cell() for a in ARMS}
    assert select_arm(results) == {
        "qualified": False,
        "arm": "A",
        "gain_s": 0.0,
        "over_A_s": 0.0,
        "reason": "no_meaningful_screen_gain",
    }
    results["B"] = cell(1.4, 0.4)
    assert select_arm(results)["qualified"]
    assert select_arm(results)["arm"] == "B"
    results["C"] = cell(1.4, 0.4)
    assert select_arm(results)["arm"] == "B"
    results["D"]["scheduled_training_complete"] = False
    assert not select_arm(results)["qualified"]
    assert not select_arm({})["qualified"]


def test_evaluation_fault_keeps_all_planned_attempts(tmp_path):
    class BrokenPolicy:
        def predict(self, obs, deterministic=True):
            raise RuntimeError("injected policy failure")

    with pytest.raises(RuntimeError, match="injected policy failure"):
        evaluate(BrokenPolicy(), StanceEnv(), [62010, 62011], "fault", tmp_path)
    rows = json.loads((tmp_path / "fault-evaluations.json").read_text())
    assert [r["status"] for r in rows] == ["invalid", "not_run"]
    receipt = json.loads((tmp_path / "fault-fault.json").read_text())
    assert receipt["last_valid"]["application"]["tick"] == 0
    with pytest.raises(FileExistsError):
        evaluate(BrokenPolicy(), StanceEnv(), [62010, 62011], "fault", tmp_path)
