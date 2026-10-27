"""Match-runner invariants: every attempt is recorded, faults are attributed,
side-swapped pairs mirror each other, and nothing is silently scored."""

import numpy as np
import pytest

from grokbot.arena import SCRIPTED_PANEL, constant, corner_result, paired, play
from joxing.env import BoxingEnv
from joxing.scene import build_scene


@pytest.fixture(scope="module")
def scene():
    return build_scene()


def nan_command(observation, history, rng):
    return np.full(700, np.nan, np.float32)


def float64_command(observation, history, rng):
    return np.zeros(700)


def test_corner_result_reports_from_the_first_policys_view():
    assert corner_result((3, 1), "A") == ([3, 1], 1)
    assert corner_result((3, 1), "B") == ([1, 3], -1)
    assert corner_result((2, 2), "B") == ([2, 2], 0)


def test_identical_controllers_draw_scoreless_from_both_corners(scene):
    env = BoxingEnv(scene, horizon_ticks=200)
    panel = {"x": constant(0.0), "x_copy": constant(0.0)}
    records = paired(env, panel, "x", "x_copy", seeds=[0])
    assert [r["first_side"] for r in records] == ["A", "B"]
    for r in records:
        assert r["invalid_reason"] is None and r["ticks"] == 200
        assert r["outcome"] == 0 and r["points"] == [0, 0] and r["events"] == 0


def test_side_swapped_pair_mirrors_fall_times(scene):
    env = BoxingEnv(scene, horizon_ticks=2000)
    as_a, as_b = paired(env, SCRIPTED_PANEL, "relaxed", "cocontract", seeds=[0])
    assert as_a["fall_tick"] == as_b["fall_tick"]
    # Both untrained athletes fall well inside two seconds; nothing is scored.
    assert all(t is not None and t < 2000 for t in as_a["fall_tick"].values())
    assert as_a["outcome"] == as_b["outcome"] == 0


@pytest.mark.parametrize(
    "bad, reason",
    [(nan_command, "nonfinite"), (float64_command, "dtype float64")],
)
def test_controller_fault_is_charged_to_its_policy_and_env_recovers(scene, bad, reason):
    env = BoxingEnv(scene, horizon_ticks=200)
    panel = {"bad": bad, "ok": constant(0.0)}
    as_b = play(env, panel, "ok", "bad", "A", seed=1)
    assert as_b["fault_player"] == "bad" and reason in as_b["invalid_reason"]
    assert "B:" in as_b["invalid_reason"]  # "bad" played corner B
    assert as_b["outcome"] is None and as_b["points"] is None and as_b["ticks"] == 0
    follow_up = play(env, panel, "ok", "bad", "A", seed=1)
    assert follow_up == as_b  # deterministic, and the env is reusable after a fault
    clean = play(env, {"ok": constant(0.0), "ok2": constant(0.0)}, "ok", "ok2", "A", 1)
    assert clean["invalid_reason"] is None and clean["outcome"] == 0


def test_external_assistance_quarantines_instead_of_scoring(scene):
    env = BoxingEnv(scene, horizon_ticks=200)

    def assisted(observation, history, rng):
        env.data.xfrc_applied[1, 2] = 50.0  # an outside hand on the first body
        return np.zeros(700, np.float32)

    record = play(
        env, {"assisted": assisted, "ok": constant(0.0)}, "assisted", "ok", "A", 0
    )
    assert "External assistance forbidden" in record["invalid_reason"]
    assert record["fault_player"] is None  # engine guard, not a command fault
    assert record["outcome"] is None and record["points"] is None


def test_same_seed_reproduces_the_record(scene):
    env = BoxingEnv(scene, horizon_ticks=200)
    first = play(env, SCRIPTED_PANEL, "noise", "baseline", "B", seed=5)
    again = play(env, SCRIPTED_PANEL, "noise", "baseline", "B", seed=5)
    assert first == again


def test_rejects_self_play_by_name_and_unknown_corners(scene):
    env = BoxingEnv(scene, horizon_ticks=200)
    with pytest.raises(ValueError):
        play(env, SCRIPTED_PANEL, "noise", "noise", "A", 0)
    with pytest.raises(ValueError):
        play(env, SCRIPTED_PANEL, "noise", "baseline", "C", 0)
