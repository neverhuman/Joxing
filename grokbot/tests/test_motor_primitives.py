"""Reach primitive and punch-capable policy (v0.0.13): the discovered reach
groups move the right/left hand forward relative to the trunk (mirror pair),
the punch only overrides its own groups inside its window, active extension
ignores whole-body motion, and the league runs with a frozen opponent."""

import json
import pathlib

import numpy as np
import pytest

from grokbot import selfplay
from grokbot.motor_primitives import (
    ARM_GROUPS,
    GROUPS,
    HAND,
    REACH_GROUPS,
    TRUNK,
    PunchPolicy,
    active_extension,
    punch_params,
    reach_effect,
)
from grokbot.scenarios import RangeBoxingEnv
from joxing.scene import build_scene

GEN4 = np.array(
    json.loads(
        pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json").read_text()
    )["archive"]["gen4"]
)


@pytest.fixture(scope="module")
def env():
    return RangeBoxingEnv(build_scene(), separation_m=0.7, horizon_ticks=400)


def test_reach_groups_are_a_mirror_pair_moving_each_hand_forward(env):
    right = reach_effect(env, REACH_GROUPS["r"][1:])
    left = reach_effect(env, REACH_GROUPS["l"][1:])
    assert right["r"][0] > 0.15 and abs(right["l"][0]) < 0.05
    assert left["l"][0] > 0.15 and abs(left["r"][0]) < 0.05
    assert right["r"][0] == pytest.approx(left["l"][0], abs=0.01)
    assert set(REACH_GROUPS["r"]) | set(REACH_GROUPS["l"]) <= set(ARM_GROUPS)


def test_a_leg_group_does_not_move_the_hands(env):
    leg = next(
        k for k, g in enumerate(GROUPS) if set(g) <= set(selfplay.REGIONS["leg_r"])
    )
    effect = reach_effect(env, [leg])
    assert all(abs(effect[s][0]) < 0.05 for s in "rl")


class Obs:
    def __init__(self, tick, values):
        self.sampled_tick, self.values = tick, values


def test_punch_overrides_only_its_groups_inside_its_window():
    values = np.zeros(1970, np.float32)
    params = punch_params(GEN4, onset=-1.1, duration=0.0, amplitude=3.0)
    policy = PunchPolicy(params)
    (onset, duration, amplitude) = policy.punches["r"]
    assert 0 < onset < 600 and 0 < duration < 600 and amplitude > 0.9
    posture = selfplay.LinearPolicy(GEN4)
    inside = int(onset) + 20 - int(onset) % 20
    reach = np.concatenate([GROUPS[k] for k in REACH_GROUPS["r"] + REACH_GROUPS["l"]])
    others = np.setdiff1d(np.arange(700), reach)
    for tick in (0, inside, int(onset + duration) + 40):
        cmd = policy(Obs(tick, values))
        ref = posture(Obs(tick, values))
        assert np.array_equal(cmd[others], ref[others])
        punching = onset <= tick < onset + duration
        assert np.all(cmd[reach] == np.float32(amplitude)) == punching
    with pytest.raises(ValueError):
        PunchPolicy(GEN4)


def test_active_extension_ignores_moving_the_whole_body():
    start = np.zeros(1970, np.float32)
    lean = start.copy()
    for i in (*HAND.values(), TRUNK):
        lean[i] += 0.3  # hand and trunk move together
    assert active_extension(start, lean) == pytest.approx(0.0)
    punch = start.copy()
    punch[HAND["r"]] += 0.25
    assert active_extension(start, punch) == pytest.approx(0.25)


def test_league_runs_a_punch_policy_against_a_frozen_champion():
    report = selfplay.run_league(
        generations=1,
        iterations=1,
        pairs=2,
        sigma=0.1,
        lr=0.05,
        eval_seeds=[0],
        workers=2,
        horizon_ticks=2000,
        seed=0,
        panel_eval=False,
        separation_m=0.7,
        anti_stalling=True,
        shaping_weight=0.5,
        shaping_key="extension",
        strike_bonus=0.5,
        init_params=punch_params(GEN4),
        policy_kind="punch",
        frozen={"v11gen4": ("params", GEN4)},
        require_valid=False,  # plumbing check; this configuration is degenerate
        log=lambda row: None,
    )
    (gen,) = report["generations"]
    assert gen["train_games"] == 4
    assert set(gen["vs_frozen"]) == {"v11gen4"}
    assert gen["vs_frozen"]["v11gen4"]["lower95"] <= 1.0
    assert {g["second"] for g in report["evaluation_games"]} == {"gen0", "v11gen4"}


def test_group_policy_from_gen4_matches_the_punch_policy_where_groups_fit_regions():
    from grokbot.motor_primitives import GROUP_REGION, GroupPolicy, group_params

    params = group_params(GEN4, onset=-3.0, duration=0.0, amplitude=2.0)
    group, punch = GroupPolicy(params), PunchPolicy(params[len(GROUPS) :])
    regions = list(selfplay.REGIONS.values())
    inside = np.concatenate(
        [
            g
            for g, r in zip(GROUPS, GROUP_REGION, strict=True)
            if np.isin(g, regions[r]).all()
        ]
    )
    assert len(inside) >= 600  # nearly every muscle's group sits in one region
    rng = np.random.default_rng(0)
    for tick in (0, 20, 400):
        values = rng.normal(size=1970).astype(np.float32)
        a, b = group(Obs(tick, values)), punch(Obs(tick, values))
        assert np.array_equal(a[inside], b[inside])
    with pytest.raises(ValueError):
        GroupPolicy(params[:-1])
    with pytest.raises(ValueError):
        GroupPolicy(params, noise=0.05)(Obs(0, values), None, None)


def test_group_feedback_policy_starts_exactly_at_the_group_policy():
    from grokbot.motor_primitives import (
        GROUP_FEEDBACK_PARAMS,
        GroupFeedbackPolicy,
        GroupPolicy,
        group_feedback_params,
        group_params,
    )

    group = group_params(GEN4, onset=-3.0, duration=0.0, amplitude=2.0)
    params = group_feedback_params(group)
    assert len(params) == len(group) + GROUP_FEEDBACK_PARAMS == 284 + 828
    a, b = GroupFeedbackPolicy(params, 0.05), GroupPolicy(group, 0.05)
    ra, rb = np.random.default_rng(3), np.random.default_rng(3)
    values = np.random.default_rng(0).normal(size=1970).astype(np.float32)
    for tick in (0, 20, 400):
        tilted = values.copy()
        tilted[selfplay.FEATURES[:6]] += 0.1 * tick / 400
        assert np.array_equal(
            a(Obs(tick, tilted), None, ra), b(Obs(tick, tilted), None, rb)
        )
    with pytest.raises(ValueError):
        GroupFeedbackPolicy(group)


def test_a_per_group_tilt_gain_moves_only_that_group():
    from grokbot.motor_primitives import (
        TILT_FEATURES,
        GroupFeedbackPolicy,
        group_feedback_params,
        group_params,
    )

    params = group_feedback_params(
        group_params(GEN4, onset=-3.0, duration=0.0, amplitude=2.0)
    )
    k = 5
    params[284 + k * TILT_FEATURES + 2] = -0.5  # group 5 on the third tilt feature
    zero = GroupFeedbackPolicy(group_feedback_params(params[:284]))
    gain = GroupFeedbackPolicy(params)
    values = np.random.default_rng(1).normal(size=1970).astype(np.float32)
    zero(Obs(0, values)), gain(Obs(0, values))
    tilted = values.copy()
    tilted[selfplay.FEATURES[2]] += 0.2
    diff = gain(Obs(20, tilted)) != zero(Obs(20, tilted))
    assert diff.any() and set(np.flatnonzero(diff)) <= set(GROUPS[k])
    assert (
        selfplay._controller(("group_feedback", params, 0.0)).group_gains[k, 2] == -0.5
    )


def test_group_pd_policy_starts_exactly_at_the_group_feedback_policy():
    from grokbot.motor_primitives import (
        GROUP_PD_PARAMS,
        RATE_FEATURES,
        GroupFeedbackPolicy,
        GroupPDPolicy,
        group_feedback_params,
        group_params,
        group_pd_params,
    )

    feedback = group_feedback_params(
        group_params(GEN4, onset=-3.0, duration=0.0, amplitude=2.0)
    )
    feedback[284:] = np.random.default_rng(2).normal(0, 0.1, 828)  # trained tilt
    params = group_pd_params(feedback)
    assert len(params) == 1112 + GROUP_PD_PARAMS == 1112 + 138 * RATE_FEATURES
    a, b = GroupPDPolicy(params, 0.05), GroupFeedbackPolicy(feedback, 0.05)
    ra, rb = np.random.default_rng(4), np.random.default_rng(4)
    values = np.random.default_rng(0).normal(size=1970).astype(np.float32)
    for tick in (0, 20, 400):
        moved = values.copy()
        moved[selfplay.FEATURES] += 0.05 * tick / 400
        assert np.array_equal(
            a(Obs(tick, moved), None, ra), b(Obs(tick, moved), None, rb)
        )
    k = 7
    params[1112 + k * RATE_FEATURES + 6] = 0.8  # group 7 on height
    gain, zero = GroupPDPolicy(params), GroupPDPolicy(group_pd_params(feedback))
    gain(Obs(0, values)), zero(Obs(0, values))
    lower = values.copy()
    lower[selfplay.FEATURES[12]] -= 0.1
    diff = gain(Obs(20, lower)) != zero(Obs(20, lower))
    assert diff.any() and set(np.flatnonzero(diff)) <= set(GROUPS[k])
    assert selfplay._controller(("group_pd", params, 0.0)).rate_gains[k, 6] == 0.8
    with pytest.raises(ValueError):
        GroupPDPolicy(feedback)
