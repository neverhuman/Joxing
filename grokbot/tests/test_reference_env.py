"""Incremental bout monitor equals the batch reference, and the reference-ending
env gives self-play a zero-sum per-step reward that telescopes to the ending."""

import numpy as np
import pytest

from grokbot.endings import BoutMonitor, classify_contacts, support_losses
from grokbot.reference_env import ReferenceEndingEnv
from grokbot.tests.test_endings import FAST, REGISTRY, rows
from joxing.env import BoxingEnv
from joxing.scene import build_scene


@pytest.fixture(scope="module")
def scene():
    return build_scene()


def synthetic(seed):
    rng = np.random.default_rng(seed)
    starts = {name: int(rng.integers(50, 400)) for name in ("strike", "body", "fall")}
    return rows(
        600,
        lambda t: [(5, 6, 300.0, FAST)]
        if starts["strike"] <= t < starts["strike"] + 5
        else [],
        lambda t: [(1, 3, 200.0)] if seed % 3 == 0 and t == starts["body"] else [],
        lambda t: [(3, 0, 500.0)] if t >= starts["fall"] else [],
        lambda t: [(0, 1, 400.0)] if seed % 2 and t >= starts["fall"] + 30 else [],
    )


@pytest.mark.parametrize("seed", range(12))
def test_chunked_monitor_equals_batch(seed):
    evidence = synthetic(seed)
    monitor = BoutMonitor(REGISTRY)
    cuts = sorted(
        np.random.default_rng(seed).choice(np.arange(1, 600), 7, replace=False)
    )
    for lo, hi in zip([0, *cuts], [*cuts, 600]):
        monitor.update(evidence[lo:hi])
    assert monitor.losses == support_losses(evidence, REGISTRY)
    strikes, blocking = classify_contacts(evidence, REGISTRY)
    assert (monitor.contacts.strikes, monitor.contacts.blocking) == (strikes, blocking)
    assert monitor.losses  # every synthetic bout has at least one fall


def test_monitor_rejects_gaps_and_reordering():
    evidence = synthetic(1)
    monitor = BoutMonitor(REGISTRY)
    monitor.update(evidence[:10])
    with pytest.raises(ValueError):
        monitor.update(evidence[11:12])


def play(env, a, b, seed=0):
    commands = {"A": np.full(700, a, np.float32), "B": np.full(700, b, np.float32)}
    env.reset(seed=seed)
    totals, steps = {"A": 0.0, "B": 0.0}, 0
    while True:
        result = env.step(commands)
        steps += 1
        assert result["rewards"]["A"] == -result["rewards"]["B"]
        for p in totals:
            totals[p] += result["rewards"][p]
        if result["terminated"]:
            return result, totals, steps


def test_first_self_caused_fall_ends_the_bout_and_pays_the_winner(scene):
    env = ReferenceEndingEnv(BoxingEnv(scene, horizon_ticks=3000))
    result, totals, steps = play(env, -1.0, 1.0)  # relaxed A vs co-contracting B
    ending = result["info"]["ending"]
    assert ending["utility"] == (1, -1) and ending["reason"] == "B self-caused fall"
    assert totals == pytest.approx({"A": 1.0, "B": -1.0})
    assert env.env.tick < 3000 and steps == env.env.tick // 20
    with pytest.raises(RuntimeError):
        env.step({p: np.zeros(700, np.float32) for p in ("A", "B")})


def test_mirror_bout_draws_and_the_bell_decides_short_bouts(scene):
    env = ReferenceEndingEnv(BoxingEnv(scene, horizon_ticks=3000))
    result, totals, _ = play(env, 0.0, 0.0)
    assert result["info"]["ending"]["utility"] == (0, 0)
    assert result["info"]["ending"]["reason"].startswith("draw: simultaneous")
    short = ReferenceEndingEnv(BoxingEnv(scene, horizon_ticks=200))
    result, totals, steps = play(short, -1.0, 1.0)
    assert result["info"]["ending"]["reason"] == "bell: points" and steps == 10
    assert totals == pytest.approx({"A": 0.0, "B": 0.0})


def test_blocked_bout_terminates_with_zero_return(scene):
    env = ReferenceEndingEnv(
        BoxingEnv(
            scene, horizon_ticks=2000, scenario="close-overlap-collision-probe-v1"
        )
    )
    result, totals, _ = play(env, -1.0, 0.0)
    assert result["info"]["blocked"] and result["info"]["ending"]["utility"] is None
    assert totals == pytest.approx({"A": 0.0, "B": 0.0})
