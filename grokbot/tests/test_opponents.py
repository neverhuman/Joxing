"""Opponent sampling: the P4 edge cases (K = 0, all-zero weights, 0/1 win
rates, unseen opponents, malformed stats, explicit priors, per-game RNG) and the
eps/K floor that keeps every archived snapshot in play."""

import math

import numpy as np
import pytest

from grokbot.opponents import history_probabilities, mixture, sample


def total(probs):
    return sum(probs.values())


def test_hard_pfsp_favours_opponents_the_learner_loses_to():
    probs = history_probabilities(
        {"easy": (90, 100), "even": (50, 100), "hard": (5, 100)}
    )
    assert total(probs) == pytest.approx(1.0)
    assert probs["hard"] > probs["even"] > probs["easy"]
    assert min(probs.values()) >= 0.1 / 3 - 1e-12  # eps/K floor


def test_variance_pfsp_favours_even_matchups():
    probs = history_probabilities(
        {"easy": (95, 100), "even": (50, 100), "hard": (5, 100)}, weighting="variance"
    )
    assert probs["even"] > probs["easy"] and probs["even"] > probs["hard"]
    assert probs["easy"] == pytest.approx(probs["hard"])


def test_unseen_opponents_use_the_prior_and_extremes_stay_finite():
    probs = history_probabilities({"new": (0, 0), "beaten": (10, 10), "lost": (0, 10)})
    assert all(math.isfinite(v) and v > 0 for v in probs.values())
    # Prior 1 win in 2 games: unseen sits at 0.5, between the two extremes.
    assert probs["lost"] > probs["new"] > probs["beaten"]


def test_floor_keeps_a_perfectly_beaten_snapshot_in_play():
    probs = history_probabilities(
        {"beaten": (1000, 1000), "rival": (0, 1000)}, prior_games=0, prior_wins=0
    )
    assert probs["beaten"] == pytest.approx(0.1 / 2)  # weight 0, floor only


def test_all_zero_weights_fall_back_to_uniform():
    probs = history_probabilities(
        {"a": (10, 10), "b": (3, 3)}, prior_games=0, prior_wins=0
    )
    assert probs == pytest.approx({"a": 0.5, "b": 0.5})


def test_no_archive_and_malformed_records():
    assert history_probabilities({}) == {}
    for bad in [(-1, 5), (6, 5), (float("nan"), 5), (1, float("inf"))]:
        with pytest.raises(ValueError):
            history_probabilities({"x": bad})
    with pytest.raises(ValueError):
        history_probabilities({"x": (1, 2)}, eps=1.5)
    with pytest.raises(ValueError):
        history_probabilities({"x": (1, 2)}, prior_wins=3, prior_games=2)
    with pytest.raises(ValueError):
        history_probabilities({"x": (1, 2)}, weighting="elo")


def test_bucket_shares_and_empty_buckets():
    history = history_probabilities({"h1": (1, 4), "h2": (3, 4)})
    full = mixture(latest=["now"], history=history, anchors=["scripted"])
    assert total(full) == pytest.approx(1.0)
    assert full["now"] == pytest.approx(0.5) and full["scripted"] == pytest.approx(0.15)
    assert full["h1"] + full["h2"] == pytest.approx(0.35)
    # No archive yet: history's share is redistributed, not lost.
    early = mixture(latest=["now"], anchors=["scripted"])
    assert early == pytest.approx({"now": 0.5 / 0.65, "scripted": 0.15 / 0.65})
    with pytest.raises(ValueError):
        mixture()


def test_a_policy_in_several_buckets_sums_its_probability():
    probs = mixture(latest=["v3"], history={"v3": 0.5, "v1": 0.5}, anchors=["v1"])
    assert probs == pytest.approx({"v3": 0.5 + 0.175, "v1": 0.175 + 0.15})


def test_mixture_rejects_bad_inputs():
    with pytest.raises(ValueError):
        mixture(latest=["a"], history={"b": 0.7})
    with pytest.raises(ValueError):
        mixture(latest=["a"], shares={"latest": -1.0, "history": 1.0, "anchors": 1.0})


def test_sampling_is_per_game_reproducible_and_matches_probabilities():
    probs = mixture(
        latest=["now"],
        history=history_probabilities({"h1": (1, 10), "h2": (9, 10)}),
        anchors=["scripted"],
    )
    first = [sample(probs, np.random.default_rng([7, game])) for game in range(50)]
    again = [sample(probs, np.random.default_rng([7, game])) for game in range(50)]
    assert first == again
    rng = np.random.default_rng(0)
    draws = [sample(probs, rng) for _ in range(20000)]
    for name, p in probs.items():
        assert draws.count(name) / len(draws) == pytest.approx(p, abs=0.015)
    with pytest.raises(ValueError):
        sample({}, rng)
