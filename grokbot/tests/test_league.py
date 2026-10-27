"""Known-payoff fixtures for the grokbot evaluation oracle.

These games have exact answers, so the oracle can be trusted before any boxing
policy is rated with it: transitive ladders, rock-paper-scissors cycles, a
latest-only cycling learner, invalid-game denominators and side bias.
"""

import itertools

import numpy as np
import pytest

from grokbot.league import ELO, Game, Ledger, exploitability, solve_zero_sum

RPS = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)


def play(ledger, first, second, outcome, *, swap=True, invalid=None):
    """One side-swapped pair: first plays corner A, then corner B."""
    ledger.add(Game(first, second, "A", None if invalid else outcome, invalid))
    if swap:
        ledger.add(Game(first, second, "B", None if invalid else outcome, invalid))


def test_game_rejects_malformed_results():
    with pytest.raises(ValueError):
        Game("x", "x", "A", 1)
    with pytest.raises(ValueError):
        Game("x", "y", "C", 1)
    with pytest.raises(ValueError):
        Game("x", "y", "A", 2)
    with pytest.raises(ValueError):
        Game("x", "y", "A", 1, "nan contact force")  # invalid cannot carry a result
    with pytest.raises(ValueError):
        Game("x", "y", "A", None)  # valid needs a result


def test_invalid_games_stay_in_the_denominator_but_never_score():
    ledger = Ledger(["p", "q"])
    play(ledger, "p", "q", 1)
    play(ledger, "p", "q", None, invalid="FloatingPointError: Nonfinite activation")
    counts = ledger.counts()
    assert counts["attempted"][0, 1] == 4 and counts["attempted"][1, 0] == 4
    assert counts["invalid"][0, 1] == 2
    assert counts["win"][0, 1] == 2 and counts["loss"][1, 0] == 2
    matrix, n = ledger.payoff()
    assert matrix[0, 1] == 1.0 and n[0, 1] == 2


def test_payoff_is_antisymmetric_and_unplayed_pairs_are_unknown():
    ledger = Ledger(["a", "b", "c"])
    play(ledger, "a", "b", 1)
    play(ledger, "b", "a", 0)
    matrix, count = ledger.payoff()
    assert matrix[0, 1] == pytest.approx(0.5) and matrix[1, 0] == pytest.approx(-0.5)
    assert np.isnan(matrix[0, 2]) and count[0, 2] == 0


def test_side_balance_detects_corner_advantage():
    fair, biased = Ledger(["a", "b"]), Ledger(["a", "b"])
    play(fair, "a", "b", 1)  # a wins from both corners: skill, not side
    biased.add(Game("a", "b", "A", 1))
    biased.add(Game("a", "b", "B", -1))  # whoever has corner A wins
    assert fair.side_balance() == 0.0
    assert biased.side_balance() == 1.0


def test_bradley_terry_recovers_a_transitive_ladder():
    rng = np.random.default_rng(3)
    truth = np.array([0.0, 150.0, 300.0, 450.0])
    names = [f"v{k}" for k in range(4)]
    ledger = Ledger(names)
    for i, j in itertools.combinations(range(4), 2):
        p = 1 / (1 + 10 ** ((truth[j] - truth[i]) / 400))
        for _ in range(400):
            ledger.add(Game(names[i], names[j], "A", 1 if rng.random() < p else -1))
    rating = ledger.ratings(anchor="v0")
    assert np.all(np.diff(rating) > 0)
    assert rating == pytest.approx(truth, abs=40)
    low, high, used = ledger.bootstrap_ratings(samples=60, seed=1, anchor="v0")
    assert used == 60
    assert np.all(low <= rating + 1e-9) and np.all(rating <= high + 1e-9)
    assert np.all(low[1:] > 0)  # every later version is credibly above v0


def test_unrated_players_are_nan_and_disconnected_scales_are_rejected():
    ledger = Ledger(["a", "b", "new"])
    play(ledger, "a", "b", 1)
    play(ledger, "a", "b", -1)
    rating = ledger.ratings()
    assert np.isnan(rating[2]) and rating[0] == pytest.approx(rating[1])
    low, high, _ = ledger.bootstrap_ratings(samples=10)
    assert np.isnan(low[2]) and np.isnan(high[2])
    islands = Ledger(["a", "b", "c", "d"])
    play(islands, "a", "b", 1)
    play(islands, "c", "d", 1)
    with pytest.raises(ValueError, match="Disconnected"):
        islands.ratings()


def test_elo_scale_matches_expected_score():
    ledger = Ledger(["a", "b"])
    for _ in range(76):
        ledger.add(Game("a", "b", "A", 1))
    for _ in range(24):
        ledger.add(Game("a", "b", "A", -1))
    rating = ledger.ratings(prior_games=0)
    assert rating.mean() == pytest.approx(0.0, abs=1e-9)  # unanchored: centred
    gap = rating[0] - rating[1]
    assert 1 / (1 + 10 ** (-gap / 400)) == pytest.approx(0.76, abs=1e-6)
    assert ELO == pytest.approx(400 / np.log(10))


@pytest.mark.parametrize(
    "matrix, value, row",
    [
        (RPS, 0.0, [1 / 3, 1 / 3, 1 / 3]),
        (np.array([[1.0, -1.0], [-1.0, 1.0]]), 0.0, [0.5, 0.5]),
        (np.array([[3.0, 1.0], [2.0, 4.0]]), 2.5, [0.5, 0.5]),
        (np.array([[0.0, 1.0], [-1.0, 0.0]]), 0.0, [1.0, 0.0]),
    ],
)
def test_solve_zero_sum_known_equilibria(matrix, value, row):
    p, q, v = solve_zero_sum(matrix)
    assert v == pytest.approx(value, abs=1e-9)
    assert p == pytest.approx(row, abs=1e-9)
    assert (p @ matrix).min() == pytest.approx(v, abs=1e-9)
    assert (matrix @ q).max() == pytest.approx(v, abs=1e-9)


def test_random_games_certify_minimax():
    rng = np.random.default_rng(11)
    for _ in range(25):
        m = rng.normal(size=(rng.integers(2, 9), rng.integers(2, 9)))
        p, q, v = solve_zero_sum(m)
        assert (p @ m).min() >= v - 1e-9 and (m @ q).max() <= v + 1e-9


def test_dominant_policy_is_unexploitable_and_dominated_one_is_not():
    # Transitive: 0 beats 1 beats 2, and 0 beats 2. Not sign-symmetric like RPS.
    ladder = np.array([[0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, -1.0, 0.0]])
    assert exploitability(ladder, [1, 0, 0]) == 0.0
    assert exploitability(ladder, [0, 0, 1]) == 1.0
    p, _, value = solve_zero_sum(ladder)
    assert p == pytest.approx([1, 0, 0]) and value == pytest.approx(0.0)


def test_rps_cycle_is_exploitable_unless_mixed():
    assert exploitability(RPS, [1, 0, 0]) == 1.0
    assert exploitability(RPS, [0.5, 0.5, 0]) == 0.5
    assert exploitability(RPS, [1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0, abs=1e-12)


def test_latest_only_cycling_learner_looks_strong_but_is_exploitable():
    """Each snapshot beats its predecessor; the latest is still fully exploitable."""
    names = ["rock", "paper", "scissors", "rock2"]
    beats = {
        ("paper", "rock"),
        ("scissors", "paper"),
        ("rock2", "scissors"),
        ("rock", "scissors"),
        ("paper", "rock2"),
    }
    ledger = Ledger(names)
    for a, b in itertools.combinations(names, 2):
        outcome = 1 if (a, b) in beats else -1 if (b, a) in beats else 0
        play(ledger, a, b, outcome)
    matrix, _ = ledger.payoff()
    # Latest-vs-previous win rate is perfect along the whole training history...
    for prev, latest in zip(names, names[1:]):
        assert matrix[names.index(latest), names.index(prev)] == 1.0
    # ...yet the newest snapshot is exploited by an archived one.
    latest = np.eye(4)[3]
    assert exploitability(matrix, latest) == 1.0
    p, _, value = solve_zero_sum(matrix)
    assert value == pytest.approx(0.0, abs=1e-9)
    assert exploitability(matrix, p) == pytest.approx(0.0, abs=1e-9)


def test_exploitability_rejects_non_probability_or_general_sum_inputs():
    with pytest.raises(ValueError):
        exploitability(RPS, [0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        exploitability(np.array([[1.0, 0.0], [0.0, 1.0]]), [0.5, 0.5])
