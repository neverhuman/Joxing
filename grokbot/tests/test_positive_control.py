"""P4 positive control: on games with known answers, latest-only self-play
cycles and stays exploitable, while the league's Nash meta-solver reaches the
equilibrium, including when payoffs are estimated from sampled games."""

import numpy as np
import pytest

from grokbot.positive_control import (
    RPS,
    best_response,
    cyclic_game,
    exploiter_proxy,
    exploiter_recall,
    random_symmetric_game,
    run,
)

GAMES = [random_symmetric_game(20, seed) for seed in range(6)]


def exploitabilities(history):
    return [row["exploitability"] for row in history]


def test_latest_only_self_play_cycles_on_rps():
    history = run(RPS, "latest", 30)
    assert [row["added"] for row in history[:6]] == [1, 2, 0, 1, 2, 0]
    assert exploitabilities(history) == [1.0] * 30


def test_nash_league_solves_rps_once_all_strategies_exist():
    history = run(RPS, "nash", 10)
    assert exploitabilities(history)[:2] == [1.0, 1.0]
    assert exploitabilities(history)[2:] == pytest.approx([0.0] * 8, abs=1e-12)
    assert history[-1]["meta"] == pytest.approx([1 / 3] * 3)


def test_fictitious_play_converges_slowly_on_rps():
    values = exploitabilities(run(RPS, "uniform", 300))
    assert values[0] == 1.0 and values[-1] < 0.05


@pytest.mark.parametrize("game", GAMES)
def test_games_have_no_pure_equilibrium(game):
    # Otherwise latest-only self-play could legitimately settle on it.
    assert not any(np.all(game[i] >= 0) for i in range(len(game)))


@pytest.mark.parametrize("game", GAMES)
def test_nash_league_reaches_equilibrium_and_latest_only_never_does(game):
    nash = exploitabilities(run(game, "nash", 20))
    solved = next(i for i, value in enumerate(nash) if value < 1e-9)
    assert all(value < 1e-9 for value in nash[solved:])
    assert min(exploitabilities(run(game, "latest", 20))) > 0.5
    assert exploitabilities(run(game, "uniform", 200))[-1] < 0.1


@pytest.mark.parametrize("game", GAMES)
def test_sampled_evaluation_keeps_the_nash_league_near_equilibrium(game):
    history = run(game, "nash", 25, games_per_pair=200, seed=1)
    assert history[-1]["exploitability"] < 0.1
    assert history[0]["exploitability"] > 0.5


def test_best_response_and_validation():
    assert best_response(RPS, np.array([1.0, 0.0, 0.0])) == 1  # paper beats rock
    assert best_response(RPS, np.full(3, 1 / 3)) == 0  # ties: lowest index
    with pytest.raises(ValueError):
        run(RPS, "elo", 3)
    with pytest.raises(ValueError):
        run(np.eye(3), "nash", 3)


def test_cyclic_game_needs_every_strategy_and_latest_only_never_gets_there():
    game = cyclic_game(20)
    assert exploitabilities(run(game, "latest", 60)) == [1.0] * 60
    history = run(game, "nash", 25)
    solved = next(row for row in history if row["exploitability"] < 1e-9)
    # Unsolved until the population holds all 20 strategies, then solved for good.
    assert len(solved["members"]) == 20 and solved["iteration"] == 19
    assert max(exploitabilities(history)[19:]) < 1e-9


@pytest.mark.parametrize("game", [cyclic_game(20), GAMES[0]])
def test_exploiter_proxy_never_overstates_and_full_budget_is_exact(game):
    rng = np.random.default_rng(0)
    for row in run(game, "latest", 30):
        proxy = exploiter_proxy(game, row["meta"], 5, rng)
        assert 0.0 <= proxy <= row["exploitability"] + 1e-12
        full = exploiter_proxy(game, row["meta"], len(game), rng)
        assert full == pytest.approx(max(0.0, row["exploitability"]))


def test_weak_exploiter_calls_exploitable_snapshots_safe_in_counter_sparse_games():
    snapshots = [row["meta"] for row in run(cyclic_game(20), "latest", 60)]
    recall, missed = exploiter_recall(
        cyclic_game(20), snapshots, 5, np.random.default_rng(0)
    )
    # Every snapshot is fully exploitable, yet most are reported as safe.
    assert recall < 0.4 and missed > 0.6
    rich = [row["meta"] for row in run(GAMES[0], "latest", 60)]
    recall_rich, _ = exploiter_recall(GAMES[0], rich, 5, np.random.default_rng(0))
    # Counter-rich games forgive the same budget: recall depends on structure.
    assert recall_rich > 0.6


def test_exploiter_reports_no_exploit_rather_than_a_negative_one():
    dominant = np.array([[0.0, 1.0], [-1.0, 0.0]])
    rng = np.random.default_rng(3)
    reports = {exploiter_proxy(dominant, [1.0, 0.0], 1, rng) for _ in range(20)}
    assert reports == {0.0}  # trying only the dominated strategy finds nothing
