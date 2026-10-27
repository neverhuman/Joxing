"""P4 positive control: on games with known answers, latest-only self-play
cycles and stays exploitable, while the league's Nash meta-solver reaches the
equilibrium, including when payoffs are estimated from sampled games."""

import numpy as np
import pytest

from grokbot.positive_control import RPS, best_response, random_symmetric_game, run

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
