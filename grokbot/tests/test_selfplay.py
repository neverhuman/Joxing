"""Self-play learner: policy plumbing, ES rank shaping, and the CI smoke test
that one fixed-seed generation beats its random initialization in
side-swapped, common-random-number evaluation."""

import numpy as np
import pytest

from grokbot import selfplay
from grokbot.selfplay import (
    FEATURES,
    N_PARAMS,
    REGIONS,
    LinearPolicy,
    centred_ranks,
    random_params,
)
from joxing.observations import FEATURE_SPEC


def test_regions_drive_every_muscle_exactly_once():
    indices = np.concatenate(list(REGIONS.values()))
    assert sorted(indices.tolist()) == list(range(700))
    assert N_PARAMS == len(REGIONS) * (1 + len(FEATURES))


def test_policy_reads_only_the_actors_own_features():
    own = {"own_body_rotation_ego", "own_body_linear_velocity_ego"}
    own |= {"own_body_angular_velocity_ego", "own_pelvis_height"}
    allowed = [b for b in FEATURE_SPEC if b["name"] in own]
    for index in FEATURES:
        assert any(b["start"] <= index < b["stop"] for b in allowed), index


class Obs:
    def __init__(self, tick, values):
        self.sampled_tick, self.values = tick, values


def test_policy_is_bounded_and_rezeroes_each_episode():
    rng = np.random.default_rng(0)
    params = random_params(rng)
    table = params.reshape(len(REGIONS), -1)
    table[:, 1:] = 50.0  # extreme feedback gains must still give valid commands
    policy = LinearPolicy(table.ravel())
    first = rng.normal(size=1970).astype(np.float32)
    command = policy(Obs(0, first))
    assert command.dtype == np.float32 and command.shape == (700,)
    for region, value in zip(REGIONS.values(), table[:, 0], strict=True):
        assert np.all(command[region] == np.float32(value))  # origin: tone only
    later = policy(Obs(20, first + 1.0))
    assert np.all(np.abs(later) <= 1) and not np.array_equal(later, command)
    again = policy(Obs(0, first + 5.0))  # new episode: new origin
    assert np.array_equal(again, command)
    with pytest.raises(ValueError):
        LinearPolicy(np.zeros(N_PARAMS - 1))
    with pytest.raises(ValueError):
        LinearPolicy(np.full(N_PARAMS, np.nan))


def test_evaluation_noise_makes_seeds_independent_and_is_seeded():
    params = random_params(np.random.default_rng(1))
    obs = Obs(0, np.zeros(1970, np.float32))
    plain = LinearPolicy(params)
    assert np.array_equal(plain(obs, None, None), plain(obs, None, None))
    noisy = LinearPolicy(params, noise=0.05)
    a = noisy(obs, None, np.random.default_rng([3, 1]))
    b = noisy(obs, None, np.random.default_rng([3, 1]))
    c = noisy(obs, None, np.random.default_rng([4, 1]))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    with pytest.raises(ValueError):
        noisy(obs, None, None)


def test_centred_ranks_give_no_update_on_flat_fitness():
    assert np.all(centred_ranks([1, 1, 1, 1]) == 0)
    ranks = centred_ranks([-1, 0, 1, 1])
    assert ranks[0] == -0.5 and ranks[2] == ranks[3] and ranks.sum() == pytest.approx(0)


def test_rejects_oversubscribing_shared_hosts():
    with pytest.raises(ValueError):
        selfplay.run_league(
            generations=1,
            iterations=1,
            pairs=1,
            sigma=0.1,
            lr=0.1,
            eval_seeds=[0],
            workers=5,
            horizon_ticks=200,
            seed=0,
        )


def test_one_generation_beats_its_random_initialization():
    """CI smoke test: fixed seed, 24 training games, 10 independent noisy
    evaluation pairs. Calibrated on seeds 0-2: the real update scores +1.0 on
    all three; at this seed lr=0 scores -0.1 and a negated lr -1.0, so both
    non-learning mutants fail the 0.8 threshold."""
    report = selfplay.run_league(
        generations=1,
        iterations=3,
        pairs=4,
        sigma=0.1,
        lr=0.05,
        eval_seeds=list(range(10)),
        workers=2,
        horizon_ticks=2000,
        seed=1,
        panel_eval=False,
        log=lambda row: None,
    )
    (gen,) = report["generations"]
    assert gen["train_games"] == 24 and gen["train_blocked"] == 0
    assert gen["vs_previous"] >= 0.8  # gen1 beats gen0 on paired evaluation
    assert gen["beats_all_earlier"]
    assert gen["side_balance"] == 0.0  # not a corner artifact
    assert gen["eval_valid_share"] > 0  # a non-degenerate league
    games = report["evaluation_games"]
    assert len(games) == 20 and {g["first_side"] for g in games} == {"A", "B"}
