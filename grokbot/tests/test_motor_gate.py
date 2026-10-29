"""Solo motor gate: real short trials, continuous fitness, fault quarantine in
the ES update, and pass rates that count faults as failures."""

import numpy as np
import pytest

from grokbot import motor_gate
from grokbot.selfplay import N_PARAMS, centred_ranks


class FakePool:
    def __init__(self, results):
        self.results = results

    def map(self, function, tasks):
        assert len(tasks) == len(self.results)
        return self.results


def result(
    stand,
    strikes=0,
    invalid=None,
    scoring=0,
    first_hit=None,
    opp_fall=None,
    opp_attribution=None,
):
    return {
        "invalid": invalid,
        "stand_ticks": stand,
        "strike_ticks": strikes,
        "scoring_strike_ticks": scoring,
        "first_scoring_tick": first_hit,
        "opponent_fall_tick": opp_fall,
        "opponent_fall_attribution": opp_attribution,
        "ticks": 3000,
    }


def test_a_short_real_trial_stands_to_the_horizon(monkeypatch):
    motor_gate._init(0.5, 400)
    out = motor_gate.trial((("scripted", "baseline"), ("scripted", "baseline"), 0))
    assert out["invalid"] is None and out["ticks"] == 400
    assert out["stand_ticks"] == 400 and out["strike_ticks"] == 0
    assert out["scoring_strike_ticks"] == 0 and out["first_scoring_tick"] is None
    assert out["opponent_fall_attribution"] is None and out["contact_ticks"] == 0


def test_fitness_is_stand_fraction_plus_strike_indicator():
    assert motor_gate.fitness(result(1500), 3000, 1.0) == 0.5
    assert motor_gate.fitness(result(3000, strikes=4), 3000, 1.0) == 2.0
    assert motor_gate.fitness(result(0, invalid="RuntimeError"), 3000, 1.0) is None
    # v0.0.18: only a head/torso hit earns the bonus; a blocked punch does not.
    blocked_punch, hit = result(3000, strikes=50), result(3000, strikes=50, scoring=9)
    assert motor_gate.fitness(blocked_punch, 3000, 1.0, "scoring_strike_ticks") == 1.0
    assert motor_gate.fitness(hit, 3000, 1.0, "scoring_strike_ticks") == 2.0

    # v0.0.19: the hit counts only if the athlete is still up 500 ticks later.
    def stay(stand, first_hit):
        r = result(stand, strikes=50, scoring=9, first_hit=first_hit)
        return motor_gate.fitness(r, 3000, 1.0, stay_up_ticks=500)

    assert stay(1500, 600) == 1.5  # hit at 600, still up at 1100
    assert stay(1100, 600) == 1100 / 3000 + 1.0  # exactly 500 ticks later
    assert stay(1099, 600) == 1099 / 3000  # fell one tick early: no bonus
    assert stay(3000, None) == 1.0  # stood, never hit
    assert stay(3000, 2600) == 1.0  # hit too late to show it stayed up

    # v0.0.22: the bonus needs a hit and then the opponent falling first.
    def knock(stand, first_hit, opp_fall):
        r = result(stand, 50, scoring=9, first_hit=first_hit, opp_fall=opp_fall)
        return motor_gate.fitness(r, 3000, 1.0, knockdown=True)

    assert knock(3000, 600, 1500) == 2.0  # hit, opponent down, still standing
    assert knock(2000, 600, 1500) == 2000 / 3000 + 1.0  # down first, then fell
    assert knock(1400, 600, 1500) == 1400 / 3000  # fell before the opponent
    assert knock(1500, 600, 1500) == 0.5  # same tick: not a knockdown
    assert knock(3000, 1600, 1500) == 1.0  # opponent fell before any hit
    assert knock(3000, None, 1500) == 1.0  # never hit
    assert knock(3000, 600, None) == 1.0  # hit, but nobody fell: a bell draw

    # v0.0.23: only a knockdown the ending attributes counts, not a blocked fall.
    def clean(attribution, opp_fall=1500):
        r = result(
            3000,
            50,
            scoring=9,
            first_hit=600,
            opp_fall=opp_fall,
            opp_attribution=attribution,
        )
        return motor_gate.fitness(r, 3000, 1.0, clean_knockdown=True)

    assert clean("knockdown") == 2.0
    assert clean("unresolved") == 1.0  # body contact before the fall: blocked
    assert clean("self") == 1.0  # fell on its own, no strike in the window
    assert clean("knockdown", opp_fall=None) == 1.0  # nobody fell

    # v0.0.24: every VALID knockdown counts; only a blocked (unresolved) fall not.
    def valid(attribution, opp_fall=1500):
        r = result(
            3000,
            50,
            scoring=9,
            first_hit=600,
            opp_fall=opp_fall,
            opp_attribution=attribution,
        )
        return motor_gate.fitness(r, 3000, 1.0, valid_knockdown=True)

    assert valid("self") == 2.0  # hit, then the opponent toppled later: valid
    assert valid("knockdown") == 2.0
    assert valid("unresolved") == 1.0  # body contact before the fall: blocked
    assert valid("self", opp_fall=None) == 1.0  # nobody fell

    # v0.0.25: a graded penalty for non-striking contact before the opponent falls.
    def contact(ticks, weight=0.5):
        r = result(
            3000, 50, scoring=9, first_hit=600, opp_fall=1500, opp_attribution="self"
        ) | {"contact_ticks": ticks}
        return motor_gate.fitness(
            r, 3000, 1.0, valid_knockdown=True, contact_penalty=weight
        )

    assert contact(0) == 2.0
    assert contact(250) == 2.0 - 0.25  # half the 0.5 s window in contact
    assert contact(5000) == 2.0 - 0.5  # capped at the full weight
    assert contact(5000, weight=0.0) == 2.0  # off by default


def test_es_drops_quarantined_pairs_and_follows_the_better_twin():
    params = np.zeros(N_PARAMS)
    pool = FakePool(
        [result(0, invalid="RuntimeError"), result(900), result(2400), result(1200)]
    )
    out, history = motor_gate.train(
        pool,
        params,
        ("scripted", "baseline"),
        np.random.default_rng(0),
        kind="params",
        iterations=1,
        pairs=2,
        sigma=0.1,
        lr=0.05,
        horizon_ticks=3000,
        strike_bonus=1.0,
        log=lambda row: None,
    )
    noise = np.random.default_rng(0).standard_normal((2, N_PARAMS))
    assert out == pytest.approx(0.05 * noise[1] / (2 * 0.1))  # only pair 1, +twin won
    assert history[0]["invalid"] == 1


def test_gate_counts_faults_as_failures():
    pool = FakePool(
        [result(3000, 2), result(3000), result(1200, 1), result(0, invalid="NaN")]
    )
    out = motor_gate.gate(
        pool, ("params", None), ("scripted", "baseline"), range(4), 3000
    )
    assert out["stance_pass_rate"] == 0.5 and out["reach_pass_rate"] == 0.5
    assert out["scoring_reach_rate"] == 0.0
    assert out["invalid"] == 1 and out["trials"] == 4


class RecordingPool(FakePool):
    def map(self, function, tasks):
        self.tasks = list(tasks)
        return super().map(function, tasks)


def test_es_cycles_a_list_of_opponents_with_twins_paired():
    params = np.zeros(N_PARAMS)
    first, second = ("scripted", "baseline"), ("scripted", "relaxed")
    pool = RecordingPool([result(1000 + 100 * k) for k in range(8)])
    motor_gate.train(
        pool,
        params,
        [first, second],
        np.random.default_rng(0),
        kind="params",
        iterations=1,
        pairs=4,
        sigma=0.1,
        lr=0.05,
        horizon_ticks=3000,
        strike_bonus=1.0,
        log=lambda row: None,
    )
    faced = [task[1] for task in pool.tasks]
    assert faced == [first, first, second, second, first, first, second, second]
    single = RecordingPool([result(1000 + 100 * k) for k in range(8)])
    motor_gate.train(
        single,
        params,
        first,
        np.random.default_rng(0),
        kind="params",
        iterations=1,
        pairs=4,
        sigma=0.1,
        lr=0.05,
        horizon_ticks=3000,
        strike_bonus=1.0,
        log=lambda row: None,
    )
    assert [task[1] for task in single.tasks] == [first] * 8
    # one opponent: identical tasks to the old single-spec call
    assert [t[2] for t in single.tasks] == [t[2] for t in pool.tasks]


def test_rank_shares_show_pooled_ranks_underweight_a_low_fitness_opponent():
    # Slots 0 and 1 interleave in a high fitness band; slot 2 (the base) sits
    # below both. Every + twin beats its - twin by the same margin.
    slots = np.arange(12) % 3
    level = np.array([1.5, 1.5, 0.5])[slots] + 0.1 * (np.arange(12) // 3)
    plus = level + 0.01 * slots
    paired = np.stack([plus, plus - 0.05], axis=1)
    ranks = centred_ranks(paired.ravel()).reshape(-1, 2)
    shares = motor_gate.rank_shares(ranks[:, 0] - ranks[:, 1], slots, 3)
    # high-band twins have the other slot's twin between them, the base's not
    assert shares == pytest.approx([0.4, 0.4, 0.2])
    assert sum(shares) == pytest.approx(1.0)
    assert motor_gate.rank_shares(np.zeros(3), np.arange(3), 3) == [0.0] * 3


def test_train_logs_rank_shares_only_when_asked_and_never_changes_the_update():
    params = np.zeros(N_PARAMS)
    first, second = ("scripted", "baseline"), ("scripted", "relaxed")
    rotation = [first, second, first, second]  # a spec may fill several slots
    stands = [1000 + 100 * k for k in range(8)]
    kwargs = dict(
        kind="params",
        iterations=1,
        pairs=4,
        sigma=0.1,
        lr=0.05,
        horizon_ticks=3000,
        strike_bonus=1.0,
        log=lambda row: None,
    )
    plain, rows = motor_gate.train(
        FakePool([result(s) for s in stands]),
        params,
        rotation,
        np.random.default_rng(0),
        **kwargs,
    )
    logged, logged_rows = motor_gate.train(
        FakePool([result(s) for s in stands]),
        params,
        rotation,
        np.random.default_rng(0),
        log_rank_shares=True,
        **kwargs,
    )
    assert np.array_equal(plain, logged)
    assert "rank_share_by_slot" not in rows[0]
    shares = logged_rows[0]["rank_share_by_slot"]
    assert len(shares) == 4 and sum(shares) == pytest.approx(1.0)
    # stand rises with the task index, so every - twin beats its + twin by one rank
    assert shares == pytest.approx([0.25] * 4)
