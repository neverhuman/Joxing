"""v0.0.12 engagement machinery: anti-stalling no-contests, per-corner strike
attribution, the boxing-range reset, the engagement promotion gate and the
training-only reach measurement."""

import numpy as np
import pytest

from grokbot.endings import ContactClassifier, Ending, anti_stalling
from grokbot.league import Game, Ledger
from grokbot.promotion import GateConfig, decide
from grokbot.reference_env import ReferenceEndingEnv
from grokbot.scenarios import RangeBoxingEnv
from grokbot.selfplay import (
    OPPONENT_TARGETS,
    OWN_HANDS,
    finite_or_none,
    reach,
    safe_ratings,
)
from grokbot.tests.test_endings import FAST, REGISTRY, rows
from grokbot.tests.test_promotion import IMPROVES, PANEL
from joxing.env import BoxingEnv
from joxing.scene import build_scene

WIN = Ending((1, -1), "B self-caused fall", 500)


@pytest.fixture(scope="module")
def scene():
    return build_scene()


@pytest.mark.parametrize(
    "strikes, points, stays",
    [
        ({"A": [], "B": []}, (0, 0), False),  # pure outlasting: no contest
        ({"A": [499], "B": []}, (0, 0), True),  # landed before the ending
        ({"A": [], "B": [500]}, (0, 0), True),  # on the ending tick counts
        ({"A": [501], "B": []}, (0, 0), False),  # only after the ending
        ({"A": [], "B": []}, (1, 0), True),  # points are engagement
    ],
)
def test_anti_stalling_voids_only_unengaged_results(strikes, points, stays):
    ending = anti_stalling(WIN, strikes, points)
    if stays:
        assert ending is WIN
    else:
        assert ending.utility is None and ending.reason.startswith("no contest")


def test_anti_stalling_passes_no_results_through():
    blocked = Ending(None, "blocked: unresolved loss attribution", 7)
    assert anti_stalling(blocked, {"A": [], "B": []}, (0, 0)) is blocked


def test_strikes_are_credited_to_the_glove_owner():
    classifier = ContactClassifier(REGISTRY)
    evidence = rows(
        40,
        lambda t: [(5, 6, 300.0, FAST)] if 5 <= t < 8 else [],  # A glove, B head
        lambda t: [(8, 5, 300.0, FAST, FAST, 0.0)]
        if 20 <= t < 22
        else [],  # B hits A's still glove
    )
    for row in evidence:
        classifier.feed(row)
    assert classifier.strikes_by == {"A": [5, 6, 7], "B": [20, 21]}


def test_range_reset_is_a_named_side_symmetric_variant(scene):
    native = BoxingEnv(scene, horizon_ticks=400)
    near = RangeBoxingEnv(scene, separation_m=0.8, horizon_ticks=400)
    assert near.contract_hash != native.contract_hash
    assert near.contract["reset_scenario"] == "grokbot-range-0.80m-v0"
    obs, _ = near.reset(seed=0)
    pelvis = [near.data.xpos[near.model.body(a + "_pelvis").id] for a in "AB"]
    assert np.linalg.norm((pelvis[0] - pelvis[1])[:2]) == pytest.approx(0.8)
    rng = np.random.default_rng(5)
    for _ in range(10):
        command = rng.uniform(-1, 1, 700).astype(np.float32)
        obs = near.step({"A": command, "B": command.copy()})["observations"]
        assert np.max(np.abs(obs["A"].values - obs["B"].values)) < 1e-6
    with pytest.raises(ValueError):
        RangeBoxingEnv(scene, separation_m=0.1)


def test_outlasting_a_passive_opponent_is_a_no_contest(scene):
    env = ReferenceEndingEnv(BoxingEnv(scene, horizon_ticks=3000), anti_stalling=True)
    env.reset(seed=0)
    commands = {"A": np.full(700, -1.0, np.float32), "B": np.full(700, 1.0, np.float32)}
    total = 0.0
    while True:
        result = env.step(commands)
        total += result["rewards"]["A"]
        if result["terminated"]:
            break
    assert result["info"]["no_contest"] and not result["info"]["blocked"]
    assert result["info"]["ending"]["utility"] is None and total == pytest.approx(0.0)


def test_engagement_gate_refuses_contactless_rating_gains():
    passive = [r | {"engaged": {"cand": False}} for r in IMPROVES]
    reasons = decide(
        passive, "cand", "inc", PANEL, GateConfig(min_engaged_share=0.25)
    ).reasons
    assert reasons == ("engagement: landed a strike in 0.000 of bouts < 0.25",)
    active = [r | {"engaged": {"cand": i % 2 == 0}} for i, r in enumerate(IMPROVES)]
    assert decide(
        active, "cand", "inc", PANEL, GateConfig(min_engaged_share=0.25)
    ).promote


def test_no_contests_are_not_counted_as_broken_games():
    stalled = [
        r
        | {
            "outcome": None,
            "invalid_reason": "no contest: no engagement before the ending",
        }
        for r in IMPROVES[:4]
    ]
    decision = decide(IMPROVES + stalled, "cand", "inc", PANEL)
    assert not any(r.startswith("integrity: invalid share") for r in decision.reasons)


def test_reach_is_the_closest_own_hand_to_opponent_head_or_trunk():
    values = np.zeros(1970, np.float32)
    values[OWN_HANDS[0] : OWN_HANDS[0] + 3] = [0.3, 0.0, 1.0]
    values[OWN_HANDS[1] : OWN_HANDS[1] + 3] = [0.1, 0.0, 1.0]
    values[OPPONENT_TARGETS[0] : OPPONENT_TARGETS[0] + 3] = [0.9, 0.0, 1.0]
    values[OPPONENT_TARGETS[1] : OPPONENT_TARGETS[1] + 3] = [0.5, 0.0, 1.0]
    assert reach(values) == pytest.approx(0.2)


def test_a_league_of_no_contests_reports_no_ratings_instead_of_crashing():
    ledger = Ledger(["gen0", "gen1", "baseline"])
    ledger.add(Game("gen1", "gen0", "A", None, "no contest: no engagement"))
    assert safe_ratings(ledger, "baseline") == {}
    assert finite_or_none(ledger.side_balance()) is None
    ledger.add(Game("gen1", "gen0", "B", 1))
    # The anchor never played: ratings are centred instead of all-NaN.
    assert set(safe_ratings(ledger, "baseline")) == {"gen0", "gen1"}
