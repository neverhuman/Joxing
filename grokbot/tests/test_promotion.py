"""Promotion gate: promote real improvement; refuse cycling, thin evidence,
broken games, blocked-bout exploitation and corner bias."""

import math

import pytest

from grokbot.promotion import GateConfig, decide, hoeffding_lower

CONFIG = GateConfig(min_valid=20)
PANEL = ["old", "mid", "inc"]


def pair(first, second, outcome, *, invalid=None, losses=(), end=None):
    """Both corners of one paired game with the same result for ``first``."""
    return [
        {
            "first": first,
            "second": second,
            "first_side": side,
            "outcome": None if invalid else outcome,
            "invalid_reason": invalid,
            "fault_player": None,
            "losses": [
                {"policy": p, "tick": t, "kind": "fall", "attribution": a}
                for p, t, a in losses
            ],
            "ending": None if end is None else {"reason": "x", "tick": end},
        }
        for side in ("A", "B")
    ]


def league(cand_vs_inc, cand_vs, inc_vs, pairs=10):
    """``pairs`` side-swapped pairs per matchup; outcomes cycle through lists.

    Odd pairs list the opponent first, so results must be read from each
    policy's own view.
    """

    def game(me, other, outcome, k):
        return pair(me, other, outcome) if k % 2 == 0 else pair(other, me, -outcome)

    records = []
    for k in range(pairs):
        records += game("cand", "inc", cand_vs_inc[k % len(cand_vs_inc)], k)
        for member in ("old", "mid"):
            records += game(
                "cand", member, cand_vs[member][k % len(cand_vs[member])], k
            )
            records += game("inc", member, inc_vs[member][k % len(inc_vs[member])], k)
    return records


IMPROVES = league([1], {"old": [1], "mid": [1]}, {"old": [1], "mid": [0]})


def test_clear_improvement_is_promoted():
    decision = decide(IMPROVES, "cand", "inc", PANEL, CONFIG)
    assert decision.promote and decision.reasons == ()
    assert decision.head_to_head["games"] == 20
    assert decision.head_to_head["lower"] > 0
    assert decision.per_member["mid"]["candidate_mean"] == 1.0


def test_cycling_candidate_is_refused():
    """Beats the incumbent every time, but loses to an archived snapshot that
    the incumbent beats: latest-vs-latest progress, panel regression."""
    records = league([1], {"old": [-1], "mid": [1]}, {"old": [1], "mid": [0]})
    decision = decide(records, "cand", "inc", PANEL, CONFIG)
    assert not decision.promote
    assert decision.reasons == ("regression vs old: -1.000 < incumbent +1.000 - 0.1",)


def test_small_regressions_inside_the_margin_are_tolerated():
    records = league([1], {"old": [1] * 19 + [0], "mid": [1]}, {"old": [1], "mid": [0]})
    assert decide(records, "cand", "inc", PANEL, CONFIG).promote


def test_thin_or_narrow_evidence_is_refused():
    thin = league([1], {"old": [1], "mid": [1]}, {"old": [1], "mid": [0]}, pairs=9)
    reasons = decide(thin, "cand", "inc", PANEL, CONFIG).reasons
    assert any(r.startswith("coverage: 18 valid games vs incumbent") for r in reasons)
    narrow = league(
        [1, 1, 1, -1, 0], {"old": [1], "mid": [1]}, {"old": [1], "mid": [0]}
    )
    decision = decide(narrow, "cand", "inc", PANEL, CONFIG)
    assert decision.head_to_head["mean"] == pytest.approx(0.4)
    assert decision.reasons[0].startswith("head to head: lower bound")


def test_broken_games_are_refused_even_when_winning():
    broken = 2 * pair("cand", "old", None, invalid="FloatingPointError: NaN")
    decision = decide(IMPROVES + broken, "cand", "inc", PANEL, CONFIG)
    assert decision.reasons == ("integrity: invalid share 0.062",)  # 4 of 64


def test_blocked_bout_exploitation_is_refused():
    blocked = [
        r
        for r in pair(
            "cand",
            "mid",
            None,
            invalid="blocked: unresolved loss attribution",
            losses=[("cand", 500, "unresolved")],
            end=500,
        )
    ]
    decision = decide(IMPROVES + blocked, "cand", "inc", PANEL, CONFIG)
    # 2 of 62 games: under the 5 % invalid cap, over the 2 % blocked-as-faller cap.
    assert decision.reasons == ("integrity: blocked-as-faller share 0.032",)


def test_corner_bias_is_refused():
    one_sided = [
        r for r in IMPROVES if not (r["first"] == "cand" and r["first_side"] == "B")
    ]
    decision = decide(one_sided, "cand", "inc", PANEL, GateConfig(min_valid=10))
    assert any(
        r.startswith("integrity: corner-A mean utility") for r in decision.reasons
    )


def test_hoeffding_bound():
    assert hoeffding_lower(0.5, 0, 0.95) == -1.0
    eps = 2 * math.sqrt(math.log(20) / (2 * 100))
    assert hoeffding_lower(0.5, 100, 0.95) == pytest.approx(0.5 - eps)
    assert hoeffding_lower(1.0, 6, 0.95) > 0 > hoeffding_lower(1.0, 5, 0.95)
