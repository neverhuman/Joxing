"""Per-policy summary: every attempt lands in exactly one column, results are
read from the right policy's view, and blocked bouts name the faller."""

import pytest

from grokbot.arena import policy_summary

COLUMNS = (
    "win",
    "draw",
    "loss",
    "blocked_as_faller",
    "blocked_other",
    "own_fault",
    "opponent_fault",
    "invalid_other",
)


def record(
    first, second, *, outcome=None, invalid=None, fault=None, losses=(), end=None
):
    return {
        "first": first,
        "second": second,
        "outcome": outcome,
        "invalid_reason": invalid,
        "fault_player": fault,
        "losses": [
            {"policy": p, "tick": t, "kind": k, "attribution": a}
            for p, t, k, a in losses
        ],
        "ending": None if end is None else {"reason": "x", "tick": end},
    }


RECORDS = [
    record("p", "q", outcome=1, losses=[("q", 900, "fall", "self")], end=900),
    record("q", "p", outcome=1, losses=[("p", 700, "fall", "self")], end=700),
    record("p", "q", outcome=0),
    record(
        "p",
        "q",
        invalid="blocked: unresolved loss attribution",
        # p also goes down later, unresolved, after the bout already ended.
        losses=[("q", 500, "fall", "unresolved"), ("p", 800, "fall", "unresolved")],
        end=500,
    ),
    record("q", "p", invalid="controller fault B: command is nonfinite", fault="p"),
    record("p", "q", invalid="FloatingPointError: Nonfinite activation"),
    record("p", "q", outcome=-1, losses=[("p", 300, "exit", "self")], end=300),
    record("q", "p", outcome=1, losses=[("p", 400, "fall", "knockdown")], end=400),
]


def test_every_attempt_is_in_exactly_one_column():
    summary = policy_summary(RECORDS)
    for row in summary.values():
        assert row["attempted"] == len(RECORDS)
        assert sum(row[c] for c in COLUMNS) == row["attempted"]


def test_results_and_failures_are_charged_from_each_policys_view():
    p, q = policy_summary(RECORDS)["p"], policy_summary(RECORDS)["q"]
    assert (p["win"], p["draw"], p["loss"]) == (1, 1, 3)
    assert (q["win"], q["draw"], q["loss"]) == (3, 1, 1)
    assert p["own_fault"] == 1 and q["opponent_fault"] == 1
    assert q["blocked_as_faller"] == 1 and p["blocked_other"] == 1
    assert q["blocked_as_faller_rate"] == pytest.approx(1 / 8)
    assert p["invalid_other"] == q["invalid_other"] == 1


def test_fall_counts_and_first_loss_times():
    summary = policy_summary(RECORDS)
    assert (
        summary["p"]["self_falls"] == 1
    )  # the exit and the unresolved fall don't count
    assert summary["q"]["self_falls"] == 1
    assert summary["p"]["knocked_down"] == 1 and summary["q"]["knocked_down"] == 0
    assert summary["p"]["median_first_loss_tick"] == 550.0  # 700, 800, 300, 400
    assert summary["q"]["median_first_loss_tick"] == 700.0  # 900, 500
    assert (
        policy_summary([record("a", "b", outcome=0)])["a"]["median_first_loss_tick"]
        is None
    )
