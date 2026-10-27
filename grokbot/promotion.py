"""Reference promotion gate for a self-play league (grokbot candidate).

Decides whether a frozen candidate may replace the incumbent, using only games
against a frozen evaluation panel, never training reward or the learner's own
recent opponents. Input is ``grokbot.arena`` records: side-swapped pairs with
common random numbers, every attempt included.

The candidate is promoted only if every check passes:

1. Coverage: at least ``min_valid`` valid games against the incumbent and
   against each other panel member, for both candidate and incumbent.
2. Integrity: the candidate's invalid share (engine, controller or blocked) and
   its blocked-as-faller share stay under their caps, and the corner-A mean
   utility over its games is within ``max_side_bias`` of zero.
3. Head to head: a one-sided Hoeffding lower bound on the candidate's mean
   utility against the incumbent is above zero.
4. No regression: against every other panel member, the candidate's mean
   utility is at least the incumbent's minus ``regression_margin``. Beating the
   latest snapshot while losing to an archived one is cycling, not progress.

Hoeffding bounds are distribution-free for utilities in [-1, 1]. They are
conservative and ignore pairing, so they never overstate significance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from grokbot.arena import policy_summary


@dataclass(frozen=True)
class GateConfig:
    min_valid: int = 20
    confidence: float = 0.95
    max_invalid_share: float = 0.05
    # Stricter than the invalid cap: going down into a no-result is the exploit.
    max_blocked_as_faller_share: float = 0.02
    max_side_bias: float = 0.1
    regression_margin: float = 0.1


@dataclass(frozen=True)
class Decision:
    promote: bool
    reasons: tuple[str, ...]
    head_to_head: dict = field(default_factory=dict)
    per_member: dict = field(default_factory=dict)


def hoeffding_lower(mean, n, confidence):
    """One-sided lower bound on the mean of n utilities in [-1, 1]."""
    if n <= 0:
        return -1.0
    return mean - 2.0 * math.sqrt(math.log(1.0 / (1.0 - confidence)) / (2.0 * n))


def _utilities(records, me, other):
    """Valid outcomes of ``me`` against ``other``, from me's view."""
    values = []
    for r in records:
        if {r["first"], r["second"]} != {me, other} or r["outcome"] is None:
            continue
        values.append(r["outcome"] if r["first"] == me else -r["outcome"])
    return values


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def decide(records, candidate, incumbent, panel, config=GateConfig()):
    """Promotion decision for ``candidate`` over ``incumbent``.

    ``panel`` lists the frozen evaluation opponents; the incumbent may be in it.
    """
    others = [m for m in panel if m not in (candidate, incumbent)]
    reasons = []

    h2h = _utilities(records, candidate, incumbent)
    head = {
        "games": len(h2h),
        "mean": _mean(h2h),
        "lower": hoeffding_lower(
            _mean(h2h) if h2h else 0.0, len(h2h), config.confidence
        ),
    }
    if len(h2h) < config.min_valid:
        reasons.append(f"coverage: {len(h2h)} valid games vs incumbent")
    elif head["lower"] <= 0.0:
        reasons.append(f"head to head: lower bound {head['lower']:+.3f} <= 0")

    per_member = {}
    for member in others:
        mine = _utilities(records, candidate, member)
        theirs = _utilities(records, incumbent, member)
        row = {
            "candidate_games": len(mine),
            "incumbent_games": len(theirs),
            "candidate_mean": _mean(mine),
            "incumbent_mean": _mean(theirs),
        }
        per_member[member] = row
        if min(len(mine), len(theirs)) < config.min_valid:
            reasons.append(
                f"coverage: {member} has {len(mine)}/{len(theirs)} valid games"
            )
        elif row["candidate_mean"] < row["incumbent_mean"] - config.regression_margin:
            reasons.append(
                f"regression vs {member}: {row['candidate_mean']:+.3f} < "
                f"incumbent {row['incumbent_mean']:+.3f} - {config.regression_margin}"
            )

    own = [r for r in records if candidate in (r["first"], r["second"])]
    if own:
        invalid = sum(r["outcome"] is None for r in own) / len(own)
        if invalid > config.max_invalid_share:
            reasons.append(f"integrity: invalid share {invalid:.3f}")
        blocked = policy_summary(own)[candidate]["blocked_as_faller_rate"]
        if blocked > config.max_blocked_as_faller_share:
            reasons.append(f"integrity: blocked-as-faller share {blocked:.3f}")
        corner_a = [
            r["outcome"] if r["first_side"] == "A" else -r["outcome"]
            for r in own
            if r["outcome"] is not None
        ]
        if corner_a and abs(_mean(corner_a)) > config.max_side_bias:
            reasons.append(f"integrity: corner-A mean utility {_mean(corner_a):+.3f}")
    return Decision(not reasons, tuple(reasons), head, per_member)
