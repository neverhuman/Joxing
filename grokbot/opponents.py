"""Opponent sampling for a self-play league (grokbot reference for P4).

P4 of ``docs/SELF_PLAY_EXECUTION_PLAN.md`` proposes two layers, and lists the
edge cases each must handle:

1. Bucket mixture. Latest/current, history and anchors, with default shares
   0.50 / 0.35 / 0.15. An empty bucket's share goes to the non-empty buckets
   in proportion; all buckets empty is an error. A policy appearing in
   several buckets sums its probability.
2. Within history: ``eps/K + (1 - eps) * normalized_weights``. The weights come
   from prioritized fictitious self-play (PFSP) on the learner's record against
   each archived snapshot, smoothed by an explicit prior, so unseen opponents
   and 0/1 win rates stay well defined. ``hard`` weighting ``(1 - p)**power``
   favours opponents the learner loses to. ``variance`` weighting ``p(1 - p)``
   favours even matchups. If every weight is zero, the weights fall back to
   uniform. Malformed or non-finite records raise. The ``eps/K`` floor keeps
   every archived snapshot in play, which is what stops a learner forgetting
   them.

Sampling takes an explicit ``numpy.random.Generator``. Pass one per game, e.g.
seeded from that game's ``SeedSequence``, so opponent choice replays exactly.
"""

from __future__ import annotations

import math

import numpy as np

SHARES = {"latest": 0.50, "history": 0.35, "anchors": 0.15}


def history_probabilities(
    records,
    *,
    eps=0.1,
    weighting="hard",
    power=2.0,
    prior_wins=1.0,
    prior_games=2.0,
):
    """PFSP probabilities over archived opponents.

    ``records`` maps opponent -> (learner wins, games), counting draws as half
    wins. With no archived opponents (K = 0) the result is empty.
    """
    if not 0.0 <= eps <= 1.0 or not math.isfinite(eps):
        raise ValueError("eps must be in [0, 1]")
    if weighting not in ("hard", "variance"):
        raise ValueError("weighting must be 'hard' or 'variance'")
    if not (prior_games >= 0 and 0 <= prior_wins <= prior_games):
        raise ValueError("Prior must satisfy 0 <= prior_wins <= prior_games")
    names = list(records)
    if not names:
        return {}
    rates = []
    for name in names:
        wins, games = records[name]
        if not (math.isfinite(wins) and math.isfinite(games) and 0 <= wins <= games):
            raise ValueError(f"Malformed record for {name}: {records[name]}")
        total = games + prior_games
        rates.append((wins + prior_wins) / total if total > 0 else 0.5)
    p = np.array(rates)
    weights = (1.0 - p) ** power if weighting == "hard" else p * (1.0 - p)
    k = len(names)
    if weights.sum() <= 0:
        weights = np.ones(k)
    probs = eps / k + (1.0 - eps) * weights / weights.sum()
    return dict(zip(names, probs.tolist(), strict=True))


def mixture(latest=(), history=None, anchors=(), shares=SHARES):
    """Combine buckets into one opponent distribution.

    ``latest`` and ``anchors`` are sampled uniformly within their bucket;
    ``history`` is a probability dict, e.g. from ``history_probabilities``.
    """
    if set(shares) != set(SHARES) or any(
        not (math.isfinite(v) and v >= 0) for v in shares.values()
    ):
        raise ValueError("Shares must give a nonnegative weight to each bucket")
    history = history or {}
    if history and not math.isclose(sum(history.values()), 1.0, abs_tol=1e-9):
        raise ValueError("History probabilities must sum to 1")
    buckets = {
        "latest": {n: 1.0 / len(latest) for n in latest} if latest else {},
        "history": dict(history),
        "anchors": {n: 1.0 / len(anchors) for n in anchors} if anchors else {},
    }
    live = {b: shares[b] for b, members in buckets.items() if members}
    if not live or sum(live.values()) <= 0:
        raise ValueError("No opponent available: every weighted bucket is empty")
    scale = sum(live.values())
    combined = {}
    for bucket, share in live.items():
        for name, p in buckets[bucket].items():
            combined[name] = combined.get(name, 0.0) + share / scale * p
    return combined


def sample(probabilities, rng):
    """One opponent drawn with the game's own generator."""
    names = list(probabilities)
    if not names:
        raise ValueError("Empty opponent distribution")
    return names[int(rng.choice(len(names), p=np.array(list(probabilities.values()))))]
