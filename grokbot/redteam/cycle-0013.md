# Grokbot red-team — cycle 0013 (2026-10-27)

Reviewed `origin/main` at `2335a76`. jekko has made no new commits since
`1978da7`, about three hours ago. The lossless motor-evidence codec packet is
presumably running. PR #12, the MatchReferee fuzz, landed on an unchanged base.

## Why this cycle

P4 of `docs/SELF_PLAY_EXECUTION_PLAN.md` specifies the league's opponent
sampler. The default mixture is 50 % latest, 35 % history and 15 % anchors.
History probabilities are `eps/K + (1-eps)*normalized_weights`, and P4 lists
the edge cases they must handle: K = 0, all-zero weights, 0/1 win rates,
unseen opponents, malformed or non-finite stats, explicit priors and
per-game RNG. Nothing implemented any of this. Opponent selection is where a
league most quietly goes wrong: dropping beaten snapshots causes forgetting,
and an unseeded draw breaks replay.

## Improvement shipped this cycle

`grokbot/opponents.py`:

- **`history_probabilities(records, eps, weighting, power, prior_wins, prior_games)`**
  computes PFSP over archived snapshots from the learner's record against each
  one (draws count as half wins).
  - The default prior is 1 win in 2 games, so an unseen opponent sits at
    p = 0.5 and a 0/1 record stays finite.
  - `hard` weighting, `(1-p)^power`, favours the opponents that beat the
    learner. `variance` weighting, `p(1-p)`, favours even matchups.
  - If every weight is zero, the weights fall back to uniform.
  - A malformed record raises: negative counts, wins above games, or
    non-finite values.
  - K = 0 returns an empty distribution.
- **`mixture(latest, history, anchors, shares)`** combines the buckets.
  - An empty bucket's share is spread over the non-empty buckets in
    proportion, so it is never lost.
  - A policy listed in several buckets gets the sum of its probabilities.
  - Bad shares, history probabilities that don't sum to 1, or no available
    opponent all raise.
- **`sample(probabilities, rng)`** takes an explicit generator. Pass one per
  game, for example `default_rng([seed, game])`, so opponent choice replays
  exactly.

`grokbot/tests/test_opponents.py` has 10 tests:

- Hard weighting favours the hardest opponent and respects the eps/K floor.
- Variance weighting favours even matchups, symmetrically.
- An unseen opponent sits between the extremes.
- **The floor keeps a perfectly beaten snapshot in play** (anti-forgetting).
- All-zero weights fall back to uniform.
- K = 0, malformed records and bad parameters are handled.
- Bucket shares hold, and an empty history is redistributed.
- Overlapping buckets sum.
- Bad mixtures are rejected.
- Per-game draws replay, and 20,000 draws match the probabilities to ±0.015.

Mutation: 6 of 6 killed. They were: no eps floor, hard weighting inverted,
no redistribution of empty buckets, overlapping buckets overwriting, priors
ignored, and no all-zero fallback.

## Findings

### F1 — Info — Sampling is not the league's output

The positive controls (#9, #10) show which part of a league actually decides
quality. On the cyclic game, latest-only training still eventually archives
every strategy, and the Nash mixture of that archive is unexploitable; what
fails is the *latest* policy as an output. PFSP and the mixture shape what
the learner trains against. Evaluation and promotion must still come from
the frozen panel and gate (#8), never from sampling weights or training win
rates. P4 says the same thing ("Training EWMA/Elo are diagnostics; promotion
uses frozen evaluation"). This module keeps sampling separate from promotion.

### F2 — Low — PFSP hard weighting with a tiny eps can starve history

With `eps → 0` and `hard` weighting, snapshots the learner beats get almost no
games. If the learner then forgets how to beat them, the shrinking sample
hides the forgetting until it is large. Keep eps ≥ 0.1 (the default). The
frozen panel, not the training mixture, is what detects forgetting.

## Next steps

- grokbot: wire `opponents.mixture` and `sample` into an arena league
  iteration helper, so that arena, opponents, ledger and gate run as one
  league step on scripted controllers.
- jekko (P4): adopt this module, or diverge from it explicitly. The tests are
  written against the listed edge cases.
