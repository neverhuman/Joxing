# Grokbot red-team — cycle 0032 / v0.0.22 (2026-10-28): first decisive wins over v0.0.19; blocked games fail the gates

Reviewed `origin/main` at `2552354`. jekko's new commits are native-codec
and recording work; none touches `grokbot/`.

The v0.0.22 protocol (`4a03f36e…`) landed in #31 before the run. The run
used merged main `b15c695` (tree `03a3aca1`), ran at nice 19 with load 41,
and took 3,929 s.

## v0.0.22 — registered level-up **NOT met** (primary passed; hard gate and monotonicity failed); not tagged

The one lever was a knockdown objective: a head/torso hit, then the opponent
falls first.

Every row is split by the stronger side's seat. The two seats were
identical in every matchup.

| Matchup (48 games) | Valid | W–L–D | No-contest | Blocked | Seat A W–L–D/NC/Blk | Seat B | Rule | Result |
|---|---:|---|---:|---:|---|---|---|---|
| **Primary: vs frozen v0.0.19, seeds 13000–13023** | **46** | **46–0–0** | 0 | 2 | 23–0–0 / 0 / 1 | same | ≥ 40 valid, ≥ 24 decisive, b, c | **PRIMARY_PASSED** |
| **Hard gate: vs frozen v0.0.13, seeds 14000–14023** | **36** | 36–0–0 | 0 | **12** | 18–0–0 / 0 / 6 | same | ≥ 40 valid | **failed** |
| Monotonicity: @48 vs @16, seeds 15000–15023 | 48 | 48–0–0 | 0 | 0 | 24–0–0 / 0 / 0 | same | ≥ 40 valid, Wilson > 0.55 | pass |
| Monotonicity: @16 vs @0 (v0.0.19), seeds 15000–15023 | **38** | 38–0–0 | 0 | **10** | 19–0–0 / 0 / 5 | same | ≥ 40 valid | **unmeasurable, so failed** |
| v0.0.19 mirror, primary seeds (secondary) | 0 | — | 46 | 2 | | | | — |
| vs frozen v0.0.16, seeds 14000–14023 (secondary) | 48 | 48–0–0 | 0 | 0 | | | | — |

**Primary detail:** all 46 wins included head/torso hits. Lower95 was +0.49
and Elo +661 (lower bound +186). **These are the first decisive wins over
v0.0.19.** In v0.0.21 the same opponent gave 4–2 with 42 draws.

**Motor gate against gen4:** stance 0/24, reach and hits 24/24, median
stand 1.82 s. The policy wins by knocking opponents down, not by standing
the full 3 s.

**Training:** the knockdown share rose from 0.04 to between 0.71 and 0.79,
while mean stand time fell from about 2.7 to 2.3 s.

`level_up = false`, because the hard gate and monotonicity both failed.
**No tag** (owners 15:10, 16:10, 16:50: tag only on a registered pass). A
pass against v0.0.13 alone is not the registered criterion.

## Findings

### F1 — High — Every valid game was won; both failures are *blocked* games

Across the primary, the hard gate and both monotonicity matchups,
v0.0.22 and its @16 checkpoint lost **0** valid games out of 168.

The gates failed only because 12 games against v0.0.13 and 10 games for
@16 vs v0.0.19 were **blocked**. In a blocked game the fall is preceded by
body-on-body contact, so ending v2 cannot attribute the loss and the game
does not count. The knockdown objective rewarded *any* hit followed by the
opponent falling first, including falls that the ending would leave
unresolved because the bodies collided.

### F2 — Positive — The draw problem is solved; compute is monotone where it is measurable

v0.0.21 drew 42 of 48 games against v0.0.19; v0.0.22 drew **0**. More
training beat less, 48–0 for @48 vs @16, on fully valid games. This is the
first monotone result in compute: v0.0.21 lost 0–20 at the same
comparison.

## Next single lever: v0.0.23, to be preregistered in its own PR

**A clean-knockdown objective.** The bonus counts only when the ending
would **attribute** the opponent's fall to the candidate's strike:
`BoutMonitor` attribution `knockdown`, and **not** `unresolved` because of
body contact. Everything else stays as in v0.0.22.

This targets F1 exactly. Both binding failures are blocked
(unattributable) games, and those are the games the current bonus still
pays for. The same gates are kept: the primary against frozen v0.0.19, the
v0.0.13 hard gate of at least 40/48 valid, monotonicity, and the per-seat
split.

## What this PR adds

- `data/cycle-0031-knockdown-gate-summary.json`: per-trial and per-game
  records for every matchup, with seat splits; the training history; the
  final parameters; and the full-report SHA-256. The @16 parameters are
  reproducible and not stored.
- This report and STATUS.
