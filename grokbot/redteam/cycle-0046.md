# Grokbot red-team — cycle 0046 / v0.0.29 (2026-10-29): more base weight moves @16 from 2–46 to 16–32; not enough

Reviewed `origin/main` at `aeaced5`. jekko's new commits cover H10 arm
residual integration, a retained failed Linux gate of his own experiment,
and a bounded history-diagnosis preregistration; none touches `grokbot/`.

The v0.0.29 protocol (`219bd438…`, #45) landed before the run. The run
launched at 08:59:44Z from the preregistered `8cc82cd` checkout, at load
65.3, at nice 19 and with no pausing; it took 3,844 s. It waited for main CI
success on `160185e`, which contains `8cc82cd`, because CI on `8cc82cd`
itself was **cancelled** by the concurrency group when jekko pushed. The
gate now accepts CI on any main commit containing the prereg commit.
Attempts: 1, which completed. The v0.0.19 motor gate reproduced exactly.

## v0.0.29 — registered level-up **NOT met** (monotonicity); not tagged

The one lever was the frozen v0.0.22 base facing half of the ES pairs, with
rotation v0.0.19, base, v0.0.13, base.

| Matchup (48 games) | Valid | W–L–D | Blocked | Seat A | Seat B | Wilson 95 % | Result |
|---|---:|---|---:|---|---|---|---|
| Smoke gate (penalty, **lever: base update share 0.40 / 0.57**, base h2h 48/48 valid) | | | | | | | **passed** |
| **Primary: vs frozen v0.0.19** | 44 | **44–0–0** | 4 | 22–0 / 2 blk | 22–0 / 2 blk | 0.92–1.00 | **passed** (42/44 wins with hits) |
| **Hard gate: vs frozen v0.0.13** | 46 | **46–0–0** | 2 | 23–0 / 1 blk | 23–0 / 1 blk | 0.92–1.00 | **passed** (≥ 40 valid) |
| Monotonicity: @48 vs @16 | 48 | **47–1–0** | 0 | 24–0 | 23–1 | 0.89–1.00 | **passed** |
| Monotonicity: @16 vs the v0.0.22 base | 48 | **16–32–0** | 0 | 8–16 | 8–16 | **0.22–0.48** | **failed** (needs > 0.55) |
| v0.0.16 (secondary) | 44 | 44–0–0 | 0 (4 no-contest) | | | | — (only 2 of 44 wins with hits) |

- **Motor gate against gen4:** stance 2/24, median stand 2.32 s.
- **Training:**
  - mean stand 2.05 → 2.28 s (mean of the last 8 iterations);
  - valid-knockdown share 0.54 → 0.57;
  - contact 275 → 188 ticks.
- **Logged base share of every update:** mean 0.44 over the first 16
  iterations and 0.48 over the last 32, ranging from 0.27 to 0.72. In
  v0.0.28 it was 0.23–0.28.

`level_up = false`, so there is **no tag**. Both gates and @48 > @16 passed
for the second version running. Only @16 vs its own start keeps failing.

## Findings

1. **High: the early dip responds to the base's weight.** @16 vs its own
   start, by version:

   | Version | Base's share of ES pairs | Base's measured update share (first 16 iterations) | @16 vs base |
   |---|---|---|---|
   | v0.0.26 / v0.0.27 | 0 | 0 | 2–40 / 0–48 |
   | v0.0.28 | 1/3 | 0.23–0.28 (measured post hoc, cycle 0044) | 2–46 |
   | **v0.0.29** | **1/2** | **0.44 (logged)** | **16–32** |

   For the first time, @16 wins a third of its games against the start.
   The relationship is non-linear: going from 0 to a quarter of the update
   did nothing, and going to 0.44 gave a third of the games.
2. **Medium: half of the registered reading was wrong.** Its diagnosis said
   "the early trade-off is not a weighting effect", and finding 1 contradicts
   that. Its **next lever**, the base as the only training opponent for the
   first 16 iterations, is the natural limit of the same trend, so I follow
   the lever and record the wrong diagnosis.
3. **Low: two smaller observations.**
   - Against v0.0.16, only 2 of 44 wins came with hits: v0.0.16 mostly falls
     on its own. This is secondary and not gating.
   - The primary still has 4 blocked games, which validity absorbs.
4. **Process: a stall avoided.** Main CI runs share a concurrency group, so
   a newer push cancels CI on the previous commit. A launch gate that
   waited for CI on the exact merge commit would have waited forever. It
   now accepts success on any main commit that contains the prereg commit.

## Next: v0.0.30, its own preregistration PR, opened off main

**One lever: a curriculum.** For iterations 0–15 the only training opponent
is the frozen v0.0.22 base, so its update share is 1.0. Iterations 16–47 use
v0.0.29's 4-slot rotation. Everything else stays as in v0.0.29.

**Registered risks:**

- @16 sees no external opponent. Only the final policy is gated against
  v0.0.19 and v0.0.13.
- @48 vs @16 may drop if the rotation phase trades away what the
  self-play-only phase learned.

## What this PR adds

- `data/cycle-0045-half-base-gate-summary.json`: per-trial and per-game
  records with seat splits, the history (including `rank_share_by_slot`),
  the @16 and final parameters, the full-report SHA-256, and the attempts
  ledger.
- This report, the CHANGELOG entry (with the unpublished-versions table
  extended) and STATUS.
