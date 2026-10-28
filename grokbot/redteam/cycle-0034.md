# Grokbot red-team — cycle 0034 / v0.0.23 (2026-10-28): VALIDITY_FAILED; the lever never fired, and why

Reviewed `origin/main` at `2f3c9ec`. jekko's new commits are recording and
native-wave work; none touches `grokbot/`.

The v0.0.23 protocol (`28408dd9…`) landed in #33 before the run. The run
used merged main `34b9cc7` (tree `b4bf9bd0`), ran at nice 19 and took
4,302 s. **The probe failed loud: exit code 1, `VALIDITY_FAILED`.**

## v0.0.23 — preregistered criterion NOT met; not tagged

The one lever was the clean-knockdown bonus: a knockdown whose fall
`BoutMonitor` attributes as `knockdown`.

| Matchup (48 games, by seat) | Valid | W–L–D | No-contest | Blocked | Seat A W–L–D/NC/Blk | Seat B | Result |
|---|---:|---|---:|---:|---|---|---|
| **Primary: vs frozen v0.0.19, seeds 16000–16023** | **5** | 0–0–5 | 35 | 8 | 0–0–2 / 18 / 4 | 0–0–3 / 17 / 4 | **VALIDITY_FAILED** (5 < 40) |
| Hard gate: vs frozen v0.0.13, seeds 17000–17023 | 14 | 14–0–0 | 4 | **30** | 7–0–0 / 2 / 15 | same | failed |
| Monotonicity: @48 vs @16, seeds 18000–18023 | 34 | 0–0–34 | 10 | 4 | | | unmeasurable |
| Monotonicity: @16 vs v0.0.19, seeds 18000–18023 | 4 | 0–0–4 | 23 | 21 | | | unmeasurable |
| v0.0.19 mirror, primary seeds (secondary) | 2 | 0–0–2 | 39 | 7 | | | — |
| vs frozen v0.0.16 (secondary) | 40 | 40–0–0 | 2 | 6 | | | — |

**Training:** `clean_knockdown_share` was **0.00 in all 48 iterations**,
while the plain knockdown share was 0–0.25. The bonus never paid, so ES
ranked on stand time alone. The result is a standing policy whose games
are no-contests (against v0.0.19) or blocked (against v0.0.13). Its motor
gate was stance 2/24 with median 1.92 s. The v0.0.19 gate re-ran
identically. The training observation was logged at iteration 8, before
any result was read.

## Diagnosis: why the bonus never fires (owner 18:40)

I replayed the **v0.0.22 policy**, which had a knockdown share of about
0.75 in training, against frozen v0.0.19 on 24 non-protocol seeds
(99900–99923), with 2 workers.

| | Count |
|---|---:|
| Trials | 24 |
| Knockdowns (`knocked_down`) | **24** |
| Opponent fall attributed `knockdown` | **0** |
| Opponent fall attributed `self` | **22** |
| Opponent fall attributed `unresolved` (blocked) | 2 |

**Mechanism.** The candidate's single strike lands at **tick about 252**.
v0.0.19 then falls **1.0–2.0 s later**, at ticks 1,479–2,496, with a
strike-to-fall gap of 1,007–2,028 ticks. That is far outside the 500-tick
attribution lookback, which contains **no strike and no body contact**. The
ending therefore attributes the fall to the opponent itself (`self`).

**Consequences:**

1. **The v0.0.23 lever was mis-specified, and the error is mine.** Requiring
   attribution `knockdown` excludes the dominant `self` falls. Those are
   **valid** wins under ending v2, because the faller loses. Only
   `unresolved` falls are invalid (blocked). The cycle-0033 diagnosis was
   right for the blocked games against v0.0.13, but it did not check how
   the *training* opponent's falls are attributed.
2. **v0.0.22's 46–0 primary was mostly "one early hit, then the opponent
   topples on its own about 1.5 s later"**, not attributed knockdowns. The
   wins are valid under the registered ending, but the causal link from
   strike to fall is weak. It is reported here as a caveat on that result.

## Next: v0.0.24 (its own preregistration PR, owner 18:40)

The lever must **provably fire**. The protocol will include a
**preregistered smoke gate**: before the full run, a short run at the base
policy must produce the bonus event in **at least 10 %** of trials, or the
full run does not start.

The candidate lever is a **valid-knockdown bonus**: `knocked_down` and the
fall is **not** `unresolved`. It pays for the valid wins and excludes only
blocked falls. In the replay above it would have fired in 22 of 24 trials.

## What this PR adds

- `data/cycle-0033-clean-knockdown-gate-summary.json`: per-game records
  with seat splits, the training history, the final parameters, the
  diagnosis replay rows, and the full-report SHA-256.
- This report, the CHANGELOG entry (untagged miss) and STATUS.
