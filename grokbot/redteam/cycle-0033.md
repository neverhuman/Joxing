# Grokbot red-team — cycle 0033 (2026-10-28): v0.0.22 miss diagnosed; v0.0.23 preregistration (clean knockdown)

Reviewed `origin/main` at `eec64e1`. jekko's `b9a7cfc` concluded scoring
phase A:

- the **referee awards points to rigid head/torso fixture strikes** (2 and 1
  points), so the scoring path works;
- whether a native athlete can earn points is still open;
- their native run failed on `No module named 'grokbot'`, an import-path
  problem in a standalone script, not a grokbot defect.

v0.0.22's miss landed in #32. Per owners 17:10 and 17:30, this PR diagnoses
the miss and lands the v0.0.23 preregistration **on its own, before any
run**.

## Diagnosis: why v0.0.22 missed (owner 17:30)

v0.0.22 lost **0 of 168 valid games**. Both failures were validity floors
broken by **blocked** games:

- the hard gate: 36/48 valid against v0.0.13, with 12 blocked;
- monotonicity @16 vs v0.0.19: 38/48 valid, with 10 blocked.

I replayed blocked hard-gate games deterministically: seeds 14002 and 14003,
both seats, using the run's final parameters.

| Seed | Candidate seat | Opponent fall tick | Attribution | Candidate head-hit ticks | Body-contact ticks in the 0.5 s before the fall | Contact pair |
|---|---|---:|---|---:|---:|---|
| 14002 | A / B | 1,375 | **unresolved** | 50 | 5 | torso to torso (`native_non_scoring`) |
| 14003 | A / B | 1,353 / 1,354 | **unresolved** | 50 | 9 | torso to torso |

The candidate lands its hits, and the opponent then falls. But **5–9 ms of
body-on-body contact** in the 500-tick attribution lookback leaves the fall
`unresolved`, so the game is blocked and does not count. v0.0.22's bonus
(`knocked_down`) paid for exactly these falls. Seats are identical, so there
is no seat effect.

## The one lever: a clean-knockdown objective (`protocols/v0.0.23.json`)

| | v0.0.22 | **v0.0.23** |
|---|---|---|
| Fitness bonus | `knocked_down` | **`clean_knocked_down`**: `knocked_down` **and** the opponent's first loss is attributed `knockdown` by `endings.BoutMonitor`, not `unresolved` |
| Unchanged | | Initialization (v0.0.19), training opponent (frozen v0.0.19), model, ES 48 × 12 (seed 13), the 0.5 m reset, ending v2 |

**Gates, the same as v0.0.22, on new seeds:**

| Gate | Seeds | Rule |
|---|---|---|
| Primary against frozen v0.0.19 | 16000–16023 | At least 40/48 valid, then at least 24/48 decisive (both fail loud), then b and c |
| Hard gate | 17000–17023 | At least 40/48 valid against v0.0.13 |
| Monotonicity: @48 vs @16, and @16 vs v0.0.19 | 18000–18023 | At least 40/48 valid, then Wilson > 0.55 |

**Tag `v0.0.23` only if every gate passes.** Every head-to-head is split by
seat, with blocked games counted separately.

**Red-team notes, registered:**

- **The attribution rule is strict.** *Any* body contact in the 0.5 s
  lookback voids a knockdown, so the candidate must keep its body off the
  opponent.
- **v0.0.13 falls on its own at about 1.35 s,** so wins against it may
  partly reflect the opponent's instability (as in cycle 0023).

## Code

| Change | Detail |
|---|---|
| `motor_gate.trial` | Now reports `opponent_fall_attribution` |
| `motor_gate.clean_knocked_down` | New |
| `fitness` and `train` | A `clean_knockdown` flag; existing modes unchanged; `clean_knockdown_share` is added to the history only when the flag is on |
| Tests | Four new cases: knockdown pays; unresolved does not; a self-caused fall does not; no fall does not. Plus the attribution key in a real short trial. A mutant that ignores attribution fails. |

**Probe:** `probes/cycle-0033-clean-knockdown-gate.py`. It is the v0.0.22
probe with the new flag and protocol. The smoke test on seeds 99970 and
above ran end to end. **No game has been played on seeds 16000–18023.**
