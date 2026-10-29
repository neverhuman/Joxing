# Grokbot red-team — cycle 0045 (2026-10-29): v0.0.29 preregistration (the base faces half of the ES pairs)

This branches from `origin/main` independently of the v0.0.28 report PR and
changes no CHANGELOG or STATUS text. It is **the preregistration on its
own**. The run launches only after merge, in one command that waits for main
CI on the merged commit to be green and for load to be below 70. It runs at
nice 19 and is never SIGSTOP-paused.

## Why this lever (cycle 0044, post hoc, non-gating)

- **The training trial agrees with head-to-head against the base.** That
  refutes v0.0.28's registered "match-outcome term" reading.
- **Over its first 16 iterations, v0.0.28 traded the base matchup for the
  frozen opponents.**

  | Trial fitness | vs base | vs v0.0.19 | vs v0.0.13 |
  |---|---:|---:|---:|
  | base | 0.99 | 1.50 | 1.50 |
  | @16 | **0.69 ↓** | 1.59 ↑ | 1.73 ↑ |

  @16 lost 2–46 to the base, then recovered: @48 beat @16 46–2, and beat
  the base 43–5 post hoc.
- **Measured: the base got 0.23–0.28 of the update's |rank difference|**,
  for a third of the pairs. Pooled ranks under-weight the low-fitness
  matchup.
- **Per-opponent ranks would only reach about 0.31,** which is too small a
  change to expect it to close a 2–46 dip.

## The one lever (`protocols/v0.0.29.json`)

| | v0.0.28 | **v0.0.29** |
|---|---|---|
| Rotation per ES pair | v0.0.19, v0.0.13, base | **v0.0.19, base, v0.0.13, base** |
| Pairs facing the base / each external opponent | 4 / 4 | **6 / 3** |
| Expected base share of the update | 0.23–0.28 (measured) | about 0.4 |
| Unchanged | | Fitness, the v0.0.22 start, lr 0.05, σ 0.1, 48 × 12 (seed 19), the 0.5 m reset, ending v2 |

**Instrumentation (tested to never change the update):**
`motor_gate.rank_shares`, and `train(log_rank_shares=True)`, which logs each
rotation slot's share of every update.

**Smoke gate.** All three checks must pass:

1. The contact penalty varies against each external opponent (seeds
   43000–43047).
2. **Lever check:** the base's mean measured update share is above 0.30 over
   two lr-0 batches. These run on their own generator (rng 43200), so
   training is unaffected.
3. A 48-game base head-to-head against v0.0.19 is at least 90 % valid (seeds
   43100–43123).

**Gates, in order, on seeds never played:**

| Gate | Seeds |
|---|---|
| Primary against frozen v0.0.19 | 44000–44023 |
| Hard gate against v0.0.13 | 45000–45023 |
| Monotonicity: @48 vs @16, and @16 vs the base | 46000–46023 |

**Tag only if every gate passes.**

## Registered red-team notes

- **The base is trained against directly,** so a pass shows the policy can
  improve on its start, not that it generalises.
- **Each external opponent gets 6 trials per iteration instead of 8,** so the
  primary or hard gate may regress.
- **This lever compensates for the pooled-rank bias; it does not remove it.**
- **v0.0.28's post-hoc 43–5 changes nothing about v0.0.28's result.**

## Code

- **`grokbot/motor_gate.py`:** adds `rank_shares` and the opt-in
  `log_rank_shares` history field. The pooled update is unchanged.
- **Tests:**
  - A synthetic case where pooled ranks give the low band 0.2 of the update
    for a third of the pairs.
  - An identical update with and without logging, and a key that is absent
    unless the logging is requested.
- **`probes/cycle-0045-half-base-gate.py`:** the v0.0.28 probe with the
  4-slot rotation and the lever check. The smoke penalty check indexes the
  two external opponents explicitly, because the new rotation's slot 1 is
  the base. It was smoke-tested on seeds 99900 and above.
