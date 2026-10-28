# Grokbot red-team — cycle 0037 (2026-10-28): why the bonus "over-fires"; v0.0.25 preregistration (graded contact penalty)

Reviewed `origin/main` at `30d0443`; v0.0.24's `SMOKE_FAILED` landed in
#36. Per owner 20:45: diagnose why the bonus "over-fires" (23/24), then
preregister v0.0.25 with **one** lever. This PR lands the preregistration
**on its own; the run starts only after merge**, launched by a single
command that waits for load below 70.

## Diagnosis: the 23/24 is saturation, not a bug

- **v0.0.22 was trained to produce this event.** I replayed the v0.0.24
  smoke head-to-heads (seeds 19100–19105, both seats; already-read seeds,
  so this is analysis and not a new evaluation). In 10 of 12 games
  v0.0.19 falls at about **2.0–2.25 s** after one early hit, attributed
  `self`, which is a valid win. The 2 blocked games (seed 19105, both
  seats) are late falls at about 2,640 ticks, left `unresolved`.
- **A binary bonus that is true in about 96 % of base trials gives ES no
  ranking signal.** Only stand time differentiates the twins.

## Design-stage checks (non-protocol seeds, 2 workers, load below 70)

| Check | Seeds | Result | Consequence |
|---|---|---|---|
| Valid-knockdown bonus, v0.0.22 base vs frozen **v0.0.13** | 99800–99823 | **22/24** (22 `self`, 2 `unresolved`) | **Saturated here too.** A drafted "train against v0.0.13" lever would fail its own saturation ceiling, so it was **discarded unregistered and never pushed**. |
| Non-striking contact ticks before the opponent's fall, base vs v0.0.19 and vs v0.0.13 | 99700–99723 | vs v0.0.19: median **376** (IQR 298–437), max 698. vs v0.0.13: median **66** (IQR 54–88), max 196. **Never 0.** | **A graded contact term carries real ranking signal.** |

## The one lever: a graded contact penalty (`protocols/v0.0.25.json`)

| | v0.0.24 | **v0.0.25** |
|---|---|---|
| Fitness | `stand/3000 + valid_knocked_down` | **`stand/3000 + valid_knocked_down − 0.5 · min(1, contact_ticks / 500)`** |
| `contact_ticks` | — | Non-striking inter-athlete contact (`ContactClassifier.blocking`) up to the opponent's first fall: the contact that leaves knockdowns `unresolved` |
| Unchanged | | Start policy (v0.0.22), training opponent (frozen v0.0.19), model, ES 48 × 12 (seed 15), the 0.5 m reset, ending v2 |

**Smoke gate, before any training**, on the v0.0.22 base against frozen
v0.0.19:

1. **The lever carries signal:** on 24 `motor_gate.trial` runs (seeds
   27000–27023), the penalty term must be nonzero in at least 10 % of
   trials and vary by at least 0.05 across them. The valid-knockdown share
   is reported.
2. **Validity:** a **48-game** head-to-head (seeds 27100–27123, both
   seats) must be **at least 90 % valid**. v0.0.24 used 12 games, which was
   too noisy.

Otherwise the result is `SMOKE_FAILED`: a non-zero exit and no training.

**Gates, in order, on new seeds:**

| Gate | Seeds | Rule |
|---|---|---|
| Primary against frozen v0.0.19 | 28000–28023 | At least 40/48 valid, then at least 24/48 decisive (both fail loud), then b and c |
| Hard gate against v0.0.13 | 29000–29023 | At least 40/48 valid |
| Monotonicity: @48 vs @16, and @16 vs the v0.0.22 base | 30000–30023 | At least 40/48 valid, then Wilson > 0.55 |

**Tag only if the smoke gate and every gate pass.**

**Registered red-team notes:**

- The penalty counts *all* non-striking contact (guard and glove too), so it
  may teach the policy to back away rather than box.
- The training opponent is still v0.0.19, while the hard gate is against
  v0.0.13.
- The smoke validity check still measures the starting policy.

## Code

| Change | Detail |
|---|---|
| `motor_gate.trial` | Now reports `contact_ticks` |
| `fitness` and `train` | A `contact_penalty` weight, default 0, so earlier modes are unchanged; `mean_contact_ticks` added to the history only when the weight is nonzero |
| Tests | Four penalty cases (none, half the window, capped, off), plus the key in a real short trial. A binary (non-graded) mutant fails. |
| Probe | `probes/cycle-0037-contact-penalty-gate.py`; smoke-tested on the pass path and a forced `SMOKE_FAILED` path, on seeds 99930 and above |

**No game has been played on seeds 27000–30023.**
