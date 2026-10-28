# Grokbot red-team — cycle 0035 (2026-10-28): v0.0.24 preregistration (valid knockdowns, smoke-gated)

Branches from `origin/main` at `2f3c9ec`, independently of the v0.0.23
report PR (#34): owner 18:40 says "not stacked" means off main, not one PR
at a time. To avoid conflicts it changes no STATUS or CHANGELOG text. This
is the **preregistration on its own; the run starts only after merge.**

## Why this lever (cycle 0034)

v0.0.23's `clean_knocked_down` never fired, because its knockdowns of
v0.0.19 are attributed `self`. A replay of the v0.0.22 policy against
v0.0.19 gave 22 `self`, 2 `unresolved` and 0 `knockdown` falls out of 24:
the single hit lands at tick 252 and the fall comes 1–2 s later, outside
the 0.5 s window.

Under ending v2, `self` falls are **valid** wins, because the faller loses.
Only `unresolved` falls are **blocked**, and those are exactly the games
that broke v0.0.22's validity floors.

## The lever (`protocols/v0.0.24.json`)

| | v0.0.22 | **v0.0.24** |
|---|---|---|
| Bonus | `knocked_down` | **`valid_knocked_down`**: `knocked_down` **and** the fall is not `unresolved` |
| Starting policy | v0.0.19 | **v0.0.22's final parameters**, the only policy that produces knockdowns. v0.0.23, starting from v0.0.19 with a bonus that never fired, regressed to standing. |
| Unchanged | | Training opponent (frozen v0.0.19), model, ES 48 × 12 (seed 14), the 0.5 m reset, ending v2 |

## Smoke gate (owners 18:40 and 19:50), the first stage, before any training

Owner 19:50 called v0.0.23's miss an instrumentation bug in the bonus path.
That path is now fixed (`valid_knocked_down`), and two checks run on the
**v0.0.22 base policy against frozen v0.0.19** before the 48-iteration
run:

1. **The bonus fires on its real path.** On 24 `motor_gate.trial` runs,
   the exact code the fitness bonus reads, on fresh seeds 19000–19023,
   `valid_knocked_down` must fire **at least once** and in **at least
   10 %** of trials.
2. **Validity.** A **12-game head-to-head** under the real ending (seeds
   19100–19105, both seats, 0.5 m, anti-stalling, ending v2) must be **at
   least 90 % valid**.

If either check fails, the result is **`SMOKE_FAILED`**: a non-zero exit,
no training and no gates, with both censuses saved.

## Gates, in order, on new seeds (as in v0.0.22 and v0.0.23)

| Gate | Seeds | Rule |
|---|---|---|
| Primary against frozen v0.0.19 | 20000–20023 | At least 40/48 valid, then at least 24/48 decisive (both fail loud), then b and c |
| Hard gate against v0.0.13 | 21000–21023 | At least 40/48 valid |
| Monotonicity: @48 vs @16, and @16 vs the **v0.0.22 base** | 22000–22023 | At least 40/48 valid, then Wilson > 0.55 |

**Tag only if the smoke gate and every gate pass.** All results are split
by seat.

**Registered red-team notes:**

- **The bonus is near-saturated at the base** (about 22/24), so the learning
  pressure falls on the blocked trials and on stand time.
- **The opponent mismatch:** training is against v0.0.19, but the hard gate
  is against v0.0.13.
- **`self` wins are causally weak:** the opponent topples 1–2 s after one
  hit.

## Code

| Change | Detail |
|---|---|
| `motor_gate.valid_knocked_down` and a `valid_knockdown` flag | Existing modes unchanged; `valid_knockdown_share` added to the history only when the flag is on |
| Tests | 4 cases: `self` pays, `knockdown` pays, `unresolved` does not, and no fall does not. A mutant that ignores `unresolved` fails. |
| Probe | `probes/cycle-0035-valid-knockdown-gate.py`, which runs the smoke gate first |

The smoke test on seeds 99950 and above checked three things:

- the pass path, where the bonus fired 2/2, the head-to-head was 2/2 valid,
  and training and the gates ran;
- the `SMOKE_FAILED` path, with an unreachable validity threshold, which
  exits non-zero with no training after saving both censuses;
- the fail-loud primary path.

**No game has been played on seeds 19000–22023.**
