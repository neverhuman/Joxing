# Grokbot red-team — cycle 0036 / v0.0.24 (2026-10-28): SMOKE_FAILED, the run stopped before training

Reviewed `origin/main` at `f4505d9`, the v0.0.24 preregistration (#35,
protocol `c428442d…`). Main CI on `f4505d9` was green before launch, as
owners 19:50, 20:15 and 20:25 required.

## Attempts (every attempt counts)

| Attempt | When | Outcome |
|---|---|---|
| 01 | ~20:09Z | **Aborted by me within about 1 minute, before any game.** I issued the launch and the load check as two parallel calls, so the launch was not gated on the check, and the check had **failed** (load 84.68). Only my own processes were stopped. The empty directory is kept as `runs/grokbot/valid-knockdown-gate-0035-attempt01-aborted-load`. |
| 02 | 20:24:53Z, load 64.08 | The load check and the launch ran as **one command**, which waited for load below 70. Result: **`SMOKE_FAILED`**, probe exit 1. |

## v0.0.24 — preregistered smoke gate **failed**; no training, no gates, not tagged

**Bonus check, on 24 `motor_gate.trial` runs of the v0.0.22 base against
frozen v0.0.19 (seeds 19000–19023): passed.** `valid_knocked_down` fired
**23 of 24** times, against a requirement of at least 1 and at least 10 %.
All 23 falls were attributed `self`; the remaining trial had no fall. **The
fixed bonus path works**, which answers owner 19:50's instrumentation
concern.

**Validity check, on a 12-game head-to-head of the base against frozen
v0.0.19 under ending v2 (seeds 19100–19105, both seats): failed.**

| Seat | Games | W–L–D | No-contest | Blocked | Fault |
|---|---:|---|---:|---:|---:|
| A | 6 | 5–0–0 | 0 | **1** | 0 |
| B | 6 | 5–0–0 | 0 | **1** | 0 |
| **Total** | 12 | **10–0–0** | 0 | **2** | 0 |

Valid share **10/12 = 0.83**, below the required 0.90. Every valid game was
a win with hits.

By the registration, `SMOKE_FAILED` means **no training, no gates and no
tag**. The v0.0.19 gate re-check reproduced exactly.

## Red-team reading

1. **This gate measures the starting policy, not the lever.** The validity
   check runs the *base* (v0.0.22) before any training. Its blocked rate is
   the very failure the lever is meant to train away, so the gate can fail
   the lever before the lever has had any chance to act. The gate was
   specified in advance and stands. **v0.0.25 should gate the instrumentation
   (bonus firing, valid outcomes recordable) separately from the base's
   blocked rate.**
2. **Twelve games is a small sample.** v0.0.22 was 46/48 (96 %) valid
   against v0.0.19. 10/12 is well within the noise of 12 games: the Wilson
   95 % interval on 10/12 is roughly 0.55–0.95. The threshold was registered
   and is applied as written.
3. **Blocked games remain the binding failure.** They are 2 of 12 here, 12
   of 48 in v0.0.22's hard gate, and 30 of 48 in v0.0.23's.

## Next (v0.0.25, its own preregistration PR)

Keep a smoke gate on the **lever's instrumentation**: does the bonus fire,
and are outcomes recorded? Measure the base's validity as a **baseline**
rather than a launch precondition, and keep the validity floors on the
**trained** policy (at least 40/48 primary and at least 40/48 hard gate).
The lever stays the valid-knockdown bonus from v0.0.22.

## What this PR adds

- `data/cycle-0035-valid-knockdown-smoke-summary.json`: both smoke checks,
  the attempts ledger, and the report SHA-256.
- This report, the CHANGELOG entry and STATUS.
