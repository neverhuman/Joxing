# Grokbot red-team — cycle 0031 (2026-10-28): v0.0.22 preregistration (knockdown objective)

Reviewed `origin/main` at `5e8128c`. jekko has made no new commits. Per
owners 13:50, 14:05 and 14:45, this PR lands the v0.0.22 preregistration **on
its own, before any run**.

## The one lever: a knockdown objective (`protocols/v0.0.22.json`)

| | v0.0.21 | **v0.0.22** |
|---|---|---|
| Fitness bonus | `hit_and_stayed_up(500)` | **`knocked_down`**: the candidate's first head/torso hit comes before the opponent's first fall, and the opponent falls before the candidate (`first_scoring_tick < opponent_fall_tick < stand_ticks`) |
| Unchanged | | The stand term, the training opponent (frozen v0.0.19), `GroupPDPolicy`, ES 48 × 12, the 0.5 m reset and ending v2 |
| Initialization | v0.0.19 | v0.0.19, the last level-up, and **not** the regressed v0.0.21 |

**Why** (cycle-0030 F1). All 42 of v0.0.21's draws reached the bell with
referee points [0, 0], although the candidate hit the head in every one.
Ending v2 decides by the first fall, so a draw means nobody was knocked
down.

A bell **tiebreak** on grokbot-classified hits was considered and
**rejected**. It would reward the feather taps that jekko's referee does not
score: it would move the ruler to make our policy win. The knockdown lever
does not depend on referee points, so the referee-points feasibility probe
is not a precondition for it. That gap stays an open finding for jekko.

## Criteria, checked in order (owner 14:45)

**Primary: frozen v0.0.19 on new seeds 13000–13023, 48 games.**

1. **Validity: at least 40/48 valid**, or the verdict is `VALIDITY_FAILED`
   (fail loud, non-zero exit).
2. **Decisiveness: at least 24/48 decisive** games (won or lost, not drawn),
   or the verdict is `DECISIVE_FAILED` (fail loud).
3. **Win rate:** (b) Hoeffding lower95 > 0, and (c) at least half of the
   wins include the candidate's own head/torso hits. If both hold the
   verdict is `PRIMARY_PASSED`; otherwise `CRITERION_NOT_MET`.

**Hard gate:** at least 40/48 valid games against frozen v0.0.13 on seeds
14000–14023. This fixes v0.0.21's 46/48 blocked.

**More training beats less:** v0.0.22@48 vs @16, and @16 vs v0.0.19, on
seeds 15000–15023. Each needs at least 40/48 valid games, then a Wilson
lower bound above 0.55 on the more-trained side's win rate. The probe keeps
the @16 checkpoint by training 16 + 32 iterations on one generator. A smoke
test showed this is **bit-identical** to a single call.

**Tag `v0.0.22` only if** the primary is `PRIMARY_PASSED`, **and** the hard
gate passes, **and** both monotonicity matchups pass.

**Reporting:** every head-to-head is split by seat, with wins, losses,
draws, no-contests, blocks and engine faults counted separately.
**Secondary:** the v0.0.19 mirror on the primary seeds, the candidate
against v0.0.16, the motor gate (with v0.0.19 re-checked per trial), and
`knockdown_share` per training iteration.

## Code

| Change | Detail |
|---|---|
| `motor_gate.knocked_down` | New |
| `fitness` and `train` | A `knockdown` flag; existing modes are unchanged |
| History | `knockdown_share`, only when the flag is on, so earlier histories keep their exact format for reproduction checks |
| Tests | 7 boundary cases: knockdown while standing; knockdown then falling; falling first; the same tick; the opponent falling before any hit; never hitting; a bell draw. Two mutants (dropping "opponent first", dropping "hit before the fall") fail. |

**Probe:** `probes/cycle-0031-knockdown-gate.py`. It checkpoints after every
stage.

The smoke test, on seeds 99980 and above, caught a real ordering bug: Python
evaluates the right-hand side of the assignment before calling `play`, so
`report[key]` did not exist yet. That is fixed. The smoke test also
exercised the fail-loud exit.

**No game has been played on seeds 13000–15023.**
