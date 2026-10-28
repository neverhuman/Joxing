# Grokbot red-team — cycle 0030 / v0.0.21 (2026-10-28): validity reached, by draws; training is not monotone

Reviewed `origin/main` at `5b3cf20`. jekko has made no new commits. The
v0.0.21 protocol (`997972f3…`) landed in #29 before the run, which used
merged main's tree `a75b705d`, ran at nice 19 and took 3,516 s.

The owner-13:10 compute-monotonicity **tag precondition**
(`protocols/v0.0.21-monotonicity-addendum.json`, `a75ec5db…`) and its probe
were pushed to this branch as `c5d6218`, **before any v0.0.21 gate result
was read**. The owner-13:35 seat-swap reporting was added as `c714c40`
before the monotonicity check ran.

## v0.0.21 — verdict **CRITERION_NOT_MET**, not tagged

The one lever was the training opponent: frozen v0.0.19 instead of the gen4
posture, with the same hit-and-stay-up fitness.

**Primary: frozen v0.0.19, seeds 10000–10023, 48 games.**

| | Valid | W | L | D | No-contest | Blocked | Fault | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Validity precondition (≥ 40)** | **48** | | | | | | | **met** |
| All games | 48 | 4 | 2 | **42** | 0 | 0 | 0 | |
| Seat A | 24 | 2 | 1 | 21 | 0 | 0 | 0 | |
| Seat B | 24 | 2 | 1 | 21 | 0 | 0 | 0 | |
| (b) lower95 > 0 | −0.46 | | | | | | | **failed** |
| (c) at least half of wins with hits | 4/4 | | | | | | | passed |

The verdict is `CRITERION_NOT_MET` and `level_up = false`. Seat A and seat B
are identical, so there is no seat bias.

**Secondaries:**

| Matchup (48 games) | Valid | W–L–D | No-contest | Blocked |
|---|---:|---|---:|---:|
| v0.0.19 mirror, primary seeds (baseline) | 3 | 0–0–3 | 43 | 2 |
| vs frozen v0.0.16, seeds 11000–11023 | 48 | 48–0–0 | 0 | 0 |
| **vs frozen v0.0.13, seeds 11000–11023** | **0** | — | 2 | **46** |

**Motor gate against gen4:** stance **1/24**, reach 19/24, hits 10/24,
median stand **1.86 s**, against v0.0.19's 16/24, 24/24, 24/24 and 3.0 s.
v0.0.19's gate re-ran identically per trial.

**Training against v0.0.19:** the hit-and-stay-up share rose from 0.46 to
0.92. ES learned its own objective.

## Compute monotonicity (owner 13:10/13:35) — **NOT_SHOWN_MONOTONE**

The iteration-16 checkpoint was re-trained from seed 11 and **reproduced the
main run's history exactly**.

| More vs less (48 games each, seeds 12000–12023) | Valid | W–L–D | No-contest | Blocked | Seat A W–L–D | Seat B W–L–D | Verdict |
|---|---:|---|---:|---:|---|---|---|
| **@48 vs @16** | 20 | **0–20–0** | 1 | 27 | 0–10–0 | 0–10–0 | unmeasurable (< 40 valid) |
| @16 vs @0 (= v0.0.19) | 14 | 2–2–10 | 0 | 34 | 1–1–5 | 1–1–5 | unmeasurable (< 40 valid) |

Even though the precondition makes both matchups unmeasurable, **all 20
valid games between the 48- and 16-iteration policies were won by the
less-trained one**, the same in both seats. More training made the policy
worse in matches.

## Findings

### F1 — High — "Valid" was reached by draws: the referee does not score these hits

All 42 draws end at the bell, tick 3,000. **The candidate landed
head/torso hits in all 42**, and v0.0.19 hit back in 14. I replayed one
draw (seed 10001, seat A) deterministically: **referee points [0, 0]**,
against 50 hit ticks per side under grokbot's classifier.

jekko's referee does not score hits of this force, which matches the open
"feather-tap ceiling" finding. So the validity precondition measured *not
stalled*, not *decisive* (owner 13:50).

### F2 — High — Training is anti-monotone in match strength

The solo objective improved from 0.46 to 0.92, but the 48-iteration policy
lost **every** valid game to its own 16-iteration checkpoint. It also
regressed on the motor gate (stance 16/24 → 1/24), and **against v0.0.13
it went from 48–0 to 46/48 blocked**.

Training against a standing v0.0.19 taught the policy to make contact that
ends in clinches (blocked) and bell draws, not knockdowns. This is the
agent-adequacy failure in Joxing form: **a training budget that does not
monotonically improve match play cannot certify any level-up.**

### F3 — Medium — Both pre-declared readings missed the measured cause

The registered reading for `CRITERION_NOT_MET` was "longer training against
v0.0.19". F2 shows longer training makes play *worse*. It is reported as
registered and not followed.

## v0.0.22 requirements (owner 13:50), to be preregistered in its own PR

- **One lever.** The data point at **referee-scored hits**: the strike bonus
  should require hits the official referee scores, not grokbot-classified
  touches. That comes only **after a feasibility probe** showing these
  athletes can score referee points at all. If they cannot, that is a
  finding for jekko, not a lever.
- **A decisive-games floor:** at least 24/48 decisive (won or lost) games
  against frozen v0.0.19, added to the criterion.
- **Hard gate:** the v0.0.13 regression must be fixed, with at least 40/48
  valid games against frozen v0.0.13.
- **Monotonicity:** the 3N-vs-N check runs again.
- **Reports:** the seat-swap split, with draws, no-contests, blocks and
  faults counted separately.

## What this PR adds

- `data/cycle-0029-mirror-opponent-gate-summary.json`:
  - v0.0.21 per-trial and per-game records, with the seat breakdown;
  - the monotonicity matchups;
  - the referee-points replay;
  - the training history and final parameters;
  - SHA-256 hashes of the full reports.

  The iteration-16 parameters are reproducible and not stored.
- The monotonicity addendum and probe, both pushed before any result was
  read.
- This report and STATUS.
