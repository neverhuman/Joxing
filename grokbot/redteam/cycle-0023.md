# Grokbot red-team — cycle 0023 / v0.0.17 (2026-10-28): first 3 s stances, glove-tap strikes; v0.0.18 preregistered

Reviewed `origin/main` at `508cd94`. jekko has made no new commits. The
v0.0.17 protocol (`850ec810…`) was on main before the run. The run's source
tree `97502709` is identical to main `508cd94`'s tree, as checked after #22
merged.

## v0.0.17 release notes — **criterion not met** (pre-release at owner request)

The candidate was `GroupFeedbackPolicy`: the v0.0.16 group controller plus
per-group tilt gains, 1,112 parameters. It was trained for 48 ES iterations
of 12 pairs from the v0.0.16 final parameters, with the same gate, seeds,
opponent and fitness. The run took 2,921 s at nice 10, reniced to 19 at
05:15Z.

| | Stance pass | Reach pass | Median stand | Stand range |
|---|---:|---:|---:|---|
| gen4 baseline | 0/24 | 0/24 | 1.464 s | — |
| v0.0.16 (re-run; **identical per trial** to the committed summary) | 0/24 | 0/24 | 2.344 s | 2.14–2.77 s |
| **v0.0.17** | **7/24 (0.29)** | **24/24 (1.00)** | **2.682 s** | 2.30–**3.00 s** |

- **Primary motor gate** (stance ≥ 0.5 **and** reach ≥ 0.5): **failed on
  stance.** Seven trials stood the full 3 s, the first ever.
- **Preregistered head-to-head** against the frozen v0.0.13 champion: **not
  run**, because the protocol makes it conditional on the gate.
- **Preregistered secondary:** the exact sign test against v0.0.16
  **passed**, with 23 of 24 trials better (p = 1.5×10⁻⁶). The Hoeffding
  bound did not: the mean gain was +0.35 s and the lower bound −1.15 s.
- **`level_up = false`: not a level-up.** Owners 05:46 and 06:10 asked
  for v0.0.17 to ship as a release honestly labelled pass or fail. Like
  v0.0.13, it is therefore a GitHub **pre-release titled "criterion not
  met"**, tagged on this PR's merge with a CHANGELOG entry. It is not a
  level-up tag. Tag `v0.0.18` only if its own `level_up` is true.
- **Training:** the strike share rose from 0.04 to 0.67–0.92 over the last
  10 iterations, and the mean stand time in training reached about 2.1–2.4 s.
  4 engine faults were quarantined.

### Exploratory, after the decision, not counted

On the protocol's pinned head-to-head seeds (2000–2011, both sides), v0.0.17
against the frozen v0.0.13 champion: **14 of 24 games valid, and v0.0.17 won
all 14.** That is 7 paired samples, mean +1.0, lower bound +0.075. **These
are the project's first valid, decided games.** v0.0.12–v0.0.14 had 0 of
2,772. F1 and F2 below explain what these wins are.

## Findings

### F1 — High — Every v0.0.17 "strike" is a punch into the opponent's glove or guard

I replayed 6 gate seeds (1000, 1003, 1007, 1011, 1015, 1020) and recorded
what A's striking glove touched on each strike tick. In **all 6**, the
strike run (ticks ~570–640) is A's glove on **`B_glove_l`** (50 ticks), and
in 4 of them also on `B_shell_ulna_l`, the forearm guard. There was **no
head or torso contact in any seed.** Attribution v2 counts glove and guard
as strike targets, because a punch into a guard is boxing, so the reach
pass is legitimate *as registered*. But "reach 24/24" means "punches the
opponent's left glove", not "hits the opponent".

### F2 — High — Anti-stalling is satisfied by a blocked punch, and the fall decides the game

All 24 exploratory head-to-head games end at ticks 1,600–1,720, which is
when the v0.0.13 champion falls *on its own*. Its solo fall time is about
1.3–1.7 s. The 14 "wins" are games where v0.0.17's glove tap at ~0.6 s made
the bout non-stalling before that spontaneous fall. The 10 no-contests are
seeds where no tap happened.

So a valid ending v2 result can be a blocked punch followed by the
opponent's own fall. That measures balance plus one glove touch, not
punching. Any Elo gate built on ending v2 alone would reward it. **v0.0.18
closes this in its gate (below) without changing the ending variant.**

## What this PR adds

| Change | Detail |
|---|---|
| `endings.ContactClassifier.scoring_by` | Per-corner ticks on which a striking glove touched a **head or torso** target (`SCORING_TARGETS`). Reporting only: ending v2 decisions are **unchanged**. |
| Exposed by | `selfplay.play_game` (`scoring_strike_ticks`, `opponent_scoring_strike_ticks`), `motor_gate.trial` (`scoring_strike_ticks`), `motor_gate.gate` (`scoring_reach_rate`), ES history (`scoring_share`) |
| `motor_gate.fitness` and `train` | Optional `strike_key`; the default keeps the old behaviour |
| `motor_primitives.GroupPDPolicy` and `group_pd_params` | Per-group gains on pelvis linear and angular velocity and height: 138 × 7 = 966 parameters. Zero gains give exactly the v0.0.17 commands. Controller kind `group_pd`. |

**Tests:**

- 4 scoring classification cases: head and torso count, glove and slow
  contacts don't. A mutant that counts glove contacts as scoring fails.
- Scoring fitness and `scoring_reach_rate` in the motor-gate tests.
- `GroupPDPolicy` equivalence: exact with zero gains, including the noise
  stream. One gain moves only its own group.

**Evidence added:**

- `data/cycle-0022-group-feedback-gate-summary.json`: v0.0.17 per-trial
  results, training history, final parameters, full-report SHA-256, and the
  exploratory head-to-head per game.
- The v0.0.18 protocol and probe.

## v0.0.18 preregistration (protocol `protocols/v0.0.18.json`, before any run)

**Change from v0.0.17:**

1. **Damping.** Per-group velocity and height feedback (`GroupPDPolicy`,
   2,078 parameters) from the v0.0.17 final parameters. v0.0.17 had
   proportional tilt feedback only.
2. **A hit-only strike bonus.** The fitness is `stand/3000 + 1·[head or
   torso hit]`, so glove taps no longer earn the bonus.

Training is 48 iterations of 12 pairs, seed 8, at nice 19.

**Level-up = the motor gate AND the Elo gate.**

- **Motor gate:** unchanged. Stance ≥ 0.5 and reach ≥ 0.5 over the same 24
  seeds. `scoring_reach_rate` is also reported.
- **Elo gate:** against the frozen v0.0.16 champion. It **always runs**,
  whatever the motor gate result. 24 fresh seed pairs (3000–3023, 48 games),
  0.5 m, anti-stalling, ending v2. It passes only if **all** of these hold:
  - (a) valid games ≥ 50 % of all 48;
  - (b) the paired Hoeffding lower bound is > 0, which is the same as an Elo
    lower bound > 0;
  - (c) at least half of the candidate's valid wins include a head or torso
    hit by the candidate. **This closes F2.**

  Elo is reported as 400·log₁₀(p/(1−p)) with p = (1 + mean)/2, together with
  its lower bound, the win rate and a Wilson interval.
- **Secondary:** exact sign test on stand ticks against the v0.0.17
  candidate, which is re-run per trial against the committed summary as a
  determinism check.

**Tag `v0.0.18` only if `level_up`.**

Pre-declared readings cover each failure mode. For example, if (b) passes
but (c) fails, the wins are glove taps and **not a level-up**.

Before preregistration, no v0.0.18 head-to-head was played on any
preregistered seed. The probe's smoke test used seed 99999, which is outside
the protocol.
