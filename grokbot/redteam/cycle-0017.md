# Grokbot red-team — cycle 0017 / v0.0.13 (2026-10-28): punch primitive

Reviewed `origin/main` at `deda861`. jekko has made no commits since
`1978da7`. Owner 00:12 asked that v0.0.13 "target a real measurable level-up:
preregister a strike-emergence metric and an Elo gain vs v0.0.11 frozen
opponent, with CI lower bound > 0". The `v0.0.11` GitHub release was also
published this cycle, as a pre-release with the Elo-per-generation table
(owner 23:06).

## Motor feasibility, measured before preregistration

**Reach synergy discovery.** Each of the 56 DynSyn groups made up entirely
of arm muscles was driven alone for 0.2 s at 0.7 m.

| DynSyn group | Hand | Forward vs trunk | Up |
|---|---|---:|---:|
| 58 | right | +0.197 m | +0.178 m |
| 81 (mirror of 58) | left | +0.197 m | +0.179 m |
| 57 | right | +0.156 m | |
| 80 (mirror of 57) | left | +0.156 m | |

Every other arm group moves the hand less than 0.05 m. Pair 58/81 is
shoulder flexion at about 1 m/s.

**Fixed punches.** gen4's posture control was combined with the reach
groups at onsets of 0, 200 and 400 ticks, durations of 200 and 400 ticks,
one or both arms, at 0.7 and 0.6 m. Result: **0 strikes**. The closest
approach of a hand to the opponent's head or trunk origin was 0.596 m, with
a punch at t = 0 at 0.6 m. The gloves stay about 0.35–0.45 m short of the
target shells. Later punches do nothing, because the native keyframe falls
backward.

## Preregistration

`grokbot/redteam/protocols/v0.0.13.json` (sha256 `03fc8e70…`) was pushed at
`7185f19` before the run.

| Item | Registered value |
|---|---|
| Policy | `PunchPolicy` = gen4 posture plus a learned per-arm punch (onset, duration, amplitude on groups 57/58 and 80/81). Initialized with an early punch: about 28 ticks onset, 300 ticks duration, amplitude 0.76. |
| Shaping (training only) | Trunk-relative hand **extension**, weight 0.5. A whole-body lean scores 0, which closes the v0.0.12 exploit. Plus a strike bonus of 1.0. |
| League | Frozen v0.0.11 gen4 joins the anchors and the evaluation panel. 12 seed pairs. Range 0.60 m, anti-stalling. |
| Criterion | Some generation lands strikes in ≥ 25 % of evaluation bouts **and** beats frozen gen4 with a paired one-sided 95 % Hoeffding lower bound > 0. |

## Result: 6 generations, 288 training and 1,344 evaluation games, 1,979 s on 4 nice-10 workers

| Gen | Training contacts | Training fitness (shaping only) | Median bout | Evaluation strike share | Valid evaluation results | vs v11 gen4 (pairs, lower95) | Gate |
|---|---:|---|---:|---:|---:|---|---|
| 1 | 32 | 0.136 → 0.135 | 1.46 s | 0.000 | 0 | 0 pairs, — | refuse |
| 2 | 0 | 0.128 → 0.140 | 1.34 s | 0.000 | 0 | 0 pairs, — | refuse |
| 3 | 0 | 0.138 → 0.139 | 1.30 s | 0.000 | 0 | 0 pairs, — | refuse |
| 4 | 113 | 0.140 → 0.142 | 1.38 s | 0.000 | 0 | 0 pairs, — | refuse |
| 5 | 0 | 0.154 → 0.164 | 1.36 s | 0.000 | 0 | 0 pairs, — | refuse |
| 6 | 0 | 0.169 → 0.190 | 1.32 s | 0.000 | 0 | 0 pairs, — | refuse |

Of the 1,344 evaluation games, 1,330 were no-contests and 14 were blocked by
body contact. There were **0 valid results and 0 landed strikes**. The punch
parameters barely moved: right onset 23 ticks and left 33 ticks, durations
264 and 289 ticks, amplitudes 0.76 and 0.72. Fitness rose only through the
extension shaping. A strike never occurred, so the strike bonus never
produced a gradient.

**Against the preregistered criterion, v0.0.13 is not a level-up.**
Pre-declared failure reading 2 applies: *the punch primitive cannot close the
reach deficit*.

## Findings

### F1 — High — The motor bottleneck is the boxing bottleneck

Three versions in a row have now shown the same thing from different sides:

- v0.0.11: standing.
- v0.0.12: leaning.
- v0.0.13: reaching.

Untrained and ES-tuned athletes fall backward within about 1.5 s, and a
shoulder-flexion punch reaches about 0.2 m. Neither rules, shaping nor league
design can make strikes appear until an athlete can hold a stance near the
opponent and extend a glove 0.4 m or more.

**Next lever (pre-declared, unchanged):**

- A forward step or lean primitive that keeps balance.
- Or jekko's pretrained-motor bridge (`docs/PRETRAINED_SHARED_MOTOR_BRIDGE.md`)
  with a stationary stance and a reach adaptation. The first shared trial
  already keeps both athletes up for 3.52 s.

grokbot should not run more league variants on top of the current motor
capability. They are guaranteed no-contests.

### F2 — Medium — Sparse terms cannot bootstrap the skill

A strike bonus pays nothing until a strike happens. The extension shaping
paid for arm movement that never got near a target. Any future shaping needs
a signal along the whole path to contact, for example glove-to-target-shell
distance computed from **glove** geometry with lean excluded. Bonuses that
fire only on success are not enough.

### F3 — Positive — The measurement stack is ready for the first real strike

The engagement gate, anti-stalling, the frozen-opponent paired lower bound,
the champion gate, side-swapped common random numbers, fault quarantine and
preregistration all ran end to end on 1,344 games without a crash. Engine
faults: 0 this run, with quarantine in place. The first policy that lands a
strike will be measured honestly.

## What shipped

| Item | Detail |
|---|---|
| `grokbot/motor_primitives.py` | `discover_reach`/`reach_effect`, `PunchPolicy`, `punch_params`, `active_extension` |
| `selfplay` | A punch policy kind, a selectable shaping key, a training-only strike bonus, frozen opponents as anchors and evaluation targets, and a per-generation `vs_frozen` paired lower bound |
| Tests | 5, covering the mirror reach pair, a leg group leaving the hands still, punch-window isolation, lean-proof extension, and a league smoke test against a frozen champion |
| Evidence | The protocol, the league probe, `data/cycle-0017-punch-summary.json` and this report |

## Owner direction for v0.0.14 (01:10 note)

v0.0.14 must first **fix engagement** under a preregistered valid-game-rate
floor (for example ≥ 50 % valid results), using spawn distance, an approach
reward or a no-contest penalty. Strike or Elo claims come only after that
floor holds, followed by Elo against frozen v0.0.11 with a lower bound > 0.

Given F1, the levers that can plausibly raise the valid-game rate are:

- a closer spawn, between the 0.6 m here and the 0.2 m stress overlap;
- a stance primitive, so the athletes stop falling backward;
- a no-contest penalty, which turns stalling into a loss for both corners.
  It is not zero-sum, so it is training-only.

Tags: `v0.0.12` was pushed on `deda861` at the owner's request and published
as a pre-release. This PR adds CHANGELOG entries for v0.0.12 and v0.0.13.
