# Grokbot red-team — cycle 0018 / v0.0.14 (2026-10-28): engagement floor

Reviewed `origin/main` at `58229ec`. jekko has made no commits since
`1978da7`. The owner notes of 01:10, 01:30 and 01:50 set the bar for this
version:

- Fix engagement first, under a **preregistered valid-game floor of
  ≥ 50 %**, before any strike or Elo claim.
- Then show Elo against the v0.0.12 baseline with a lower bound above 0.
- Commit the criterion before reading results, and **tag only if it
  passes**.

## Preregistration

`grokbot/redteam/protocols/v0.0.14.json` (sha256 `ecb44090…`) was pushed at
`8b5dc2e` before training started.

| Item | Registered value |
|---|---|
| Stage 1 (non-degenerate league) | The generation's evaluation valid-result share is ≥ 0.50. A valid result is not a no-contest, not blocked, and not an engine fault. |
| Stage 2 (level-up) | In a stage-1 generation, the paired 95 % Hoeffding lower bound against frozen `v12base` is > 0 over 12 seed pairs, **and** the champion gate promotes with the engagement gate. `v12base` is the v0.0.12 champion: its gen0, which is the v0.0.11 gen4 parameters. |
| Levers (the owner's list) | A 0.50 m spawn, chosen because scoping put "blocked" at ≤ 0.45 m and "no contact" at ≥ 0.6 m. Training-only terms: valid-game bonus 1.0, no-contest penalty 0.5, strike bonus 1.0, trunk-relative extension shaping 0.5. `PunchPolicy` initialized from the baseline. |
| Pre-declared reading if stage 1 fails | Engagement cannot be fixed by spawn distance and training terms with the current motor primitives. The next lever is motor capability. |

## Result: 4 generations, 192 training and 840 evaluation games, 1,195 s on 4 nice-10 workers

| Gen | Training valid share | Training contact ticks | Evaluation valid share | No-contest | Engaged | vs v12base | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| 1 | 0.000 | 570 | **0.000** | 0.958 | 0.000 | 0 pairs | refuse |
| 2 | 0.000 | 260 | **0.000** | 0.857 | 0.000 | 0 pairs | refuse |
| 3 | 0.000 | 516 | **0.000** | 1.000 | 0.000 | 0 pairs | refuse |
| 4 | 0.000 | 0 | **0.000** | 1.000 | 0.000 | 0 pairs | refuse |

All 840 evaluation games failed to produce a result: 778 were no-contests
and 62 were blocked by body contact. They contained 6,110 contact ticks and
0 strikes. **Stage 1 fails in every generation.** Stage 2 never applies,
and **v0.0.14 is not a level-up. It is not tagged**, per the 01:50 rule.

## Findings

### F1 — High — The levers on the owner's list cannot create valid games with this motor

The training valid share was 0.000 in all 192 training games. So the
valid-game bonus and the no-contest penalty **never produced a gradient**.
Every perturbation scored the same, a no-contest at −0.5. The scoping probe
at 0.40–0.70 m (cycle 0017) had already found 0 valid outcomes in 32 bouts.
This run confirms the result under learning. The bodies either collide
(blocked) or stay apart (no-contest). The punch reaches about 0.2 m relative
to the trunk while the athlete falls backward, so a glove never meets a
target shell at speed.

This is the fourth version in a row, after v0.0.11–v0.0.13, in which league
design was not the binding constraint. **The binding constraint is a stance
that holds near the opponent, plus a targeted reach of about 0.4 m or
more.**

### F2 — Recommendation — What must change before v0.0.15 is preregistered

1. **Motor capability first, measured as a motor gate, not a league.**
   - Option (a): jekko's pretrained bridge. The first shared trial kept both
     athletes upright for 3.52 s, but walking. It would need a stationary
     facing stance and reach adaptation.
   - Option (b): a grokbot stance controller trained only to stay upright in
     the facing 0.5 m spawn for ≥ 3 s with the arms raised. That is a
     single-agent objective with a measurable gate.
2. **Only once a policy can deliver a glove to a target shell at ≥ 0.5 m/s
   active speed in ≥ 50 % of solo trials** should a v0.0.15 league be
   preregistered with the stage-1 valid-game floor.
3. **No new league runs on the current motor.** Each one is a guaranteed
   stage-1 failure.

## What shipped

| Item | Detail |
|---|---|
| `selfplay` | `valid_result`; training-only `valid_bonus` and `no_contest_penalty`; per-generation `eval_valid_share` and `train_valid_share` |
| Tests | 2: valid-result classification, and an exact ES update showing a valid bout outranks a no-contest |
| Evidence | The protocol, the probe, `data/cycle-0018-engagement-floor-summary.json` and this report |
