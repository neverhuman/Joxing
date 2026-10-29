# Changelog

## Unpublished versions (clarification, owner 2026-10-29 01:10)

**The latest release is still v0.0.19**, the last preregistered level-up. Every grokbot version since then missed its preregistered criterion, so none was tagged or released. The one exception is v0.0.17, which got a pre-release labelled "criterion not met" at the owner's request before v0.0.19. Each untagged version is recorded here and in `grokbot/redteam/cycle-*.md`:

| Version | Lever | Why it was not released | Report |
|---|---|---|---|
| v0.0.20 | Two-player self-play league | No promotion; primary 3/48 valid | cycle 0028 |
| v0.0.21 | Train against frozen v0.0.19 | 42/48 bell draws; not monotone | cycle 0030 |
| v0.0.22 | Knockdown objective | Hard gate 36/48 (blocked); monotonicity | cycle 0032 |
| v0.0.23 | Clean-knockdown bonus | `VALIDITY_FAILED`; the bonus never fired | cycle 0034 |
| v0.0.24 | Valid-knockdown bonus | `SMOKE_FAILED`; base head-to-head 10/12 valid | cycle 0036 |
| v0.0.25 | Graded contact penalty | Hard gate 0/48 (blocked); monotonicity | cycle 0038 |
| v0.0.26 | Both training opponents | Monotonicity (@16 lost 2–40 to base) | cycle 0040 |
| v0.0.27 | Smaller ES step (lr 0.02) | Monotonicity (@16 lost 0–48 to base; @48 lost 11–37 to @16) | cycle 0042 |
| v0.0.28 | Frozen start as a self-play training opponent | Monotonicity (@16 lost 2–46 to base; @48 beat @16 46–2) | cycle 0044 |

## v0.0.28 — 2026-10-29 (preregistered level-up NOT met; not tagged)

grokbot PR #43 (preregistration) plus the cycle-0044 report.

- **Lever:** the frozen v0.0.22 start joins the training opponents; ES pairs cycle through v0.0.19, v0.0.13 and v0.0.22.
- **Smoke gate, primary and hard gate: passed.** The primary vs v0.0.19 went 44–0 with 4 blocked; the hard gate vs v0.0.13 went 48–0.
- **Monotonicity: FAILED.** @48 beat @16 46–2, the first such pass, but @16 lost **2–46** to the base.
- **Diagnostic (post hoc, non-gating):**
  - The training trial agrees with head-to-head, which refutes the registered "match-outcome term" reading.
  - ES traded the base matchup for the frozen opponents early, then recovered. @48 beats the base 43–5 post hoc, which does not count as a gate.
  - Measured: the base received 0.23–0.28 of the ES update, against a nominal 0.33. Per-opponent ranks would only lift it to about 0.31.
- **Next (v0.0.29):** the base faces half of the ES pairs.

## v0.0.27 — 2026-10-29 (preregistered level-up NOT met; not tagged)

grokbot PR #41 (preregistration) plus the cycle-0042 report.

- **Lever:** ES lr 0.05 → 0.02 on the v0.0.26 recipe.
- **Smoke gate, primary and hard gate: all passed.** Both the primary vs v0.0.19 and the hard gate vs v0.0.13 went 48–0.
- **Monotonicity: FAILED.** @16 lost **0–48** to the v0.0.22 base, and @48 lost **11–37** to @16.
- **Attempts:** attempt 01 was lost at training iteration 18 by operator error. SIGSTOP-pausing during a load spike orphaned the run, and it was killed by SIGHUP. Attempt 02 completed.
- **Finding:** the collapse against the policy's own start is not a step-size effect. The objective never rewards beating an opponent of the agent's own strength.
- **Next (v0.0.28):** add the frozen base as a self-play training opponent.

## v0.0.26 — 2026-10-29 (preregistered level-up NOT met; not tagged)

grokbot PR #39 (preregistration) plus the cycle-0040 report.

- **Lever:** train the v0.0.25 recipe against **both** frozen opponents, v0.0.19 and v0.0.13, alternating per ES pair. The smoke gate passed.
- **Primary vs v0.0.19: PASSED.** 46–0, all with hits.
- **Hard gate vs v0.0.13: PASSED for the first time.** 48/48 valid, 48–0; v0.0.25 had 0/48.
- **Monotonicity: FAILED.** @16 lost **2–40** to the v0.0.22 base, and @48 vs @16 was **24–24**.
- **Motor gate:** stance 11/24, median stand 2.82 s.
- **Next (v0.0.27):** a smaller ES step, lr 0.05 → 0.02.

## v0.0.25 — 2026-10-28 (preregistered level-up NOT met; not tagged)

grokbot PR #37 (preregistration) plus the cycle-0038 report. The last level-up is still **v0.0.19**.

- **Lever:** a graded contact penalty, `−0.5 · min(1, contact_ticks/500)`, starting from v0.0.22 and trained against frozen v0.0.19. The smoke gate passed: the penalty varied, and the 48-game base head-to-head was 48/48 valid.
- **Primary vs frozen v0.0.19: PASSED.** 48/48 valid, **48–0**, all wins with hits, Elo +669 (lower bound +191).
- **Hard gate vs v0.0.13: FAILED.** **0/48 valid: all 48 blocked.** Monotonicity also failed: @16 lost 4–44 to the v0.0.22 base, and @48 vs @16 had only 38/48 valid.
- **Contact rate vs v0.0.19:** it fell from 353 to 135 ticks in training while the valid-knockdown share stayed at 0.75–0.83. Primary games had median contact 113 against the base's 380. This is a real reduction, not avoidance gaming, but it **does not transfer** to v0.0.13, which training never saw.
- **Next (v0.0.26):** alternate both training opponents (v0.0.19 and v0.0.13).

## v0.0.24 — 2026-10-28 (preregistered smoke gate FAILED; not run, not tagged)

grokbot PR #35 (preregistration) plus the cycle-0036 report. The last level-up is still **v0.0.19**.

- **Lever:** a valid-knockdown bonus (any knockdown whose fall is not blocked), starting from the v0.0.22 policy and gated by a preregistered smoke test.
- **Bonus-path check passed:** the bonus fired in 23 of 24 trials (all `self` falls), so the instrumentation fix works.
- **Validity check failed:** the 12-game base head-to-head against v0.0.19 was **10/12 valid (83%), short of 90%**, with 2 blocked games. **`SMOKE_FAILED`, so there was no training run.**
- **Attempts:** attempt 01 was aborted within a minute, before any game, because the launch was not gated on a failing load check. Attempt 02 was launched at load 64.
- **Red-team:** the validity check measured the *starting* policy's blocked rate, which is the failure the lever targets, on only 12 games. v0.0.25 should gate the instrumentation separately from that baseline.

## v0.0.23 — 2026-10-28 (preregistered criterion NOT met: VALIDITY_FAILED; not tagged)

grokbot PR #33 (preregistration) plus the cycle-0034 report. This is a diagnostic, not an unassisted boxing result. The last level-up is still **v0.0.19**.

- **One lever:** a clean-knockdown bonus that pays only for falls attributed `knockdown`. **The bonus never fired:** its share was 0.00 in all 48 training iterations. Training therefore reduced to stand time.
- **Primary vs frozen v0.0.19:** **5/48 valid** (35 no-contests, 8 blocked), so `VALIDITY_FAILED`. The probe exited non-zero.
- **Hard gate vs v0.0.13:** 14/48 valid (30 blocked). Monotonicity was unmeasurable.
- **Why:** a replay of the v0.0.22 policy against v0.0.19 found **22/24 knockdowns attributed `self`**. One hit lands at about tick 252 and the opponent falls 1–2 s later, outside the 0.5 s attribution window. The rest were 2 `unresolved` and 0 `knockdown`. `self` falls are valid wins under ending v2, so this lever excluded the valid outcomes.
- **Caveat on v0.0.22:** its 46–0 was mostly one early hit followed by the opponent toppling on its own.
- **Next (v0.0.24):** a valid-knockdown bonus that excludes only blocked falls, behind a preregistered smoke gate that requires the event in at least 10% of smoke trials.

## v0.0.22 — 2026-10-28 (preregistered level-up NOT met; not tagged)

grokbot PR #31 (preregistration) plus the cycle-0032 report. This is a diagnostic, not an unassisted boxing result. v0.0.20 and v0.0.21 were also untagged misses (cycles 0028 and 0030). The last level-up is still **v0.0.19**.

- **One lever:** a knockdown objective. The training bonus requires a head/torso hit, then the opponent falling first. It was trained against frozen v0.0.19.
- **Primary, against frozen v0.0.19: PASSED.** 46/48 valid, **46–0–0**, every win with hits, Elo +661 (lower95 +186). These are the first decisive wins over v0.0.19; v0.0.21 drew 42 of 48 against it.
- **Missed criteria:**
  - The **v0.0.13 hard gate failed**: 36/48 valid, against 40 required, with 12 games blocked.
  - **Monotonicity was unmeasurable** for @16 vs v0.0.19: 38/48 valid, with 10 blocked.
  - @48 vs @16 went 48–0.
- **Why it missed:** 0 valid losses in 168 games. Both failures are **blocked** games, where the opponent's fall follows body contact and cannot be attributed. The bonus still paid for those falls. The next lever (v0.0.23) rewards only knockdowns the ending attributes.

## v0.0.19 — 2026-10-28 (preregistered level-up PASSED)

grokbot PRs #24 and #25 plus the cycle-0026 report. This is a solo-trained motor policy evaluated against frozen opponents. It is a diagnostic, not an unassisted boxing result: it does not claim jekko's completion battery, and native boxing games remain 0. v0.0.18 was not tagged, because it missed its criterion.

- **One change from v0.0.18:** a "hit and stay up" fitness. A head/torso hit earns the bonus only if the athlete is still standing 0.5 s afterwards.
- **Preregistered primary, against the frozen v0.0.13 champion on new seeds:** 48/48 valid, **48–0, every win with head/torso hits**, Elo +669 (lower95 +191). **Passed.**
- **Motor gate passed for the first time:** stance 16/24, reach and hits 24/24, median stand 3.0 s (v0.0.18: 1/24, 2.31 s).
- **Against frozen v0.0.16:** 47–1, Elo +669 (lower95 +172); v0.0.18 went 15–29. Caveats: v0.0.18 also beats v0.0.13, and only 21 of 47 wins against v0.0.16 include hits. It wins by outlasting the opponent more than by knocking it down.

## v0.0.17 — 2026-10-28 (preregistered criterion not met)

grokbot PRs #20–#22 plus the cycle-0023 report. This is a solo motor diagnostic. It does not claim jekko's completion battery, and native boxing games remain 0. v0.0.14–v0.0.16 were not tagged, because each missed its preregistered criterion.

- **Controller ladder:**
  - v0.0.15: `PunchPolicy`, 10 regional tones. 0/24 stance; reproduced exactly on a rerun.
  - v0.0.16: `GroupPolicy`, 138 DynSyn-group tones. Median stand 1.46 → 2.34 s.
  - v0.0.17: `GroupFeedbackPolicy`, per-group tilt gains. Median stand **2.68 s**. **7/24 trials stood the full 3 s**, the first ever, and **24/24 landed a strike**.
- **Result:** the primary motor gate needs stance ≥ 0.5 and reach ≥ 0.5. It **failed on stance (0.29)**, so the preregistered head-to-head against the frozen v0.0.13 champion did not run. The sign test against v0.0.16 passed (23/24, p = 1.5e-6).
- **Exploratory, not counted:** on the pinned seeds, v0.0.17 vs v0.0.13 gave 14 of 24 games valid and **14 of 14 won**. These are the first valid games. However, every v0.0.17 strike lands on the opponent's glove or guard, and each win is that tap followed by the opponent's own fall. v0.0.18 adds head/torso hit tracking and an Elo gate that requires hits.

## v0.0.13 — 2026-10-28 (preregistered criterion not met)

grokbot PR #17 adds a punch primitive. It does not claim jekko's completion battery, and native boxing games remain 0.

- A DynSyn reach-synergy discovery found that groups 58/81 move the hand about 0.20 m forward relative to the trunk in 0.2 s. Fixed punches landed 0 strikes, and the gloves stayed about 0.4 m short of the target shells.
- A `PunchPolicy` learns punch timing and amplitude on top of gen4 posture control. Its shaping rewards trunk-relative extension, which a body lean cannot fake. The frozen v0.0.11 champion served as anchor and as evaluation target.
- **Result:** 1,344 evaluation games produced 0 strikes and 0 valid results; 1,330 were no-contests. Neither the strike-emergence criterion nor the gain-vs-v0.0.11 criterion was met. The bottleneck is motor: the athletes fall backward and cannot reach.

## v0.0.12 — 2026-10-27 (preregistered criterion not met)

Tagged at the owner's request on the #16 merge (`deda861`).

- Added: anti-stalling (no strike and no points means a no-contest), a 0.80 m range reset, an engagement promotion gate, and fault quarantine.
- The re-run champion gate voids the zero-contact v0.0.11 Elo.
- Contact emerged, but all of it was body-on-body leaning. There were 0 strikes, and 0 of 588 evaluation games produced a valid result.

## v0.0.11 — 2026-10-27

The owner asked for this tag, which sits on main after grokbot PR #14 (`55280e4`) and this entry. **It does not claim jekko's v0.0.11 completion battery.** That battery (`docs/V0.0.11_COMPLETION_PACKET.md`) remains outstanding, and native boxing games remain 0. The v0.0.9 and v0.0.10 notes are in `docs/V0.0.9_RESULTS.md` and `docs/V0.0.10_RESULTS.md`.

- **Main line (jekko).**
  - Diagnostic shared two-player `BoxingEnv` with opponent-aware observations.
  - Causal sensor timing: all eight D1 conditions, four exact replays.
  - Lossless evidence storage.
  - Pretrained shared-motor input oracle.
  - First shared pretrained walking trial: both athletes stayed upright for 3.52 s. The four-seed proof is incomplete after a 32 MiB storage cap.
- **CI.** A GitHub Actions gate mirrors the locked v0.0.11 runtime (grokbot #1). On the #14 head: 1017 passed, plus 1 registered strict xfail.
- **Grokbot red-team lane (#1–#14, `grokbot/` only).**
  - League evaluation: a league/Elo/Nash oracle, a side-swapped arena with common random numbers, and a reference ending table with contact-classified attribution.
  - A promotion gate.
  - PSRO and exploiter-recall positive controls.
  - Fuzzes of `HitTracker` and `MatchReferee`.
  - The P4 opponent sampler.
  - **The first two-player self-play league (diagnostic).** The champion is gen4, at Elo +1081 against the scripted baseline. Side balance was 0.000 every generation. However, there were **0 contact ticks in 876 games**: the league learned to stand, not to box.
- **Known blockers.**
  - `HitTracker.restore` snapshot aliasing, carried as a strict xfail.
  - The feather-tap scoring ceiling.
  - No registered boxing-range reset.
  - The passive equilibrium. Its fix is in progress for v0.0.12.

See [cycle reports](grokbot/redteam/), [shared motor results](docs/SHARED_MOTOR_PHYSICAL_RESULTS.md) and [status](docs/STATUS.md).

## v0.0.8 — 2026-10-26

- Add signed every-substep physical evidence and scratch end-posture telemetry while preserving native dynamics/cached policy features exactly.
- Separate original tracking failure from physical failure; all40 attempts valid, both conditions0/20 physical20s completions, all failures excess tilt around5s.
- Pass all12 boundary/mid-command fixed-action/policy/repeat replays exactly and127 Mac/Linux tests. Retain development failures, every outcome, full public raw archive, plot/video and actual process exits.
- Record70.33/130.16 control TPS at one/two workers with optimizer/referee excluded; register v9 profiling/batched-inference work and a worker-ready route to meaningful historical-opponent self-play.

See [results](docs/V0.0.8_RESULTS.md), [book](experiments/v0.0.8/README.md), [evidence](artifacts/v0.0.8/), [next mission](docs/V0.0.9_MISSION.md) and [self-play/HPC plan](docs/SELF_PLAY_HPC_PLAN.md).

## v0.0.7 — 2026-10-26

- Add canonical-reference/live-body mapping for both observations and rewards after glove insertion; preserve all 700 muscles, native inertias and fixed 20ms control cadence across 1ms/2ms physics.
- Pass exact native feature/reward/state and actual v6 continuation oracles; independently audit all 80 paired trials and all 12 first-seed action/policy/repeat replays with zero error.
- Complete the frozen longer-horizon factorial: every condition 0/20 native tracking completions, zero invalids. Native baseline also fails; no glove-only cause or learned improvement established.
- Retain post-hoc joint/force/velocity evidence: tracking stops with sampled foot support still present. Publish all outcomes, full public raw archive, hashes, plot/video and exited processes.
- Pass 99 Linux tests and Mac full 98 plus final 27 targeted checks; measure full-policy CPU throughput and startup cost. Register the bounded v8 substep physical-continuation mission.

See [results](docs/V0.0.7_RESULTS.md), [book](experiments/v0.0.7/README.md), [adapter](docs/NATIVE_TRANSFER.md) and [next mission](docs/V0.0.8_MISSION.md).

## v0.0.6 — 2026-10-26

- Qualify the frozen upstream DynSyn-SAC walking controller on40 paired native attempts:40 horizon completions, zero invalid/incomplete/not_run; no optimization.
- Reproduce decoder schedule loss on checkpoint load; restore the scheduled0.075 attribute, preserve immutable normalization and prove corrected save/load predictions match exactly.
- Add native full-state/reference/RNG/cached-observation snapshots and independent source-host fixed-action/closed-loop replay: zero errors on both modes.
- Pass80 tests on Mac/Linux in a separate locked checkpoint-compatible runtime. Preserve failed Mac compiled-contract portability and every deployment/test/render failure.
- Publish all attempts, traces, hashes, audits, plot/video and owned PID exit. Schedule active20-minute health/mission supervision and register the next bounded glove/timestep transfer mission.

See [results](docs/V0.0.6_RESULTS.md), [experiment book](experiments/v0.0.6/README.md), [evidence](artifacts/v0.0.6/) and [next mission](docs/V0.0.7_MISSION.md).

## v0.0.5 — 2026-10-26

- Add preregistered constant-control, real-optimizer profiling and matched reward/entropy SAC experiments, then a separately authorized fourfold-budget extension with 16 simulation workers across four learners.
- Complete 1.44 million training transitions, 354,000 scheduled updates and all 912 new evaluations, zero invalids. Preserve every experiment description, actor, progress/monitor record, comparison and not_run reason.
- Candidate B improves over its initial policy on two fresh seeds but loses to A on one; registered replicated superiority fails. Every final policy has 0/32 five-second survivors; no formal motor proof or boxing self-play ran.
- Measure 4.69× common-data optimizer throughput and 844.67TPS at 16 fixed-policy workers (optimizer excluded). Publish source-host replay, both confirmation seeds, 72 passing Mac/Linux tests and all owned PID exits.
- Preserve failed strict cross-platform policy replay, disclose constant-baseline confirmation timing, and correct diagnostic preparation accounting additively.
- Audit delivered velocities/muscle feedback versus missing opponent/history/jitter/game/league interfaces; inspect upstream motor checkpoint metadata and provide a worker-ready transfer plan.

See [results](docs/V0.0.5_RESULTS.md), [experiment book](experiments/v0.0.5/README.md), [state/history audit](docs/STATE_HISTORY_SELF_PLAY_AUDIT.md) and [next motor plan](docs/UPSTREAM_MOTOR_TRANSFER_PLAN.md).

## v0.0.4 — 2026-10-26

- Implement ordinary-gravity native-root MS-Human-700 feedback control with all 700 muscles, 138 named action groups, 1,023 measured observations and audited reset-only preparation.
- Add signed/persistent foot support, applied-force/gravity assistance guards, physical failure versus external truncation, bounded CPU SAC, atomic progress/policy checkpoints and raw paired development evaluations.
- Add twelve root probes, 32-reset audit, complete control-boundary snapshots, malformed-state rejection and sorted-JSON channel-order regression.
- Add independent checkpoint/receipt/replay audit, fixed-scenario plot/video and 1/2/4-worker workload benchmark tools. Linux installs CPU Torch wheels.
- 63 tests pass on both Mac and Linux with the motor extra. The wall-capped pilot completed 39,040 transitions and 64 paired evaluations, zero invalids; mean survival 1.018 -> 1.023 s, interval includes zero, five-second survivors 0/32. Motor continuation gate failed; larger proof was not launched.
- Linux continuation is exact. Preserve the failed strict cross-platform full-state check and the caught/fixed development replay defect. Measured 1/2/4-worker inference/reset throughput: 65.71/131.81/240.52 control transitions/s, optimizer excluded.

See [results](docs/V0.0.4_RESULTS.md), [execution plan](docs/V0.0.4_PLAN.md) and [status](docs/STATUS.md). v0.0.3 was a planning milestone, carried into this implementation.

## Planned — v0.0.3

- Make `neverhuman/Joxing` public and verify repository/file access without credentials; add a documentation index and concise training/test guide.
- Define worker-ready gravity, root-fall, support/reset, stance-learning and push-recovery tests with frozen evaluation, replay and bounded compute requirements.
- Separate test-platform delivery, motor competence and evidence of learned improvement. Implementation/results remain pending; released package remains v0.0.2.

See [v0.0.3 test plan](docs/V0.0.3_PLAN.md).

## v0.0.2 — 2026-10-26

- Connect actual MS-Human-700 muscle control to physical glove contact, geometry-derived onset/face/withdrawal evidence and the existing point referee.
- Add complete engine/application snapshots and mid-contact restore validation.
- Add bounded three-seed policy search, immutable proof protocols, atomic progress/checkpoints, held-out evaluation and fixed-scenario visual evidence.
- Frozen proof: trained controllers hit once and withdraw in 192/192 trials, versus 0/192 baseline withdrawals; both retain 192/192 legal hits. Zero invalids; 256.813 seconds on four xbabe1 CPU workers.
- Verify all three checkpoints and contact snapshots independently on Mac; 40 tests pass on both Mac and Linux.
- Preserve fixed-pelvis/zero-gravity assistance and model-force limitations. Unassisted stance, full-body contact/rules, self-play and GPU parity remain future gates.

See [results](docs/V0.0.2_RESULTS.md), [plan](docs/V0.0.2_PLAN.md) and [raw evidence](artifacts/v0.0.2/).
