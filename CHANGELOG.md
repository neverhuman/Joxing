# Changelog

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
