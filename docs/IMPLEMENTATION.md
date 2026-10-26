# Implementation and handoff plan

Repository: [neverhuman/Joxing](https://github.com/neverhuman/Joxing), public. Current implemented iteration is v0.0.7: [transfer mapping and exact replay pass, longer native tracking fails](V0.0.7_RESULTS.md). All 80 attempts are valid, every condition 0/20 horizon completions; 99 Linux tests pass. [v0.0.8](V0.0.8_MISSION.md) separates tracking from physical failure using substep evidence before another motor-learning wave. No new optimization or boxing competence is claimed. The P0–P5 sections describe the broader program, not delivered features. [Index](README.md), [current mission](MISSION.md).

The [upstream motor-teacher roadmap](UPSTREAM_MOTOR_TRANSFER_PLAN.md) has completed its native loader/normalizer/replay qualification in v0.0.6. The [v0.0.7 glove/timestep transfer](V0.0.7_RESULTS.md) preserves interfaces and exact replay, but fails long native tracking. Next is the [v8 physical motor evidence mission](V0.0.8_MISSION.md). Native walking does not satisfy stationary stance, guard, reach/return or competitive gates.

## Objective, scope and completion

The [state/history/self-play audit](STATE_HISTORY_SELF_PLAY_AUDIT.md) refines P2/P3/P5 with exact delivered/missing mechanisms, causal observation/history/latency contracts and tests. The user-authorized current CPU allowance is 16 simulation workers total under the [v0.0.5 resource](V0.0.5_RESOURCE_ADDENDUM.md) and [budget](V0.0.5_BUDGET_ADDENDUM.md) addenda; earlier four-worker entries describe initial caps.

Produce a reproducible virtual boxing game in which two independently muscle-driven humans learn motor skills and improve tactics through self-play. Final completion requires stable unassisted motion, calibrated simulation-rule contact scoring, full replay, historical-opponent learning and paired fixed-panel evidence of improved match strength. The first delivery is the auditable CPU foundation plus a bounded learning-pipeline experiment.

Non-goals for v0: injury/concussion prediction, deformable organs/foam, photorealism, articulated fingers, official judging, mixed weight classes, real-human tracking and robotics transfer. No cloud hardware purchase is assumed.

Assumptions: equal morphology; full 700-muscle reference; state-based observations; native scalar-root baseline until a separately validated variant wins; simulation-consistent force telemetry accepted initially. Dependencies: pinned submodules, Python/MuJoCo lock, reachable SSH hosts, geometry/reset calibration, licensed motion or procedural goals, competent motor policy before tactics, exact-model GPU compatibility before acceleration.

Stop conditions: numerical invalidity, unexplained contact-stage/sign error, missing physical contact or buffer overflow, duplicate scores, action/critic leakage, unresolved active-foul fall/exit exploitation, disk/resource cap, or changed model/rules/bundles inside a match. Preserve fault traces and halt promotion; do not report aborted games as draws or silently omit them.

## Read first, in order

1. `AGENTS.md`, `README.md`, `docs/STATUS.md`.
2. `docs/GAME.md`, `docs/SPEC_REVIEW.md`, `docs/COMPUTE.md`.
3. `src/joxing/scene.py`, `physics.py`, `rules.py`, `provenance.py`.
4. `tests/test_scene.py`, `test_physics.py`, `test_rules.py` and receipts.
5. `third_party/ms-human-700/MS-Human-700.xml`, native body/equality/tendon files; `third_party/msgym/msgym/envs/locomotionFull_v1.py`, `imitation_trajectory.py`, `DynSyn-SAC/configs/locomotionFull.json`, wrapper and DynSyn layer source.
6. All four `docs/references/*.md` for rationale; their commands are proposals, not executable authorization.

Local checkout: `/Users/bentaylor/Documents/ChatGPT/JOXING`. First compute checkout: `/home/ubuntu/joxing/bootstrap-20261026` on xbabe1. Portable commands below are run from checkout root; prefix with `rtk` on the local Mac per global instructions.

## Ownership and write boundaries

`third_party/**` is read-only; upgrade only via explicit gitlink commits and regression results. `artifacts/generated/**`, `runs/**`, `.venv/**` are generated/ignored. Curated small evidence lives in `artifacts/receipts/**`; no secrets or large rollout files. Source/docs/tests are editable. Only one writer may own scene/registry/rules schemas; any change must include consumers/tests/docs. Do not launch parallel writers against these files. Later geometry and motor preparation can run as separate work packets only with disjoint files and a designated integration owner. No delegation is needed for this initial slice.

## Ordered work packets

### P0: Executable foundation (initial delivery)

Files: `pyproject.toml`, `uv.lock`, `src/joxing/{scene,physics,rules,motor,provenance,cli}.py`, current tests and docs.

Compile two intact primary anatomies with `build_scene`; verify 170 qpos/170 qvel/1,400 actuators/84 equalities, mass increment and native actuator properties. Independently integrate contact forces and compare momentum, rotated normals and reversed order. Verify one-event scoring, no guard impulse transfer, onset latching, retraction rearming, terminal shaping and mid-hit restore. Run bounded CEM motor experiment in named fixed-pelvis/zero-gravity rig; save all candidates, seeds, progress and held-out errors. This gate proves infrastructure only.

Commands: `uv sync --extra dev --extra viz`, `uv run pytest -q`, `uv run joxing inspect`, `uv run joxing fixture`, `uv run joxing benchmark --steps 500`, `uv run joxing train-reach --workers 4`. Expected: generated XML/manifest, contact fixtures, passing tests, resource-limited checkpoint and result JSON. Failure to learn is a diagnostic; do not select a lucky clip and call it boxing.

### P1: Complete geometry and physics qualification

Edit `scene.py`; create `geometry.py`, `config.py`, `fixtures.py`, `tests/test_geometry.py`, `tests/test_health.py`, `configs/smoke.json`, `configs/calibration.json`.

1. Define immutable `GeomRecord` with semantic identity, actor, body, role, material, glove ID, local anatomical frame and scoring-region map. All physical geoms resolve, including native collider leftovers.
2. Fit massless full external shells in native frames, review native skull/pelvis overlap, and implement explicit nonadjacent own-body pairs. Avoid redundant overlapping surfaces and duplicate tissue mass. Verify compiled inertia/COM, not only scalar mass.
3. Compute face membership and current all-shell clearance geometrically, with pre-impact glove/trunk histories. Implement target-local front/side/back zones with deterministic seams. Do not add overlapping scoring-only collision surfaces.
4. Add full pose/probe suite: neutral/guard/reach/crouch/trunk rotation/fall, six view renders, moving-target and rotating-glove fast-pass stress. Compare 2/1/0.5 ms at fixed motor cadence.
5. Reproduce native-root side-fall singularity claim. If confirmed, add a named free-root variant and equivalence tests, then consciously choose reference. Do not silently replace qpos layout.
6. Save simulation calibration with units, scene/solver hashes, impulse convergence (<5% candidate), force-peak sensitivity and unknown empirical calibration clearly labeled. Resolve production config only after reviewed tests.

Validation: engine collision fixtures must prove guards/self-contact block traversal; no warning/nonfinite/overflow; no unreported fast pass; staged forces/poses agree. Compile success alone fails P1.

### P2: Authoritative game and replay

Edit `rules.py`, `physics.py`; create `env.py`, `observations.py`, `referee.py`, `replay.py`, tests `test_env.py`, `test_referee.py`, `test_replay.py`, `test_observation_leakage.py`.

Define typed `ContactSample`, `ScoreEvent`, `ActorObs`, `CriticState`, `ReplaySnapshot`, resolved rules schema. Add foul motion attribution/pair exemptions, sustained-drive detection, support/boundary persistence and separation state. Fall/exit cannot bypass foul review. Process simultaneous hits before sporting termination; finalize pre-bell evidence without more physics. Match reward is terminal utility plus telescoping potential; gamma=1.

Both actions use one completed state; scatter muscles via semantic ownership; decoder every 10 ticks, strategy every 20 candidate ticks. Capture every physics substep. Pure collector cuts preserve state; real bell terminates, external truncation bootstraps final observation, numerical failures quarantine. Actors get visible kinematics only; critic-only hidden state cannot affect them.

Replay engine with `mjSTATE_INTEGRATION` plus all application state: phase/clocks, scorer/ledger/onset histories, controls/latents, normalizers, recurrent states, delay/filter queues, randomization/model draws and RNGs. Require restore during contact, between motor updates and before bell reproduces event IDs/scores. Extend existing physical sphere-to-referee tests to fitted dual-human surfaces, guard bypass and semantic ownership; the isolated rigs alone do not pass this gate.

### P3: Fast motor learning and latent codec

Create `groups.py`, `motor_teacher.py`, `motion.py`, `skill_codec.py`, `eval_motor.py`, `tests/test_groups.py`, `tests/test_motor_api.py`; retain `motor.py` as named diagnostic.

Convert upstream 138-group indices into muscle-name partition; assert all 700 exactly once. Implement goal-conditioned SAC with explicit deterministic group expansion and preserved legacy sigmoid for reproduction. Direct [0,1] excitation is a new training contract; do not reuse incompatible checkpoints/replay. Teacher reward uses normalized tracking, support, nonnegative effort and smoothness, not signed summed muscle force.

Build feasible procedural goals and/or license-checked noncyclic motion. IK respects equalities/limits; velocities use model-aware differences. Train stance, independent reach/return, footwork and contact recovery. Proposed gate: 95% held-out 20-second stance completion and 90% reach/return completion over 200 trials at <5 cm glove and contralateral error. Split offending groups if independent arms fail; never lower targets to hide failure.

Train conditional encoder and state-conditioned decoder from successful teacher rollouts; evaluate 32/64/128 dimensions with equal budgets; aggregate corrections on student states. Distillation action error alone is insufficient. Freeze/version decoder before tactical PPO; no blind decoder unfreezing under a latent likelihood loss.

### P4: HPC backend and complete-workload benchmark

Create `backends/{base,cpu,warp}.py`, `benchmark.py`, `tests/test_backend_parity.py`, `docs/GPU_FEATURE_AUDIT.md`.

Benchmark CPU process counts 1/2/4/8 and exact-model Warp worlds 8/32/128/512 subject to measured memory headroom. Gate all muscle/tendon/wrapping/equality features and contact completeness. Port scorer/recovery/clock reductions device-side only after CPU oracle tests. No maxforce-only sensor reduction may drop legitimate manifold impulse. Log solver/contact/event capacity independently. Compare controlled impacts and accepted events; long chaotic bitwise trajectories are not expected to match forever.

Run end-to-end simulation+inference+referee+reset+optimizer throughput, warmup separate. Report world transitions, substeps, learner samples, wall and simulated seconds, GPU memory/utilization, faults and disk use. Initial resource cap four CPU workers; expansions require current load check. Keep xbabe4 native CPU unless a measured alternative warrants Docker.

### P5: Self-play and visible proof

Create `rl/{actor,critic,ppo,rollout}.py`, `league.py`, `evaluation.py`, `report.py`, tests `test_ppo.py`, `test_league.py`, `test_evaluation.py`.

Recurrent tactical actor samples tanh-squashed Gaussian latents with exact stable likelihood, detached behavior logp/advantages, padded sequence masks and burn-in/full-history checks. Separate bootstrap and trace masks. Freeze behavior bundle per rollout; reconstruct hidden state on new weights or keep version through match. Historical opponent retains actor+decoder+normalizers+timing+RNG contract.

Start shared learner versus scripted styles and immutable snapshots. Add historical/PFSP mixture with uniform floor, frozen evaluation panel and payoff matrix. Elo comes from evaluation, never the changing training distribution. Candidate promotion: preregister 200 paired scenarios, swapped sides, lower 95% paired panel-score improvement bound >0, no critical regression/fault. Five independent seeds for method comparisons; one debug seed never proves general improvement.

Replay UI overlays clock, points, location, force/impulse, event reason, actor and decoder hashes. Learning report plots panel strength versus wall hours and simulation budget, confidence and invalid rate. Show initial/intermediate/final policies on identical scenarios. Camera polish does not replace these metrics.

## Logging, receipts and handoff

Every run saves resolved config, code/source/upstream hashes, engine/backend/precision, hardware, seeds, model/rules/observation hashes, action transforms and assistance. Save candidates/controls, optimizer history, checkpoint hashes and held-out protocol before looking at results. Record commands, elapsed time, artifact paths and failures in STATUS.md. A failed run leaves a fault receipt and attempted-match count.

Final handoff lists achieved gates, remaining blockers, exact continuation command, host/process if still running, expected outputs and path to receipts. Commit source before training; receipts may reference the preceding source commit. Push to NeverHuman/Joxing. Do not mark competitive learning complete until P1–P5 evidence exists.
