## v11 independent foundation while frozen v10 runs

Implemented bodies/{base,native}.py: semantic85joint/700activation ownership per corner and six measured body positions/linear velocities from the shared MjData. Measurements are owned immutable copies with a shared timestamp; Jacobian velocity readout is checked against independent point-Jacobian evaluation. Private opponent activation/joint measurements are not policy observations. This is not BoxingEnv, a history stack, a referee or a bout. New body+recording subset3tests pass5.58s with0new simulation steps in the body test. Recursive provenance now includes nested body modules; a separate hash-mutation test passes0.17s. These edits are in the development checkout only; remote D2sourceb1d1690and its196-test gate are unchanged.

## Active frozen Linux qualification/pilot —2026-10-27 10:43UTC

Public/frozen sourceb1d1690ec6bc0c4d248336ce407a5c71a3f0e7c6; protocol canonical725313890a6e260887fb4542a39878646a8c73508c5a6a35f8294d69061013ec anonymously byte-verified, repositorypublic. Dedicated xbabe3:/home/ubuntu/joxing/v010-pilot-20261027, supervisor2726271/start_ticks919982114. Launch scanownedempty,100.86GBavailableRAM/37.45GBdisk. Exactsource/input/288upstream file hashes verified. Supervisor requires literal196Linux passes before one fixedD2pilot, then independent saved-evidence audit; stage deadlines180/420/2130/180s. Main550kphysics/1800s/1GiB including prior10,394physics/25.066975s; separate50k/300sauditreserve, max4simulationworkers/one thread. No executing-source mutation or duplicate launch. Local196tests passed177.52s, Ruffpasses.

Read actual runs/v0.0.10/supervision-01/{supervisor.json,tests.log,D2.log,audit.log} and pilot-01/{result.json,events.jsonl,seed-*/worker-*/progress.json}. Check fresh counters/CPU/RSS and faults, preserve all32evaluation/two training/two replay rows. On completion reconcile actual PID exits, full raw manifest, checkpoint/update evidence and independent audit; publish honest pilot conclusion and register/activate v11connected-recording work. No larger unchanged walking sweep.

Independent local rendering smoke: attempt01failed PIL-to-video type boundary, repaired locally; attempt02decoded1static1024×576frame/25fps with correct diagnostic/no-score HUD, both athletes visible,0newphysics. This is renderer QA, not a motion clip or bout. Failure/repair retained in artifacts/v0.0.10/D2/recording-smoke.json. Render fix is only in development/public main; frozen remote b1d1690unchanged. D2reproduction/audit must use that exact source checkout: its protocol hashes intentionally reject later development files. User-facing bout delivery remains BOUT_RECORDING_MILESTONE.md.

## 2026-10-27: visible-bout priority and D2 launch preparation

User priority: reach recordings of two responsive boxing athletes. New [recorded-bout delivery packet](BOUT_RECORDING_MILESTONE.md) makes shared-world footage an explicit v11 deliverable, followed by qualified guard/reach exchanges and fixed-scenario before/after historical self-play. No native valid bout exists yet. `bout_recording.py` and `render_bout_recording.py` implement passive dual-body state tapes, checksums, bounded output and zero-step scratch playback. First schema is diagnostic-only with points unavailable. Two fixture tests cover both complete bodies, capture nonmutation, clock/hash corruption, causal sampling and bounds; no boxing footage or learner improvement is claimed.

D2 parent/protocol/auditor source is now implemented, including independent saved contact/reward/first-latch arithmetic, all-attempt accounting, checkpoint/optimizer hashes, teacher-load RNG isolation and bounded cleanup. New supervisor gates the exact196-test Linux suite, records PID/start-time identities, then permits one fixed two-seed pilot and saved-evidence audit. No formal D2 run has started at this entry. Fresh xbabe3 scan10:35:46UTC: owned jobs0,95.16GB available RAM,38.12GB disk, load11.22/11.44/11.65. A first remote scan failed because rtk is absent there; local commands remain prefixed with rtk, remote Python is invoked through that outer proxy. Refresh capacity at actual launch.

Raw credits finalized before any D2 outcomes: parent192MiB + eight workers32MiB +32evaluations12MiB +two replays12MiB +prior32MiB =888MiB, within unchanged whole-version1GiB. This replaces the unregistered draft distribution and leaves196MiB margin; no physics/wall/worker/learning/win gate changed. Physical main allowance remains550k/1800s including prior10,394/25.066975s; separate50k/300s audit reserve. Full source/protocol publication and Linux qualification still precede formal outcomes. Earlier draft statuses below are historical.

# Implementation and handoff plan

Repository: [neverhuman/Joxing](https://github.com/neverhuman/Joxing), public. Current implemented iteration is v0.0.8: [all40 physical attempts valid, 0/20 horizons per condition, all12 replays exact](V0.0.8_RESULTS.md);127 Mac/Linux tests pass. Physical failure follows tracking by about0.8s. No new optimization or boxing competence is claimed. The [self-play/HPC execution route](SELF_PLAY_HPC_PLAN.md) and [v0.0.9 mission](V0.0.9_MISSION.md) prioritize measured full-workload scaling/batched inference alongside motor prerequisites. The P0–P5 sections describe the broader program, not delivered features. [Index](README.md), [current mission](MISSION.md).

The [upstream motor-teacher roadmap](UPSTREAM_MOTOR_TRANSFER_PLAN.md) has completed its native loader/normalizer/replay qualification in v0.0.6. The [v0.0.7 glove/timestep transfer](V0.0.7_RESULTS.md) preserves interfaces and exact replay, but fails long native tracking. [v8 physical evidence](V0.0.8_RESULTS.md) now measures collapse around5s; next is the [v9 HPC mission](V0.0.9_MISSION.md). Native walking does not satisfy stationary stance, guard, reach/return or competitive gates.

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

Additional acceptance cases from the [external proposal review](EXTERNAL_REVIEW_2026-10-26.md): front/wide/body guards, moving parry, evasion, guard-then-direct-target and moving/rotating target fast passes. Verify measured contact ordering and direct impulse rather than assumed percentage damage reduction. Actor-exchange tests must specify spatial/vector/action transforms and any left/right anatomical permutations. These remain proposed engine fixtures, not delivered learned defenses.

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

Profile observation assembly, IPC/transfer and synchronization separately before selecting buffer reuse, batched inference or native-code work. The [additional proposal's](EXTERNAL_REVIEW_2026-10-26.md) simplified-engine throughput targets are not exact-model benchmarks or launch requirements. Any future surrogate retains a separate model identity and cannot pass native motor/backend gates.

### P5: Self-play and visible proof

Create `rl/{actor,critic,ppo,rollout}.py`, `league.py`, `evaluation.py`, `report.py`, tests `test_ppo.py`, `test_league.py`, `test_evaluation.py`.

Recurrent tactical actor samples tanh-squashed Gaussian latents with exact stable likelihood, detached behavior logp/advantages, padded sequence masks and burn-in/full-history checks. Separate bootstrap and trace masks. Freeze behavior bundle per rollout; reconstruct hidden state on new weights or keep version through match. Historical opponent retains actor+decoder+normalizers+timing+RNG contract.

Start shared learner versus scripted styles and immutable snapshots. Add historical/PFSP mixture with uniform floor, frozen evaluation panel and payoff matrix. Elo comes from evaluation, never the changing training distribution. Candidate promotion: preregister 200 paired scenarios, swapped sides, lower 95% paired panel-score improvement bound >0, no critical regression/fault. Five independent seeds for method comparisons; one debug seed never proves general improvement.

Include shell/counter, jab/move and close-range hook baselines once valid motor controllers exist. Record misses, blocks, direct hits, counters/combinations and engagement distance as defined diagnostics, without changing canonical points. `test_league.py` must cover empty pools, all-zero/all-one win rates, sparse evidence, uniform-floor/fallback behavior and fixed-seed sampling. Freeze EMA-derived opponents at match reset; never mutate an active opponent bundle. Monotonic Elo and bot mastery alone do not establish improvement. See the [additional review](EXTERNAL_REVIEW_2026-10-26.md) for rationale and input provenance.

Replay UI overlays clock, points, location, force/impulse, event reason, actor and decoder hashes. Learning report plots panel strength versus wall hours and simulation budget, confidence and invalid rate. Show initial/intermediate/final policies on identical scenarios. Camera polish does not replace these metrics.

## Logging, receipts and handoff

Every run saves resolved config, code/source/upstream hashes, engine/backend/precision, hardware, seeds, model/rules/observation hashes, action transforms and assistance. Save candidates/controls, optimizer history, checkpoint hashes and held-out protocol before looking at results. Record commands, elapsed time, artifact paths and failures in STATUS.md. A failed run leaves a fault receipt and attempted-match count.

Final handoff lists achieved gates, remaining blockers, exact continuation command, host/process if still running, expected outputs and path to receipts. Commit source before training; receipts may reference the preceding source commit. Push to NeverHuman/Joxing. Do not mark competitive learning complete until P1–P5 evidence exists.

### v9 measured HPC: N1 and H1

Frozen batching remains unqualified: identical normalized inputs diverge first at latent_pi.0; MKLDNN disable does not change7.15e-6 final error. No old tolerance changed. H1 full native scalar loop completed6/6cases,18kphysics, all three profiled/unprofiled trajectories exact. Baseline median62.48control transitions/s; physical/evidence/telemetry dominates63–70%,actor+normalization23–29%,snapshot~4%. These are one-human motor diagnostics, not valid boxing games or optimizer throughput. Source core remains v8 unchanged; no new rollout backend or actor service. See experiments/v0.0.9 and artifacts/v0.0.9/{N1,H1}.

Next implemented harness is bounded scalar CPU scaling1/2/4/8/16 with serial per-control full-snapshot/action/reward/telemetry hash anchors and source-frozen protocol. It retains common start, startup/RSS/resources, all failed/skipped rows and hard-crash counter uncertainty. Source/protocol precede outcomes; full Linux regressions gate formal H2. Next work packet docs/V0.0.9_H2_SCALING.md. Motor stability/history/referee/league gates remain unmet; scaling cannot replace them.

### H2 partial scaling audited; roomier-host qualification ready

146Linux tests pass. H2attempt01 stopped on fixed16worker memory reservation before launch;31completed/47not_run cases,37,200physics. All15completed parallel cases match serial snapshot/action/reward/contact-telemetry hash histories.1/2/4/8first-repeat steady43.82/82.21/155.66/226.44controlTPS, startup-inclusive9.85/17.98/34.41/56.18; no repeated median or16worker claim. All22PIDs exited and source unchanged. Exact counter/denominator audit retained in artifacts/v0.0.9/H2/attempt-01.

Attempt02 source/protocol ready for xbabe3after independent~102GB RAM/49GBdisk capacity check. Original H2source/protocol/resource guard unchanged. New qualifier saves actual final snapshots and signed-contact/posture NPZ for source-v8 exact boundary/mid-held-command fixed/closed-loop replay. Full153Linux regressions and4replay paths gate formal H2. See docs/V0.0.9_H2_ATTEMPT02.md and experiments/v0.0.9/scaling-attempt-02.json;11targeted Mac tests/Ruff pass, no newhost formal outcome yet. No game/learning/gate change; H3 remains pending.

### H2 complete; exact collector candidate ready

Roomier-host all78cases/10waves completed with62parallel histories exactlymatching serial,0invalid/incomplete/not_run; median16worker steady520.66controlTPS/startup-inclusive124.88. All4source-v8 native state/action/contact replay paths and153Linux tests pass; independent all-row/raw-tail audits pass, all71PIDs exited/source unchanged. Source/receipts in artifacts/v0.0.9/H2/attempt-02, preserve xbabe1's capacity-stopped attempt. No optimizer/referee/game is benchmarked.

H3implemented as an opt-in AllocationReducedTask, unchanged reference/default and physics/math/700muscles. Exact typed contact ownership, zeroed reusablewrench, fewer finite/warning temporaryarrays, nativefloat64-valued immutable contact scalars; every substep/check/evidence row retained.9Mac physical invariants+3harness tests pass/Ruffpass,165tests collected. Protocol/worker-ready docs/V0.0.9_H3_COLLECTOR.md freeze equal4worker/counterbalanced300control comparison, exactsource/rawtape/contact/replay and1.2x steady/.95startup gate. FullLinux suite and source push precede outcomes. An insufficient candidate is declined; v9remainsinprogress and subsequent motor-learning mission remains required.


### v9 concluded; next motor adaptation contract

All H0/N1/H1/H2/H3 registered scope is concluded in V0.0.9_RESULTS.md. Exact scalar rollout at16workers reaches520.66steady parentTPS, with78/78cases/62parallel histories exact and source-v8 hostreplay. H0batchingfailsparity, H3collector1.01829x fails1.2performancegate despite all24rawtrajectories/fourreplays exact. ReferencePhysicalMotorTask staysdefault. Main241,588physics/438.8812s, zerooptimizer/game; allownedPIDs exited. Allnegative/stopped/not_runrows/raw archive retained.165formalLinux regressions plus14post-runMacsubset and Ruffpass.

Next implementation owner follows docs/V0.0.10_MISSION.md. First offline input/failure audit, then separate motor_residual.py wrapper over unchangedphysicalmotor. Frozen deterministicteacher plus bounded138group correction mappedtoall700, frozen3601normalizer/decoder; ordinary latent-actionSAC critic/replay/entropy have explicit138contract. Do not blindly resume upstreamdecoder's independentlysampled562weights with group-onlylogprob. Zeroresidualsource-v8replay, physicalreward/terminalmask/workerownership and sourcefreeze gates precedetwo-seed8192-transitionpilot. Stationaryguard/reach/contact/history/referee/frozenleague and largerheldoutmotorproof remain unmet. No versionnumberis evidenceoflearning.


### v10 reference prerequisite

D0all7savedcase inputs/reset/current+future goals match, but reported native last-frame root velocity violates its unwrapped position derivative by119.46. Native resetgrid excludes this frame; cause of physicalcollapse remainsunproven. Registeredgate stops residualpilot. New opt-in reference_seam.py plus R1protocol qualify an explicit last-frame0/2velocity correction without editing native/default/positions/model/actor. All42rows/raw arrays and171Linux regressions gate R1; D1physical/source replay/likelihood/masks still follow, no learning yet. See V0.0.10_D0_RESULTS.md and V0.0.10_R1_REFERENCE_SEAM.md. Keep v10inprogress, original D0stop/tolerances unchanged.


### v10 neutral residual interface implemented; source proof active

R1reference correction qualified42/42rawcases withindependent audit and171Linux regressions;0physics, no causalbalanceclaim. New motor_residual.py owns138latentcommands mappedto700native muscles aroundimmutableteacher, explicit corrected-reference/content/normalizer/group/bundle hashes andphysical-alive reward/masks/terminalobservation. Pending snapshotsresume without newdecisions. Dtypemetadata restores originalreference arithmetic; newmidreplaytest caught an inherited~1.18e-8rewarddifference andcorrectedonlythenewwrapper. Seven localtests pass11.80sincludingexactneutralengine/contact/reward, atomicrestore/assistance/terminalownership, andfiniteordinarySAC138syntheticupdate/replay/target/save-load. No pilotoptimizer updates yet.

Source7acd207pushed/frozen; D1bounded full178Linux regressions then4source-v8physicalpaths active onxbabe3,wrapper3601585. Runtimecaps60s/12kphysics/32MiB/oneworker/thread;10,394expectedphysics. Noexecutingsourceedits/duplicates. Independent4rowrawtypedstate/action/contact/telemetry/reward/mask/counter audit andD2pilot/evalmachineprotocol remainrequired. See V0.0.10_D1_NEUTRAL_INTERFACE.md/MISSION. No default/third_party/model/rules changes orv10tag.


## 2026-10-27 03:39UTC: D1 physical interface concluded; D2 foundations

Executed7acd207ca69784935657d5f6b2933aff5a23e3f9 on xbabe3:/home/ubuntu/joxing/v010-neutral-20261027. Full178Linux tests passed248.40s (supervisor250.893544s); formalD1all4boundary/mid × recorded/closed-loop paths exact,10,394physics/21.369797876104712s,0invalid/incomplete/not_run/optimizer/boxing games. Independent verify_motor_residual.py reads all4actual typed starts/finals,700-command tapes,every signed contact/load/posture tail and control trace, reconstructs physical reward/termination and compares native reward across paths. No old v8native-reward oracle is invented; per-control clipping/held-no-new-decision/bundle immutability rely on frozen executable assertions, masks/terminal ownership on unit tests. Audit0newphysics. All6recorded PIDs exited/ownedscanempty/source diffempty03:39:55UTC. Entire6,759,968byte formalraw plus independent audit and manifest are in artifacts/v0.0.10/D1; source checkout unchanged. No motor competence or causal reference-fix gain claimed.

New motor_pilot.py and motor_evaluation.py implement the D2foundation: disjoint counter-based physical reset streams, fresh zero-initialized ordinarySAC138 construction, persistent private worker chunks/reset/transition/progress/rawfault accounting, fixed32attempt paired conclusion, actual initial/boundary/mid/final held-out evidence and exact source-host scalar closed-loop continuation. Four worker/SAC/denominator fixture tests pass13.99s;2evaluation/replay/fault fixtures pass5.18s;2independent D1corruption tests pass.41s. Synthetic real-gradient update and synthetic-teacher physical fixtures remain separate from formal pilot counts0. Initial replay fixture1failed/1passed5.50s: all447continuation steps/control traces exact, but direct Python equality rejected tuple contacts versus transported JSONlists; canonical JSON digest fixes representation equality without numeric tolerance changes. Failure/repair retained artifacts/v0.0.10/D2/development-validation.json.

Current next packet docs/V0.0.10_D2_PILOT.md specifies exact APIs/read-first/forbidden paths/source freeze/caps/implementation/tests/logging/handoff. Parent scripts/train_motor_residual.py, independent scripts/audit_motor_residual.py and experiments/v0.0.10/protocol.json do not exist yet. Implement/test/freeze them before formal8192×2transitions/four workers/32heldout attempts/two replay reserve launches. No duplicate source jobs or permission/agents needed. D0/R1/D1formal25.066974967718124s/10,394physics; mainremaining1774.933025s/539,606physics,50kauditreserveunused. v10inprogress/untagged; v9latestcompleted.

Full current local regression suite: **186passed160.74s**, Ruffpasses. This includes the new8post-D1tests; executed D1Linux source remains178tests, with no fresh D2Linux claim.


## 2026-10-27: three-plan review and integrated self-play execution route

User asked for a detailed plan from the new Joxing master/test documents and Jarambit scientific audit. All3inputs are byte-preserved withSHA256identities in docs/references/2026-10-27 and artifacts/reviews/2026-10-27/input-manifest.json. SELF_PLAY_EXECUTION_PLAN.md supplies concrete read-first/files/APIs/ownership/invariants, finite worker packets, tests/receipts/completion/stop conditions and routes from fixedv10question to connectedgame, rule/zone correctness, causalhistory, cheap learner/league controls, boxingmotors and1k→10k→100kvalidnativegames. V0.0.11_MISSION.md is registered next but inactive untilv10concludes. No supplied health/KO/professionaljudging/directmovementforces/largecompute instructions were adopted. Jarambit's transferable contribution is evidence/status/claim separation, not its unrelated source instructions or results.

Exploratory read-only static probe scripts/audit_self_play_readiness.py:01failed on guessed nonexistent foot body names before anyphysics;02computed2.793576s with a hand/calcaneus subset;03computed2.982173s with pinned teacher's exactsixkeybodies. All3attempts retained;0newmj_step/actor/optimizer/boxinggames. Geometrybothcomputedruns64head/79native_non_scoring/10guard from153neutral-pose first-depth rays;4mm sampling/tie rule and scratch placement limitations explicit.03world-axisfeaturedifference0.4018903313m,yawcanonical4.44e-16m;own glove/forearm-head collisionbitsdisabled,opponentpairsenabled;native skull collidable/non-scoring;spawnrightglove-headgap1.687933m;implicitfast1ms/tol1e-10 differs from teacherEuler2ms. Static findings do not establish physical punch awards, motor invariance or fullcoverage. Reports preserved with explicit dirty-tree source-context annotation. Corrected SPEC_REVIEW's prior skullvisual-only wording; no sourcephysics/geometry/rules/default changed.

Duringheartbeat before the new request, added unqualified parentdraft motor_run.py/train_motor_residual.py plus bounded motor_storage.py and extendedprivateworker/evaluation output credits. Three parentfoundationtests pass6.93s;combined9motorfoundationtests pass12.60s. Full local189tests passed170.04s; final full regression receipt follows after small parent attempted-row/uncertain-bound/partial-constructor cleanup refinements. No independentD2outcomeauditor ormachineprotocol exists yet, so no formallearninglaunched. Exactsource/protocol/workerfault/counter/PID qualification remains the current concrete task. No v10tag/nativecompetitiveclaim or changedpilotcriterion.

Final current local regression receipt: **189passed163.78s**, Ruffpasses; exactsourcechecksums/local-validation-final.json in artifacts/reviews/2026-10-27. Fresh04:20:11UTCxbabe3ownedscanempty. No formalD2pilot ornativecompetitivegame launched.
