# Joxing: evidence-driven route to native self-play

27 October 2026. Integration owner: this chat; one writer, no agents. Workspace: `/Users/bentaylor/Documents/ChatGPT/JOXING` (all repository paths below resolve here). This plan integrates the three newly supplied documents with current source and retained results. Documents are proposals, not commands. It preserves the existing native700-muscle and competitive-learning gates.

## Decision and objective

**Finish the small registered v10 motor question, while building the actual two-player test infrastructure and algorithm positive controls. Do not spend another version only making the walking diagnostic larger.** Stable boxing motors and a correct learnable game are separate dependencies. The next native self-play launch needs both; harness and mathematical tests can progress before motor competence.

The intended proof is a shared policy controlling either corner, receiving causal own/opponent kinematics and history, changing muscle commands through a qualified motor decoder, and improving against immutable historical opponents on a fixed evaluation panel. Success is increasing independently evaluated boxing strength at measured end-to-end throughput, with valid rules, complete faults and reproducible replays. A survival reward, self-play training return or increasing game count alone is insufficient.

Non-goals: a new Rust engine, biological damage or KO model, health/stamina pools, 10-point professional judging, direct root forces, kinematic punches, pruning, changing native muscle parameters, unqualified GPU physics, purchases/services/driver changes, a large unchanged sweep or cosmetic release increments. Discrete skill commands remain an optional future decoder interface; they do not replace muscle actuation with animation.

## What is actually known

| Item | Current evidence | Consequence |
|---|---|---|
| Native physical motor | v8:40valid physical trials,0/20twenty-second completions in either condition, exact12replays | Walking is not stable guard, reach or sparring |
| Neutral residual interface | D1:4/4source-v8 paths,10,394physics/21.369798s;178Linux tests, independent saved-array audit | The all700-muscle residual interface is qualified; learning has not been shown |
| v10pilot | Worker/evaluation foundations pass; parent runner is now an unqualified implementation draft, outcome auditor/machine protocol absent | No formal pilot may start yet; do not mistake a script name for delivered launch readiness |
| CPU throughput | v9:520.66one-human scalar controlTPS at16workers; frozen batching failed, collector1.01829x declined | This is not two-boxer training/game throughput |
| Two-player learning | Scene/contact/referee pieces exist; no connected BoxingEnv, opponent history or league | Valid native boxing games remain0 |
| New geometry probe | One neutral pose,153sampled directions:64head/79native_non_scoring/10guard first contacts | Head-surface qualification is urgent; a static query is not a physical scored punch |
| Corner frame | Exact upstream six-key-body subset differs by0.401890m in world axes; yaw-canonical difference4.44e-16m at reset | Actor/body frames need explicit qualification; this is not proof the teacher works at arbitrary heading |
| Collision/physics contract | Own glove/forearm-to-head masks off; opponent masks on; sceneimplicitfast1ms/tol1e-10, native teacherEuler2ms | Match-scene motor qualification cannot inherit the native result |
| Spawn | Current right-glove/head surface gap1.687933m | Striking-range curriculum needs a declared reachable spawn; do not equate pelvis distance with reach |

The new exploratory audit uses zero `mj_step`, actor or optimizer calls. It explicitly moves a scratch sphere's geometry for distance queries, with4mm depth sampling and a stated tie rule; it is not evaluated boxing or an assisted learning result. Raw records and both exploratory attempts are in `artifacts/reviews/2026-10-27/`. One earlier attempt failed on guessed body names; it is retained in the attempt register. Exact current key bodies are read from the pinned teacher source. No geometry or collision setting was changed.

## Disposition of the three inputs

Copies and SHA256 identities are under `docs/references/2026-10-27/` and `artifacts/reviews/2026-10-27/input-manifest.json`.

| Input | Adopt | Correct, adapt or defer |
|---|---|---|
| Joxing RL master specification | Yaw-based ego frame; common decision boundary; separate recurrent state; learner-vs-frozen ownership; historical opponents; exploit probes | It assumes health pools, scripted punch phases, stun, swept capsules and10-point judging which this repo does not implement. Its world movement force is incompatible with unassisted muscle boxing. Its64features omit most muscle proprioception and some feature groups undercount their fields. Its50M-step/64–128world plan is not a launch authorization |
| JOXING_SELF_PLAY_TEST_PLAN | Build the harness now; zone/collision/heading checks; rule exploit battery; algorithm positive controls; match-specific motor transfer; full-loop benchmarking | Its probe counts/timing were external claims until checked here. Our static geometry counts agree; its726physics steps/s is still not our measured full workload. A confidence interval merely containing zero does not establish corner equivalence. A cheap-body positive-control failure does not uniquely locate a bug |
| Jarambit scientific audit | Separate immutable observations, annotations and claims; per-observable identity/status; complete requirement crosswalk; source/runtime/evaluator hashes; sampling/selection limits; cold/warm and full-pipeline reporting | This audits another project. Its Rust requirements,29test count, design/FE/mesh results and publication campaign do not apply to Joxing. Borrow a small evidence schema and claim checks; do not make a large reporting rewrite the critical path to boxing |

Further corrections to the proposed RL design:

- A dense term with coefficient floor0.05 does not vanish; damage/ring-centre bonuses can change the game even when they sum to zero. Keep canonical terminal match utility and its verified potential shaping. New shaping is a named ablation with its own objective, not a silent anti-passivity fix.
- With gamma1 and terminal potential0, the sum of potential differences telescopes exactly. With another gamma use the matching discounted transformation and terminal/bootstrapping treatment. The master proposal's varying annealing and general-sum terms do not inherit a blanket Nash-policy invariance guarantee. See the original [reward-shaping paper](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) for its stated MDP result, rather than extending it by analogy.
- A many-to-one environment action transform is not automatically an invalid policy-gradient problem. The sampled action and its matching behavior log probability must remain paired; replacing the stored action while retaining another action's log probability is wrong. If discrete skills later use masks, store and reconstruct the exact mask; an all-invalid mask is a contract fault, not a silently repaired training example. PPO's behavior-policy objective is the reference, not the supplied illustrative code: [PPO paper](https://arxiv.org/abs/1707.06347).
- PFSP is a tool for diversity, not a guarantee of convergence. Use an explicit uniform probability floor and sparse/empty-pool rules. Distinguish historical snapshot archival from champion promotion; rejecting every weaker snapshot biases the archive. A frozen panel and a payoff matrix expose forgetting. [AlphaStar's primary report](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/) motivates the league approach, not the proposed boxing thresholds.
- Both same-bundle corners can contribute actor samples when the complete behavior bundle, normalizers, stochastic transforms and likelihood computation match. The samples share a world and are statistically dependent. Historical-opponent data are excluded from ordinary learner PPO actor loss.
- A balanced RPS oracle cannot admit a policy beating all three pure strategies85% of the time. Test the known payoff matrix and exploitability of a time-averaged mixture, not an impossible universal bot win threshold. Actual boxing anchors need measured discriminability and achievable targets.
- A policy's average two-corner zero-sum return being zero does not mean every useful learning metric is flat. Report per-opponent/frozen-panel performance, trajectories and strategy coverage. Antisymmetric exact values require a matched joint-policy/conditioning oracle; do not force arbitrary approximate critics to be antisymmetric under different hidden opponent distributions.
-15/20Hz does not mean3–4physics ticks at1kHz. Start with an explicit integer tick schedule; native50Hz motor is not silently changed to100Hz. Decision cadence is distinct from reaction delay, muscle activation and command queues.

## Invariants, ownership and read-first order

Read in order: AGENTS.md; `/Users/bentaylor/.codex/RTK.md`; README; docs/{MISSION,STATUS,GAME,SPEC_REVIEW,IMPLEMENTATION,PHYSICAL_MOTOR,STATE_HISTORY_SELF_PLAY_AUDIT,SELF_PLAY_HPC_PLAN}.md; current v10mission/D2packet and experiment book; artifacts/v10D1audit; the three frozen inputs; src/joxing/{scene,physics,rules,motor_residual,motor_pilot,motor_evaluation,motor_run}.py; tests for those modules.

Forbidden writes: `third_party/`, executed source checkouts, historical protocols/receipts/raw, v8/v9runtime locks and existing qualified motor defaults. Generated MJCF goes in `artifacts/generated/`; large/raw runs/checkpoints in `runs/`, not source or submodules. Reference anatomy retains700muscles per athlete. Every new geometry, own-contact, integrator, timestep, scoring or assistance choice gets a named variant and its own evidence.

One owner integrates `scene`, referee, observation and replay schemas. Logical workstreams below may progress independently only with disjoint files and no overlapping simulation writers. This plan does not authorize agents, other-chat messages or simultaneous launches. During a healthy owned run, executing source stays fixed; only independent analysis/tests/docs change.

## Ordered packets and acceptance

### P0 — Conclude v0.0.10; no larger walking sweep

**Question:** does the fixed small residual improve survival on both registered seeds? Keep the master mission's two81011/81029seeds,8192transitions each,four persistent workers,32fixed initial/final evaluations and two final source-host replays. Retain the original directional gate: each seed mean paired survival gain>=.5s and>=4/8final five-second survivors, zero invalid evaluations, both exact replays. Passing still proves no guard/boxing competence.

Files: finish `src/joxing/motor_run.py`, `scripts/train_motor_residual.py`; create `scripts/audit_motor_residual.py`, `experiments/v0.0.10/protocol.json`, supervisor integration tests; maintain `motor_storage.py`, `motor_pilot.py`, `motor_evaluation.py`. The parent draft now has shape-only-to-four-env handoff, bounded pool cleanup, separate evaluator/replay phases and pre-write raw credits. It is not qualified. Audit source/config constants, private worker counters, crash bounds, model/optimizer identity, warmup/update cadence, source-host replay and all output credits before launch.

Specific remaining hard parts: source/protocol checks must be independent of a mutable co-located protocol; register actual sampling/cadence settings; assign full worker PID/start-time ownership including partial constructor failure; correct confirmed-versus-allocated progress at every event; retain partial commands if an exception occurs after physics but before a transition write; bind checkpoint hashes and optimizer states; distinguish main physics/time from audit reserves; close pools before single-worker evaluation; avoid a blocking close after child death. Report measured costs rather than claiming every timing component is separately available.

Main version550kphysics/1800formal seconds/1GiBraw; prior10,394physics/25.066975s; separate50k/300saudit reserve. Draft disjoint raw credits are192MiBparent,48MiBpertrainingworker×8,8MiBperevaluation×32,8MiBperreplay×2 plus32MiBconservativeprior charge =880MiB including local fault reserves; verify the actual prior charge and per-writer temporary-file peaks. Global max4simulation owners. Inspect fresh capacity, freeze/push protocol/source, full pinned Linux tests, then one bounded launch. Preserve a cap/fault stop honestly; no changed objective or silent retry. Finish README/docs/book and audit/publication; tag only completed scope.

**Decision afterward:** whether positive or negative, move to boxing-specific motor goals and game infrastructure. A failed residual pilot should select a measured cause, not an unchanged budget increase.

### P1 — v0.0.11: first simultaneous two-player test environment

**Objective:** an executable, replayable two-player loop using all1400muscles, shared physics, causal observations and connected candidate points referee. It is a diagnostic environment; unresolved surface/foul/motor gates keep competitive training disabled. It need not wait for a competent boxer.

Create `src/joxing/env.py`, `observations.py`, `history.py`, `referee.py`, `replay.py`, `bodies/base.py`, `bodies/native.py`; `tests/test_{boxing_env,observations,history,match_referee,match_replay}.py`; `experiments/v0.0.11/{protocol.json,README.md}`; `scripts/qualify_boxing_env.py`. The next mission is `docs/V0.0.11_MISSION.md`. Keep `scene.py` unchanged initially; call its existing composition profile with its actual1msimplicitfast contract. A later validated match variant cannot masquerade as the native2ms teacher task.

Minimum APIs:

```python
BodyAdapter.command_ids(player) -> ndarray  # disjoint700 IDs, union1400
BodyAdapter.measure(data, player) -> BodyMeasurements
BoxingEnv.reset(seed, scenario_id) -> {player: ActorObservation}, info
BoxingEnv.step({"A": command700, "B": command700}) -> ParallelTransition
MatchReferee.stage(contact_evidence, start_tick) -> tuple[CandidateEvent, ...]
MatchReferee.commit(events, end_tick, bell) -> MatchState
ReplaySnapshot.capture(env, actor_states, queues) -> snapshot
```

`ParallelTransition` includes per-player owned obs/reward/termination/truncation/info, shared game ID/tick, immutable behavior IDs and diagnostics. Direct normalized700commands use an explicitly hashed transform exactly once; zero normalized command is not zero excitation. Later tactical latents use a frozen decoder, not a different physical actuator path.

Step order: freeze both observations at tick t; collect both commands; validate episode/player/decision IDs and bounds; write only owned actuators; advance one shared MjData through `step_contacts` at every physics substep; stage all glove events from the same physical evidence; commit the ledger and any global ending after both sides' events are gathered. Never independently integrate one boxer and apply its hit before the other; do not add a second scripted knockback impulse on top of MuJoCo contact forces.

Initial control interval20ms over20×1ms substeps; this is a diagnostic acting schedule, not teacher transfer. Subsequent cadence choices require named compatibility evidence. Placeholder controllers issue bounded muscle commands; no evaluated pose writes or external balance force.

Tests: perturb A's chosen action and prove B's same-tick input/decision is unchanged; ownership and muscle-count preservation; permute candidate-event enumeration with identical ledger; real contact-to-referee forwarding each substep; true bell finalization; collector cut preserves live state; global ending terminates both; invalid physics quarantines both. Snapshot engine INTEGRATION plus clocks/score/hit trackers/contact onset/held commands/history/queues/RNG/recurrent state. Restore during contact and mid-held-command, require identical event IDs, scores, terminations and full continuation state under one recorded source-host acting schedule. Grouped2-corner batching is a new contract; do not retroactively pass v9H0 or assume trained scalar/batch equivalence.

Proposed preregistered diagnostic cap:2workers/one thread,120kphysics/900s/256MiB, firstlimit; no overlap with v10formal compute. The worker must finalize the finite case schedule and exact invariants before source freeze. Expected outputs: protocol, every attempted case, actual snapshots/control/contact/event tapes, counter/fault/exit receipts and an independently checked result. This packet is complete when the loop and its correctness tests execute reproducibly, with unresolved readiness gates explicit; valid competitive-game count remains0 until the full gate passes.

### P2 — Make the sport learnable and non-exploitable

Files: new `src/joxing/zones.py`, `scene_variants.py`, `rules_variants.py`; extend `referee.py`; `scripts/{map_scoring_zones,probe_rule_exploits}.py`; `tests/test_{zone_coverage,rule_exploits,ending_incentives}.py`; `docs/RULES_V1.md`; per-experiment protocols/results.

1. Expand static maps to reviewed guard/crouch/head±30° poses and torso/belt/back regions. Measure angular coverage and unresolved seams. Confirm with force-producing scripted contact rigs at several speeds through the actual contact/referee adapter; direct scratch queries cannot award points.
2. Prefer first testing a **semantic native-head-surface variant**: classify actual skull contacts by local front/side/back contact location and orientation. Do not relabel the entire skull legal: that could reward illegal back-of-head contact. A shell/envelope/collision alternative changes physical forces and needs separate convergence/replay evidence. Avoid double-counting overlapping manifold/zone impulses.
3. Review own-body pairs, masks and existing native excludes. Measure penetration in guard/motion scenarios. No embedded forearm may form invincible guard inside the head. Define pair-specific exclusions for intended anatomical overlap rather than enabling every self-contact indiscriminately. Preserve native inertia/muscles.
4. Script minimum-touch speed/impulse, high-frequency peck/rearm, guard-first-then-head, glove tie-up, pushing/clinch, belt/back hits, same-tick trades, final-substep impacts and hits during a fall. Measure points/time and each reason. Cheap touches are not automatically a bug: decide the intended rule from the data, then freeze a named rules revision before learning.
5. Freeze an ending utility table: bell higher points wins; a self-caused valid fall/exit loses even when leading; simultaneous independently valid losses draw; an attributed illegal shove cannot reward the offender; ambiguous/unresolved attribution blocks valid competition. A brush is not automatically a fall. Attribution is a declared rule using measured evidence, not biological causation. Numerical invalids stay invalid, never a draw/win; action-correlated invalids halt promotion.
6. Keep head2/torso1/guard0 and force/impulse telemetry. No health pools, injury or professional judges are introduced. Proposed geometry gate:>=95%coverage over a preregistered intended legal region,0unintended legal awards, every remaining seam explicitly listed; protocol must define samples, solid-angle weights and diagnostic tolerances. Post-outcome exclusions cannot rescue the gate.

Validation: table-driven endings and telescoping utility for every real termination; history-preserving collector cuts; real final observation for external truncation; gamma mismatch rejects learner construction. Full physical/rule replay and dt sensitivity are required for a new scene/rules variant. This is a separate finite experiment, not tuning weights while a learner exploits them.

### P3 — Causal state/history and a shared two-corner policy

`ActorObservation` schema is generated from named fields and slices, not a magic64-vector: own joint/muscle state; own head/torso/glove kinematics; opponent visible head/torso/glove/pelvis pose and linear/angular velocity; ring geometry; public clock/score; previous applied command; sensor age and validity. Keep semantic left/right muscle identity under corner rotation. Opponent controls, hidden state, private goals, RNG, future queued commands and bundle identity never enter actor features. Critic-only full state is a separate type/interface.

`TimedHistory` uses `(game_id, player_id, sampled_tick, received_tick, valid_mask)` with `sample_time <= decision_time <= apply_time`. History may include only past samples. Implement a zero-delay reference and deterministic action/observation queues now, with tests for one/two-tick delay, reordering, staleness, episode reset, asynchronous corners and dropped samples. Random jitter robustness is a later ablation, after the zero-delay oracle learns. This meets the user's history requirement without introducing untested latency into first-signal learning.

Yaw frame: transform positions relative to own pelvis; rotate linear/angular vectors consistently; preserve gravity/tilt/height and relative orientation. Distinguish yaw-only from full-pelvis frame so a fall does not appear upright. Global rotation/swap tests use actual supported reset/kinematic states; arbitrary array rotation can violate the scalar-root representation. No left/right muscle permutation for a180°corner swap; orthodoxy/southpaw reflection is a different experiment.

The frozen3601teacher is not made invariant by silently replacing its input. Create a separately hashed canonical-frame adapter, requiring theta0 baseline parity and translated/yawed paired tests; if features cannot preserve its normalizer/reference semantics, train/requalify a new motor bundle. Qualified native snapshot/input arithmetic stays unchanged.

Recurrent policy/critic state keys are `(game_id, player_id, bundle_id)` independent of current batch row. Start with a small MLP+GRU and explicit sequence burn-in/masks; choose hidden size and history span in the positive-control protocol. Store behavior recurrent state, sample logp, latent, action transform and mask if applicable. New weights require a documented recurrent-state refresh/freeze policy; no stale-state carry between bundles. A versioned frozen decoder must map latents to700native muscles without bypassing motor/contact gates.

Tests: arbitrary valid asymmetric corner swap; no critic leakage; timestamp causality; separate A/B history; stale-reply rejection; exact episode reset; batched row permutation preserves ownership; padded/burn-in sequence masking; rollout/chunk/terminal bootstrap distinction; saved likelihood reconstruction and finite gradients; source-host recurrent mid-match replay. Statistical corner equivalence uses a preregistered acceptable bias interval and a CI wholly inside it, not merely a CI containing0. Keep pathwise symmetry/divergence horizon separate from ensemble equivalence in chaotic physics.

### P4 — Prove the learner and league cheaply

Create `src/joxing/rl/{actor,critic,ppo,rollout}.py`, `league.py`, `evaluation.py`, `opponents.py`; `tests/test_{ppo,league,evaluation,positive_controls}.py`; `experiments/self_play_positive_controls/`. Reuse exactly these modules later with the native body adapter.

Order: known-payoff algorithm fixtures → a small separately named capsule-body diagnostic with its own stable motor/contact/rules/replay gates → labelled assisted700-muscle reach/sparring only after its applicable gates. None qualifies native unassisted boxing; assistance is a model identity and cannot be silently removed while claiming the same qualified motor. Do not bypass AGENTS' competitive gates merely by renaming a run a positive control.

Known-payoff fixtures verify exact reward signs, frozen-vs-current actor ownership, update/logp correctness, reset state isolation, payoff matrix and exploiter calculations. Balanced RPS evaluates population exploitability against its known equilibrium, including a deliberately cycling latest-only baseline. Capsule/assisted studies use three fresh seeds under fixed small caps, an unchanged discriminable panel and real optimizer updates. A failure can implicate task design, learning dynamics or implementation; diagnose rather than claiming anatomy is exonerated.

League bundle includes actor, decoder, normalizers, history/cadence/queues, action transforms, precision/acting schedule and content hashes. Freeze at match reset; no in-place opponent or EMA update mid-match. Archive predeclared snapshots, including weaker policies; champion promotion remains separate. Proposed initial opponent mixture50%current/latest,35%history,15%anchors is a testable starting protocol, not a requirement. History probabilities use `eps/K + (1-eps)*normalized_weights`; handle K0,all-zero weights,0/1winrates,unseen opponents, malformed/nonfinite stats, explicit priors and per-game RNG. Training EWMA/Elo are diagnostics; promotion uses frozen evaluation.

When both corners have the same complete behavior bundle, admit two distinct rollout columns with separate hidden/RNG states. Historical corner actor-loss mask is0. Never relabel an old opponent's actions as learner-on-policy data; never update on a fictitious mask or decoded-action logp. Count world transitions and admitted actor samples separately.

Required proof: all ownership/likelihood/temporal/replay tests pass; fixed-budget learning improves against the preregistered panel across three diagnostic seeds; payoff coverage and fixed-budget exploiters expose cycling/forgetting. Do not set an unmeasured85%-against-all-bots requirement. Formal method comparisons later use the existing five-training-seed convention and appropriately powered paired scenarios.

### P5 — Train boxing motors, not just gait survival

Files: `motor_goals.py`, `motor_skills.py`, `motor_frame.py`, `eval_motor_skills.py`, `tests/test_{motor_goals,motor_frame,motor_skills}.py`; named static/guard/reference variants and protocols. Existing qualified teacher/default remains read-only.

Train/test: stationary stance and raised guard; independent left/right reach/return with contralateral guard; stepping/pivot/stop; perturbation recovery; then qualified latent compression. The next goal variant removes continuous walking as the desired behavior; use feasible joint/equality-consistent targets, not zeroing a gait frame and assuming it is a valid stationary controller.

Every motor must be requalified in the exact match scene, equipment, self-contact and numerical settings. Compare with opponent far away; either exact parity or a named transfer variant with its own result. Do not infer independent70ms reaction or a motor response threshold from inference FPS. Measure observation-to-decision, queue delay, muscle activation, glove response and contact-to-guard response separately.

Before10s curriculum: proposed screen>=95%20s guard completion and>=90%reach/return success over200held-out trials per registered motor condition; glove/contralateral error<5cm; report confidence intervals and failures. These observed proportions are screens, not claims of a guaranteed population rate. Extend to60s before60s rounds; a full-round motor gate should match the actual round length. Keep existing v10gates unchanged. Independent physical-model/injury validation is not implied.

Use impulses measured from qualified contact diagnostics to set recovery test distributions, including the measured90thpercentile and head/chest/glove sites. Measure reach-workspace coverage and legal-hit first-signal rate with a one-sided interval bounded above0 before large self-play. If initial hit rate is effectively0, use reachable spawns, qualified skills/scripted partners and short rounds; do not expect100kzero-zero games to create the signal. Preregister response-time targets after measuring strike durations. Before native curriculum, proposed fall-decided-game rate<10%on a fixed scripted-sparring panel; report actual denominator and reasons.

### P6 — First native league:1,000 then10,000 valid mini-games

Require P1–P5's native contact/rules/motor/replay/state/learning gates. Use a separately named5–10s striking-range curriculum; no footwork requirement is hidden in an unreachable default spawn. Preserve native700percorner, muscle actuation and canonical points/utility. Independent per-corner action RNG breaks deterministic lockstep; RNG IDs never enter actor input.

Before launch freeze: seeds/scenarios, opponent mixture/history floor, candidate checkpoints, replay cadence, invalid handling, stopping rule, held-out panel and compute ceilings. Training count targets are subordinate to hard physics/wall/disk caps. Start four workers, rather than64/128promised worlds. Promote only with all invalid rows retained, exact required replays, no critical regression and a positive lower95%paired panel-score improvement bound. Size panel pairs from pilot variance and a declared smallest useful effect;200is a starting design, not universal power. Define whether panel score is win/draw/loss utility[-1,1] or points[0,1] before computing variance/sample size. Do not pool reset pairs as independent training seeds.

Report fixed-panel utility versus wall time and world transitions; paired seed-level gains; win/draw/zero-zero rates; hit/miss/blocked counts and zones; attributed falls/fouls/exits; engagement distance; action/latent entropy; seam-contact contribution; numerical-invalid rate; payoff matrix and fixed-budget best-response exploiter score. Flat panel strength with rising shaped reward is a diagnosis, not a reason to scale. A dedicated exploiter gives a scoped exploitability proxy, not proof of Nash equilibrium or absence of other exploits.

Run10konly after1khas both valid mechanics and measured panel progress.100krequires a new protocol justified by the measured bottleneck and learning slope; no automatic spending simply because a counter reached10k.

### P7 — HPC throughput matched to complete workload

Use native one-thread CPU physics first. Fresh03:55UTCsnapshot: xbabe3ownedscanempty,111.25GBavailableRAM/44.00GBdisk,load9.64on128logical CPUs. This is a historical capacity observation, not a future reservation. Reinspect before every launch. Dedicated~/joxingcheckouts; global16owned simulation workers total across hosts, and tighter packet limits. Two-human memory must be measured anew; do not reuse a one-human2GiBreservation without evidence. xbabe1/2are alternatives for independent seeds after fresh checks; xbabe4nativeCPU is useful for development/replay/rendering with a separately qualified platform contract. Docker is not a default speedup.

Measure1/2/4/8/16workers subject to resource headroom on the **complete** world: both motors/inference, every required physics/referee interval, causal histories/queues, resets, IPC, optimizer, logging and evaluation. Report startup/cold,steady,warm and full-wall cost separately; median repeats, peakRSS, fault and skipped rows. Preserve the same physics/rules/model and policy schedule during a throughput comparison. Every candidate optimization must pass retained fixed-action/state/event oracles; an unchanged-looking video is insufficient.

First candidates after profiling: persistent workers/model replicas; owned preallocated/shared buffers with episode/tick IDs; batched inference under a newly declared acting schedule; sampled persisted training telemetry with complete online impulse/referee reductions and deterministic audit/fault capture. Never remove contact or referee steps to advertise FPS. Profile optimizer/inference before selecting CPU versus GPU for networks. Native GPU physics remains behind exact-model feature and contact/referee parity gates; model surrogates have separate identities and no native promotion.

Capacity calculation uses measured aggregate complete-workload world transitions/s `T` at `f`decisions/s:

`hours = valid_games * mean_game_seconds * f / (T * 3600 * valid_fraction)`

This is only a first-order estimate: report length distribution, startup/evaluation/checkpoint overhead and invalid rates, with a direct held-out end-to-end wall benchmark as authority. At50Hz,100k×10s contains50Mworld transitions;100k×60s contains300M. Two corners do not double world steps; current-vs-history may admit only one actor stream. A raw dual-physics speed from another core and a one-human rollout speed cannot fill in T. Select workers from actual throughput, tail latency, RAM and host load; require useful learning gain per wall hour, not maximum simulator FPS.

## Minimal evidence contract; no reporting detour

Add `src/joxing/evidence.py`, `scripts/validate_evidence.py`, `tests/test_evidence.py` only to cover actual new experiment requirements. Reuse provenance.digest/manifest/write_json; do not replace all old receipts. Keep original files byte-stable; retrospective annotations declare `exact_from_record`, `derived_from_pinned_source`, `inferred_with_limit` or `unknown`.

Run identity binds source/protocol/config/runtime, model/rules/observation/action/cadence/inference/bundle hashes, source/host/platform, thread limits, ownership/PIDs/start times, seeds and all counters. Per-observation fields: metric, value,units,status,method,denominator,model/artifact/behavior identity,source receipt and limitations. Missing/not_run value is null, not0; a raw nonfinite failure belongs in fault NPZ with statusinvalid. Keep measured outcomes separate from interpreted gate/claim support. A filename/hash declared by a run is not automatically a hash independently verified by the auditor.

Adverse tests: delete an unfavorable row; change a score/rules hash; substitute a different actor/normalizer or scene; present a diagnostic as native; remove a replay; confuse world transitions with actor samples; replace not_run with0; mutate a source receipt; truncate JSONL; reorder JSON keys; corrupt raw contact sign/dtype; change a plotted artifact while leaving its claim. Verification must reject each inappropriate promotion/join. Generate README tables from compact audited summaries where helpful; a separate publication platform is unnecessary.

Each run retains protocol before outcomes, commands/versions/seeds/hashes, progress/events, raw faults/last valid state, completed/invalid/incomplete/not_run rows, source-host tapes/snapshots, timing/resources and process-exit/source-clean receipts. Compact all-attempt outcomes and checksums stay in Git; full raw/checkpoints in bounded runs and public release assets when the version concludes. Claim hierarchy: software fixture → numerical consistency → scoped physical-model experiment → learned motor → valid native competitive learning. Evidence cannot jump levels by relabeling it.

## Validation, stop rules and release handoff

Current executed commands work in the stated workspace; future commands require their named files/protocol to be implemented first:

```text
rtk proxy uv run --project experiments/v0.0.10/runtime --locked pytest -q
rtk proxy uvx ruff check src scripts tests
rtk proxy uv run --project experiments/v0.0.10/runtime --locked python scripts/audit_self_play_readiness.py --out runs/reviews/2026-10-27/new-unique-static-audit.json
```

For P0use the D2packet's train/audit commands only after its outstanding prerequisites. For P1the planned command is `python scripts/qualify_boxing_env.py --protocol experiments/v0.0.11/protocol.json --out runs/v0.0.11/harness-01`; it does not yet exist. Each later packet must similarly commit a finite machine protocol and source before formal execution.

Global stop: any nonfinite/source/bundle drift, unsupported model/rules variant, assistance during native evaluation, leaked future/private opponent information, altered action likelihood/mask, contact/referee/replay mismatch, unknown writer/PID, first cap/resource exhaustion or action-correlated invalid. Keep faults and every unrun planned row. No changing weights/gates after seeing outcomes. Logical experiments can conclude negatively; a release tag only marks a completed scoped question, not claimed competence.

Supervisor: inspect current Git/mission rather than old version numbers; check actual owned processes and fresh physics/transition/optimizer/event progress; do independent work while healthy source remains fixed; reconcile all attempts on completion. Remain quiet for unchanged healthy status. When idle pick the next finite packet above; do not automatically launch a large sweep. No extra approval or agent delegation is needed for the authorized work.

Handoff: exact source/protocol/model/bundle hashes and source-clean status; all gates achieved/unmet; counts and counter bounds; commands/tests/timings; raw/compact receipt locations; remaining budgets; actual host/process/checkpoint if active; next measured bottleneck. Update README/GAME/IMPLEMENTATION/STATUS/book and public Git before a completed micro-version tag.

The first clear wins are therefore: conclude the fixed residual question; run the first connected two-player loop; prove legal-zone and ending behavior; demonstrate a cheap positive-control league improving against a frozen panel; qualify guard/reach/recovery in the match scene; then spend on native self-play and scale based on its measured learning throughput.
