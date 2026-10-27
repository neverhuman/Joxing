**Current next packet:** SHARED_MOTOR_PHYSICAL_PROOF_PACKET.md. Original-host
archived input/normalizer/next command oracle passed all4rows after422Linux tests;
independent byte/hash/exit checks pass. New shared motor core/reference reset/
20ms commands/atomic replay has11passing Mac fixture checks,48targeted integration
passes. No actual shared pretrained trajectory yet. Implement independent physical
proof/evidence/runner/auditor/video gates next; do not rerun the completed oracle.

**Next executable packet:** SHARED_MOTOR_INPUT_ORACLE_PACKET.md. Actual native
cached3601samples, immutable replay serialization and scalar paired frozen
teacher inference are implemented;25Mac tests pass. A registered422Linux gate
then0physics archived input/normalizer/next-command oracle on originalxbabe1
comes before the independent shared-model geometry/cache/reset/reference proof.
D1sensor proof is complete across18attempt rows, not a learned motor claim.

**Implementation started:** `src/joxing/shared_motor.py` now supplies semantic
NativeMotorMap, immutable MotorReferenceState and read-only3601feature capture.
Five zero-step synthetic mapping/frame/validation checks pass3.76s. This is not
yet the saved native input/normalizer/actor oracle or an integrated controller.
Continue with packet2below, then reference/reset/controller/replay integration.
The active storage qualification stays frozen07ee40a/376tests; these five new
tests require a later bridge-specific Linux gate before motor trajectories.

# Next motor mission: use the pretrained policy in the shared world

User priority2026-10-27: explain the falling diagnostic and use available
pretrained locomotion behavior to reach a convincing muscle-driven recording.
This is a worker-ready next motor mission, not an executed training result.
Finish/retain the currently running storage gate and register the317remaining
sensor steps separately; neither requires another motor sweep. Do not alter an
executing checkout or increment a version solely for this plan.

## Evidence, objective and limits

The v11clip uses `boxing_evidence.diagnostic_command`, an untrained feedback
fixture around normalized action−0.2, not the released walking policy. It has
no learned balance objective and begins from the shared scene keyframe rather
than the upstream policy's reference-phase reset. The sigmoid maps−0.2 to about
0.0293 excitation; neither a low constant command nor the small feedback terms
constitute a trained postural controller. These code facts explain why that
clip cannot be a test of pretrained walking competence; they do not identify
all causes of instability in the learned controller.

We have actually executed `LocomotionFull.zip`, not merely found a candidate:
v6all40short native episodes completed, averaging3.52s. v8longer frozen-policy
trials had0/40twenty-second completions and physical tilt failures around5s.
v10small residual pilot failed its two motor-improvement gates. Preserve those
negatives; do not promise that the original weights already support guard,
running, a stationary stance or combat. No running checkpoint is qualified here.

Objective: connect the frozen pretrained700-muscle policy to each corner in one
shared MjData and produce a reproducible, clearly labeled two-person locomotion
recording. First establish input correctness and short reference-task transfer;
then measure the actual motor deficit before choosing adaptation. Non-goals:
calling walking boxing, replacing the anatomy, animation/kinematic assistance,
blindly grafting an unrelated humanoid policy, training from scratch, silently
retuning the existing teacher, or scaling failed episodes into a game count.

Dependencies: v6checkpoint/model/normalizer hashes and source-host inference;
v7mapped features; v10reference seam audit/reset exclusions; v11shared body,
command/contact/replay path; current source/model/recording/storage gates. The
teacher consumes3601features, while game observations have1970. The teacher has
reference targets, muscle forces/lengths/velocities and acceleration not present
in the tactical vector. Feeding the game vector into it is invalid.

## Read-first and ownership

Read AGENTS/RTK; README; MISSION/STATUS/IMPLEMENTATION; GAME; this plan;
V0.0.6_RESULTS, V0.0.7_RESULTS, V0.0.8_RESULTS, V0.0.10_D0_RESULTS,
V0.0.10_RESULTS; UPSTREAM_MOTOR_TRANSFER_PLAN;
V0.0.11_MISSION/COMPLETION_PACKET/D1_STORAGE_MISSION and current qualification
receipt. Then read `src/joxing/{upstream_motor,native_transfer,physical_motor,
motor_residual,reference_seam,env,replay,sensed_env,boxing_evidence}.py`, `bodies/native.py`, `scene.py`, v6/v10protocols,
`artifacts/v0.0.6/native/{loader-oracle,normalizer}.json`. Read original locomotion `_get_obs`/reset/step and decoder
source under third_party without editing it.

One integration owner in this chat. No agents or concurrent schema writers.
Third-party, old protocols/raw/normalizers/weights and executed checkouts are
read-only. Do not edit boxing_evidence.py or reuse its diagnostic controller ID
for the teacher. New src files below own the bridge; minimal named integration
changes to env/replay only after their exact contract is designed and tested.
No changes to model anatomy,700muscles, actuator parameters or sporting weights.
Generated models belong in artifacts/generated, large weights/raw in runs.

## Ordered implementation packets

1. **Input inventory and zero-step feature adapter.** Create
   `src/joxing/shared_motor.py` with `NativeMotorMap`,
   `MotorReferenceState`, `SharedTeacherInput` and
   `native_teacher_observation(data, mapping, reference, cached_dynamics)`.
   Build semantic corner-prefixed maps for85qpos/85qvel/700act/700actuators,
   six native keypoints and pelvis. Validate uniqueness and exact native order,
   including pelvis_tz,ty,tx and tilt/list/rotation. Map muscle force/length/
   velocity through actuator IDs, never contiguous assumptions. Output exact
   upstream3601block order/dtypes/clips/units. Preserve measured qacc/forces from
   the actual sample; mj_forward must not replace the cached policy input.
   Current/future self-reference targets are known motor goals, never future
   opponent data. Record world/reference translation and yaw transforms for
   both corners; no unchecked Euler-axis or universal-up assumptions.

2. **Oracle before rollout.** Add `tests/test_shared_motor.py` and
   `scripts/qualify_shared_motor_inputs.py`. Use saved native v6/v7states and
   explicitly copied cached dynamics for a zero-step extraction oracle; compare
   every block, normalized vector and scalar policy output. At a shared state,
   compare extraction against independently indexed semantic measurements.
   Validate A/B permutation, different spawn translations/yaw, force scaling,
   empty/invalid references, phase wrap and cache timing. Make the limits of
   this oracle explicit: different1ms/2ms dynamics cannot be claimed identical.
   Freeze actor963d2f56… and original normalizer/scheduled decoder0.075; hashes
   come from full v6receipts. `load_controller`/`predict` in native_transfer.py
   already provide frozen loading/inference and should be reused, not rewritten.

3. **Causal two-actor controller and reset variant.** Add
   `SharedFrozenMotorController` and `SharedMotorSnapshot` in the new module.
   At each20ms boundary freeze both actors' inputs before either scalar inference;
   use two separate reference/task states and immutable normalization. Queue each
   normalized700command through the existing begin_control/advance_substep path;
   apply sigmoid exactly once. Record controller/input/reference hashes. Add an
   explicit reference-based reset variant, initialized once before evaluation
   from registered upstream phase/velocity/activation/noise; compare reset
   projection/constraints with the native oracle. Generic v11keyframe reset is
   a different starting distribution. No pose writes, external forces, hand
   targets that move physics kinematically, state copying or gravity support
   after evaluated stepping begins. The new controller bundle changes the
   experiment contract; old diagnostic D1outcomes are not a teacher motor proof.

4. **Replay state.** Store full cached3601raw input, reference phase/index/
   targets, RNG, decoder/normalizer/controller identities and both held queues
   with the engine snapshot. Restore on scratch before atomic commit. Test an
   actual mid-command checkpoint: resume held steps with no actor call, then
   exact commands, caches, contacts and final state on the same source host.
   Changes to bundle, phase, sample time or normalization must fail closed.

5. **Small preregistered physical bridge, then footage.** Create
   `scripts/{register,qualify,audit,render}_shared_motor.py` and an explicit
   new-version mission/protocol only after the above tests pass. Proposed ceiling
   one shared world/one thread,60000aggregate physics,180s runner/300s post-gate,
   64MiB raw, firstlimit, and lower if the active version's remaining allowance
   is tighter. Count native-reference/oracle/replay/render steps too; no overlap
   with the live storage or sensor completion job. Source/protocol public and
   anonymously verified before outcomes; full pinned Linux tests first.
   Register four seeds71000..71003, fixed phase/start pairing, paired native
   source and two-corner teacher conditions of one original three-cycle horizon;
   first seed gets one actual mid-command continuation per shared condition.
   Freeze exact rows/physics after the reset inventory determines each horizon,
   and stop if their sum exceeds the cap. Start with separated, noninteracting
   athletes and reviewed collision masks; retain full scene physical contacts.
   Label this locomotion-transfer diagnostic; competition stays disabled.
   Require all4short shared trials to match the registered native task-completion
   criterion with no numerical faults and exact replay. Any fall remains in the
   denominator; do not swap seeds/clip start/end or terminate footage before it.
   Record first registered shared trial regardless of outcome, labeled with
   controller, simulated time, contact/points and termination/fault reason.

6. **Select adaptation from the observed failure.** If short transfer fails,
   inspect first-divergence feature/reference/frame/reset/contact evidence; fix
   demonstrated mapping defects before learning. If it passes, separately test
   extended locomotion and stationary guard with explicit new targets and the
   unchanged motor gates. A walking-reference policy is not automatically a
   standing policy. Preserve the pretrained full-body behavior as initialization
   or teacher, train a state-conditioned motor adaptation with small bounded
   residuals and imitation/pose/velocity regularization. Do not assume leg/arm
   muscles can be independently switched; balance is coupled. Train reach/return
   and guard with recovery, then admit tactical opponent-aware history/league
   after motor/contact/rule/replay qualification. Run a matched no-teacher
   baseline only for the small new adaptation question, never an unchanged giant
   walking sweep. Stronger transferred motors precede1000→100000valid games.

## Completion, stops and handoff

Validation commands after files exist:
`rtk proxy uv run pytest -q tests/test_shared_motor.py tests/test_upstream_motor.py`
and the full pinned source-host suite. Expected artifacts: complete input oracle,
all planned/attempted/valid/invalid/incomplete/not_run rows, block/actor/normalizer/
reference/model/source hashes, actual commands/caches/physics/contact arrays,
actual mid snapshot and exact replay, first-trial video with retained falls,
RSS/throughput/ownership/PID+start ticks/exit audit and a complete raw manifest.
A software/input-oracle pass alone is not a motor gate; a short walking pass is
not prolonged balance or boxing. Preserve failed qualification and source drift.

Stop first on mapping/cache/reference/bundle mismatch, assistance, numerical
fault, unknown process, resource limit, replay mismatch or unmet native criterion.
No silent weight/normalizer changes or enlarged training budget to make a plot
look successful. Update README/docs/experiment book with measured results and
limitations, retain raw and compact receipts, push public source/results, and
leave either a bounded healthy task or the exact next mission. No tag until the
registered micro-version's complete criteria are concluded honestly.
