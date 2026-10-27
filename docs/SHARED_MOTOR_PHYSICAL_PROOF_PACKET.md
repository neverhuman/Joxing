# Next ready mission: record the pretrained policy in the shared world

The original-host archived-input oracle passed all four rows after422Linux
tests; both scalar corner predictions match the actual saved native command
with zero error. All475source/reference/weight checks and an independent raw/
normalization/action byte audit pass, no owned jobs remain. Read
SHARED_MOTOR_INPUT_RESULTS.md. This does not establish shared motor competence.

## Objective, dependencies and ownership

Objective: run the verified frozen700-muscle teacher for two intact athletes in
one world, retain every attempted outcome and record the first registered trial.
The current core is src/joxing/shared_motor_task.py: NativeReferenceDriver,
SharedMotorTask, initial reference reset, simultaneous input capture, normalized
commands,1ms staged CPU contacts,20ms holds, snapshots and atomic restore.
Eleven Mac development checks pass, including an actual23tick checkpoint and
17held substeps without a new decision followed by an exact engine/contact tail.
These use a deterministic fixture controller, NOT trained weights. Full fresh
Linux qualification of the new source remains required before formal motion.

Non-goals: calling walking boxing, competitive learning, changing the native
anatomy, applying evaluated pose/root writes or external support, using a static
standing reference as if already trained, or rerunning an unchanged walking
training sweep. No opponent-aware tactical policy/league is introduced here.
Both actors' own motor targets are known reference goals, not future opponent
state. All700muscles per corner, native gains/dynamics and sporting weights stay.

The explicit parallel-walk-v1 spawn variant in scene.build_scene places both
athletes at y±1.25m with yaw0 BEFORE attachment. Default boxing spawns stay the
default. The motor model retains the shared1ms implicitfast physics and glove/
shell equipment; native qualification used2ms Euler. This is an explicit transfer
condition, not a bit-exact native-dynamics claim. Initial reference pose/velocity/
activation is copied only during reset; every evaluated step is muscle driven.
The first development build changed an already attached frame and failed the
Bheading check before any evaluated step. All11setup errors/0physics and its
source hashes remain in shared-motor-scene-attempt-01.json; no gate was loosened.

One owner in this chat; no agents/concurrent writers. The externally maintained
grokbot/** and .github/workflows/ci.yml changes are preserved/read-only for this
packet; do not write into that concurrent work scope. CI now runs both tests and
grokbot/tests: include all executed sources in the next source manifest and
recount both suites before freezing the final gate. Third_party, completed
checkouts, old protocols/weights/normalizers/raw are read-only. Do not mutate
executing source or change unrelated jobs/services/drivers. New generated files
belong under artifacts/generated or runs; never commit weights/credentials.

## Ordered read-first files

AGENTS.md and /Users/bentaylor/.codex/RTK.md; README; MISSION/STATUS/IMPLEMENTATION;
GAME/SPEC_REVIEW; this packet; PRETRAINED_SHARED_MOTOR_BRIDGE;
V0.0.6_RESULTS, V0.0.8_RESULTS, V0.0.10_RESULTS,
SHARED_MOTOR_INPUT_RESULTS and V0.0.11_D1_COMPLETION_RESULTS. Then:

- src/joxing/{shared_motor_task,shared_motor,shared_motor_inputs,physics,scene}.py
- native_transfer.load_controller/predict, upstream_motor.NativeTask and its
  snapshot/reset, physical_motor signed-floor/evidence/audit helpers
- sensor_packed_evidence.py/evidence_storage.py/motor_storage.py,
  boxing_audit/sensor_physics_audit array methods and physical_motor replay
- bout_recording.py, scripts/render_bout_recording.py,
  scripts/validate_bout_render.py and tests/test_bout_recording.py
- tests/test_shared_motor_task.py and tests/test_shared_motor{,_inputs}.py
- experiments/v0.0.6/protocol.json and all frozen motor-input protocol/receipts.

## Small ordered implementation steps

1. Review core against its11tests. Add an independent shared-model semantic input
   audit: source native names→shared actor joint/actuator/activation/body IDs;
   actual shared qpos/qvel/act, force/length/velocity units and clips, keypoints
   transformed to the native spawn frame, exact self-reference fields. Capture
   real raw motor samples BEFORE any later mj_forward changes cached dynamics.
   Preserve actual qacc/force/keypoint caches; declare which checks reuse saved
   measurements versus independent FK or physical derivation. Do not replace
   delayed/actual samples with recomputed current truth. Add corruption tests
   for A/B swap, wrong phase/yaw/order, altered input/reference/cache and controller.
   Current snapshot validates reference/self-target/cache owner/time, commands,
   engine/control clocks and exact continuation; it does not independently
   reconstruct every saved cached dynamic value. Close/document that gap before
   claiming engine-bound motor-input qualification.

2. Create src/joxing/shared_motor_evidence.py with MotorTape.collect_decision,
   capture_substep and save. Keep unchanged typed fields and full signed contacts,
   applied700commands/excitations/corner, engine hashes, actual3601raw inputs,
   normalized inputs, reference/current/future targets, tracking/posture/support
   and actual time/counters. Reuse qualified gzipJSON/ZIP_LZMA writers with the
   same pre-write credits and temporary/fault retention. Save initial, actual
   mid203 and final complete engine/reference/cache/control snapshots. Save the
   first fixed source's visual states at tick0/every40ticks/final for25fps video;
   rendering must use scratch data and never add an evaluated physical step.

3. Create scripts/{register,qualify,audit,supervise,render}_shared_motor.py and
   tests/test_shared_motor_proof.py. Registrar writes a NEW protocol once, binds
   all source/upstream/model/reference/actor/normalizer/decoder/bundle/file hashes,
   source-host identity, exact conditions and immutable native dependencies.
   Use FrozenPairTeacher.from_inputs with the already verified original assets;
   never substitute random weights or retune decoder0.075/normalizer. Construct
   fresh native reset donors; assert their physics_total stays0. Query reference
   timelines only; donors must never advance shared motion or provide assistance.
   Both actors use the same fixed seed per shared trial, so eight actor outcomes
   across four worlds are correlated, not eight independent replications.

4. Preregister four shared sources seeds71000..71003,3520ticks each, plus the
   first source's actual203→3520replay3317ticks. Planned new physics17397. Bind
   their existing v6native trials as fixed dependencies; do not rerun those
   successful native conditions merely for a side-by-side plot. Stop on a changed
   actual native horizon/reference prerequisite rather than silently changing
   rows. No optimization. Source/model/reset/motor state is a named variant.

5. Hard proposed proof limits:60000aggregate physics/180s runner/300spostgate/
   32MiBraw/one simulation process and one numerical thread, firstlimit; lower
   if actual remaining whole-v11credit is tighter. Global<=16owned simulation
   workers, no overlap with another proof. Source regressions are separately
   declared fixtures; the current total is433tests, recount after adding proof
   tests. Fresh host load/PID scan and8GiBRAM/10GiBdisk floors twiceatstart.
   At this handoff whole-v11formalphysics5390, raw202,302,252bytes including old
   originals/qualified copies/motor-input proof, leaving66,133,204bytes under
   256MiB. The original remaining v11seed/asymmetric/adverse battery still needs
   its own measured budget;32MiB is a ceiling, not a commitment to consume it.
   Keep original whole120kphysics/900spostgate limits and all recorded failures.

6. Allocate all five attempted rows and resources BEFORE model load; charge each
   attempt before engine entry, persist progress at most20ticks apart, record
   PID/startticks/ownership, caps/memory/faults/last-valid state and every later
   not_run slot. Frozen public source/protocol+anonymous byte verification and
   final pinned Linux full suite are prerequisites. Supervisor runs proof then
   a separate saved-evidence audit; stop first on mismatch/fault/drift/cap/unknown
   job, retain all partial data. No hidden replay/audit physics: count any actual
   scratch stepping explicitly within the registered total before it occurs.

7. Auditor reads all source/tail rows and recomputes application clocks, owner/
   action transform, every signed contact/approach, force/support/posture evidence,
   reference timing, native tracking health and exact mid-held replay from saved
   data. Preserve both pre-reference-update tracking termination and post-update
   telemetry, since the native task computes them at different points. Use the
   existing physical failure thresholds/first-latch persistence; do not call a
   recording successful merely because a file writes. Require all4worlds/both
   actors to reach the registered short native horizon without numerical faults,
   tracking failure or persistent physical latch, and require exact first-source
   replay. Any failure remains failed; short success is not stationary guard,
   20second balance, boxing or replicated learning improvement.

8. Render ONLYthe first registered source, including its fall/fault/last-valid
   final frame if it fails. Label frozen pretrained locomotion, simulated time,
   actual controls/reference/physical failure and diagnostic status. No seed/clip
   selection to hide collapse. Decode and verify every frame with the bounded
   repaired reader; optional display-only repair must be separately frozen and
   retain its failure. Full raw stays ignored/released appropriately, curated
   result/negative rows/commands/manifests/video and exact reproduction stay in Git.

## Validation and final handoff

Before source-host launch, run targeted tests for the new proof plus existing
shared_motor_task/shared_motor_inputs/shared_motor and scene/boxing_env checks,
then the complete final source-host suite. Current core11tests and input25tests
are development fixtures; exact trained archived command equality is already
qualified independently. Tests must cover actual mid-hold without actor call,
reference/cache restore atomically before commit, complete signed contact tail,
normalizer immutability, all rows/faults, quota peak, counter/ownership/source
mutation and video handling. Run Ruff and git diff --check. No regressions or
new policy/environment arithmetic can be waved through by the old422test receipt.

Expected artifacts: runs/v0.0.11/shared-motor-proof-01/{protocol,result,progress,
events,raw-manifest,independent-audit}; four source folders and replay71000,
first-source recording/clip; supervision and independent exit/input/raw audit.
Exact command/source/config/model/actor/normalizer/reference/decoder/seed hashes,
attempted/valid/invalid/incomplete/not_run denominators, actual physics/time/
RSS/byte/ownership receipts and next action are mandatory. Push public source
before outcomes and honest results afterward. Tag only a completed whole version;
v11still has the original remaining correctness battery. On failure define the
next small measured reset/feature/contact/reference/motor issue, never enlarge
an unchanged failed walking sweep. After successful locomotion transfer, adapt
stationary balance/guard/reach/recovery from the frozen teacher before tactics,
causal opponent history and frozen-opponent league training.
