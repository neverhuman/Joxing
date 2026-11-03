# H33 implementation packet: stop the pretrained walker before more boxing RL

Status: CONCLUDED. Execution/replay qualified; all stance criteria FAILED.
[Results](NATIVE_STANCE_REFERENCE_RESULTS.md). Current next mission:
[NATIVE_RESET_PLACEMENT_MISSION.md](NATIVE_RESET_PLACEMENT_MISSION.md)/H34.
The original implementation/preregistered intent below is retained for history.

## Objective and reason

Produce a candidate unassisted stationary motor prior that can later support
learned guard/striking. H31actually trained both native agents, but all16evals
fell before3s, scored0–0 and completed no guard cycle. H32independently replayed
all data. Fixed recordings show forward approach/body contact with low hands.
This suggests missing stopping/interaction control, but does NOT prove contact
causes every fall. Separate transfer, interaction and reference-goal effects.

The released actor already consumes its OWN future motion reference. Commanding
a different reference could reuse that actor without learning balance afresh.
H10warns that freezing a gait clock changes the learned task: it is a NEWtestable
motor-reference variant, never an automatically validated standing controller.
The pinned Oct29asset inventory found locomotion/legs/manipulation, no compatible
standing checkpoint. Do not substitute a different embodiment or repeat inventory
without new evidence.

Non-goals: no new PPOtraining/100k-game sweep, competitive promotion, points/
fall-rule changes, larger residual amplitude, altered muscles/vmax/mass, gravity
compensation, external forces, evaluated pose/velocity writes or upstream edits.
All700muscles/body, native axes, excitation, original teacher/normalizer/decoder
and physical/rule parameters remain. Actual reset poses/velocities stay identical
to the corresponding native walking start; only named spawn/reference goals vary.

## Ordered read-first and ownership

1.AGENTS.md and /Users/bentaylor/.codex/RTK.md; README, docs/GAME.md, MISSION,
STATUS, IMPLEMENTATION and experiments/v0.0.11/README.md.
2.research/README.md,H1,H2,H10,H31,H32,H33; NATIVE_PAIR_TWO_SEED_RESULTS.md,
NATIVE_PAIR_RL_PILOT_MISSION.md, UPPER_RESIDUAL_RESULTS.md and native stance risks.
3.src/joxing/shared_motor_task.py NativeReferenceDriver/sample/snapshot and
SharedMotorTask.restore; shared_motor.py/input projection; native_pair_motor_rl.py
NativePairMotorCore/PairMotorConfig/from_inputs/reset/snapshot/restore;
scene.py build_scene; support monitors, native_pair_motor_pilot.py evaluator and
H32replay/export scripts. Read all matching tests before changing an interface.
4.Closed protocol03and H32protocol/results/raw manifests. Actual inputs/runtime
are onxbabe4under~/joxing/native-pair-pilot-xbabe4-20261103-01; they are read-only.
Never modify either closed native or post-task root, source, weights or receipts.

Likely write scope: NEWsrc/joxing/native_stance_reference.py and
native_stance_probe.py; tests/test_native_stance_reference.py and
tests/test_native_stance_probe.py; new probe entry/auditor/protocol, docs/results/H33notes.
Additive edits ONLY as needed in scene.py and native_pair_motor_rl.py for explicit
placement/reference variants; existing defaults must remain bit-exact. One owner
for these constructors; no concurrent writers. No referee/schema, old protocols,
third_party, old worker backends, services/drivers/packages or other projects.
Generated XML goes under artifacts/generated or the new run, never third_party.

## Implementation order

1.Inventory the ACTUAL load_controller behavior/amplitude restoration and input
order from source; retain the H31actor/normalizer/reference/group hashes. No
new checkpoints. Keep default NativeReferenceDriver.sample arithmetic untouched.
2.Add named `opposed-separated-lanes-v1` scene/spawn variant: original x±.8m and
A0/Bπyaws, y−1.25/+1.25m. Preserve every collider/mask/material/inertia/equality,
actuator and solver setting. This changes spawn only, separating interactor
contact while retaining opposed rotations. It is a diagnostic, not boxing.
   Extend explicit placement dispatch/core map contracts in opt-in code. Do not
   initialize a new variant and silently label it `opposed`. Test native170/170/
   1400dimensions, all muscle parameters and physical masks, and rigid translated
   baseline state/command mappings. If reference mapping fails, stop physics.
3.Add NEW`walk-to-hold400ms-v1` reference driver, same frozen actor. Smooth phase
clock in seconds: f(t)=t−t²/(2T) for0≤t≤T, f(t)=T/2 thereafter, T=.4s.
   For a physical control at t, query the SAMEnative trajectory at
   init_time+f(t+.02+j*.02), j=0..5, retaining original current/future layout and
   own keypoints. Verify reset/current target conventions from native source;
   do not assume a target row is a live pose. The real engine clock advances
   normally1ms; this changes desired motion goals, not time integration.
   Initial source reference must be consistent with this profile; actual native
   qpos/qvel/activation reset values remain unchanged. No actual velocity zeroing.
4.Bind mode/T/phase clock/trajectory/native-axis identity into every contract,
input/reference hash and snapshot. Restore with actual tick/time and exact
reference reconstruction; no extra teacher decision inside a20ms hold.
   An adapter may compose around the loaded donor/teacher. Do not mutate an
   existing env.contract after evaluation starts. Source walking mode must retain
   zero/signedzero command/engine/contact/replay parity bit-for-bit.
5.Unit/source tests: phase monotonicity/bounds/future order/units/endpoints,
unknown variants, corrupt clock/mode/T/snapshot/reference/hash rejection, both
sides/native joint conventions, atomic pair staging, all1400muscles unchanged,
reset-only pose writes, baseline source parity and203+17held continuation for
new variants. Existing99controls stay green. Collect exact test IDs/fixture
physics; a source fixture uses a synthetic teacher and is not the physical result.

## Prospective finite physical test to freeze BEFORE outcomes

Use the qualified xbabe4profile, a NEW~/joxing root, one persistent paired world/
thread1 and fresh same-limit admission (32CPUs:load≤16,≥16GiBfree,≥10GiBdisk).
No overlapping writer. Freeze protocol/code publicly before any physical probe.

Four reset seeds71000/71016/71032/71048; twelve NEW3s paired trials:
- separated-lanes + original walking reference;
- separated-lanes + walk-to-hold400ms reference;
- original opposed placement + walk-to-hold400ms reference.
Use zero learned residual for both actors, actual pretrained700-channel outputs.
Reuse the FOURexisting audited opposed/walking baseline evals as prior evidence,
not new valid attempts or a repeated training sweep. Independent replay of those
four for contact-onset analysis is allowed ONLY if included in the frozen budget.

Measure every1ms: support loads/tilt/drop, failure onset, interactor first actual
force-producing contact by surface, root XYdrift/speed, glove/target distance,
heading and unchanged candidate points. Record first contact relative to failure;
correlation alone is not causal proof. Preserve all starts, numerical faults,
not_run rows and ends. No omission of poor starts or choosing a favorable phase.

Frozen task caps:1200swhole/setup60/source120/
physical360/audit300/render3×30; first limit;8GiB/12process/1worker/thread1,
100000TOTALphysics including source reserve/prep/holds/replays and baseline
contact replay,384MiBraw/16MiBtext.12×3000primary+12×3000independent replay+
6×237primary/audit holds+4096source reserve=77518≤100000. The optional four old baseline
contact replays are NOT part of this packet; prior audited motor results/video
remain prior evidence. Do not infer a causal contact effect without those traces.
Forecast exact native evidence/media/text/temporary/fault bytes before launch;
reuse H31/H32bounded writers, do not accumulate unbounded overlapping tapes.
Export the ENTIREfirst reset for each NEWcondition with H32compact renderer,
including falls/0points; label diagnostic condition and desired-reference mode.

## Interpretation, completion and stop conditions

Candidate **stance prior only**: both actors on≥3/4starts complete3s under original
support/fall rules AND XYdrift≤.30m AND mean horizontal speed in the final1s≤.15m/s.
Freeze units/body frame/aggregation precisely in source/protocol before outcomes.
This is separate from H31's FAILEDunchanged guard/boxing criteria; it cannot
rescue that result or establish guard, a legal punch, force calibration or a bout.

If separated walking survives while opposed walking fails after interactor force
onset, prioritize stopping/guard/footwork under interaction. If both walking
conditions fail, revisit transfer/reference/initial-state compatibility. If the
new reference fails, retain the negative result; a gait-clock warp is not a proven
standing controller. Do not tune T/thresholds or choose checkpoints after results.
Only a qualified stance candidate can motivate a separately registered upper-arm
reference/residual guard curriculum, then legal strikes and historical-opponent
self-play. No unchanged huge PPOsweep to force a version increment.

Stop on source/parity/assistance/identity/numerical/ownership or first resource
fault; preserve prefixes and charge every entered step. A physically valid fall
is a failed motor outcome, not an invalid solver run or sporting win. Audit every
attempt, all700commands per side, exact continuation, old default parity, raw
manifests, all owned PIDs/start/exit, and source/runtime/model/config hashes.
Publish small receipts, all negative outcomes and fixed clips in Git; weights/
tapes stay outside. Update README/MISSION/STATUS/IMPLEMENTATION/book/H33/index;
leave the next measured question. No tag until its full micro-version scope ends.


## Implemented source and exact placement recipe (2026-11-03)

The driver preserves the upstream reset convention: at tick0 current target is
init_time+f(0), future rows init_time+f(j*.02), j=1..5. At later controls use the
source next-control .02s lead, then warp all six own-reference times. Actual
walking reset qpos/qvel/activation remain byte-identical. Driver snapshots bind
mode, T, formula, reset/later conventions, trajectory content and native axes.
Default walking arithmetic and old default environment contracts are unchanged.

The first direct translated MjSpec changed eleven compiler-derived arrays by
rounding; an attempted128epsilon comparison also failed. Both failures remain.
The adopted named scene compiles the ORIGINALopposed frames, then translates
ONLY the two fixed root body frames before creating live MjData. It keeps every
other compiled numeric array BIT EXACT, including all rest-length/inertia caches.
The spec records matching fixed offsets; scene hash binds XML PLUS this recipe.
Reconstruct with build_scene(spawn_variant="opposed-separated-lanes-v1"); loading
only XML would recompute derived constants and does not reproduce this profile.
No translation occurs after scene construction or during evaluated motion.

StanceMetrics measures every completed1ms. Root means pelvis body origin, in
world meters; drift is MAXIMUM XYexcursion since reset, not final displacement.
Speed is the norm of its world-XYpoint velocity (mj_jac times completed qvel);
last full second is all1000samples t=2.001..3.000s. Each corner must independently
pass survival+drift+speed on≥3/4fixedstarts. Pair episode stops at the original
first physical failure/control boundary. Missing/invalid/short trials cannot pass.
All staged contacts are hashed every1ms; first positive-normal-force interactor
surface pair/tick is retained. This is simulated force, not calibrated impact.

Local development:30cases/2fail;37/1fail;37/0fail; full136/0fail (41.764s).
All old99controls remain; three new actual203+17held/nextdecision replays pass.
6,367static-counted synthetic-teacher fixture physics across all four attempts;
additional compile/forecast inspections0physics. This is no pretrained outcome.
[All faults/commands](../artifacts/v0.0.11/development/h33-development-01.json).

Task source fixture reserve4096 covers2674static successful loop steps. Primary
max36711 (12×3000+3×237), independent replay max36711; total reserve77518.
Exact archive forecast117344136B plus explicit source/text/media/temp/fault
reserves=268339080B<384MiB. No weights/checkpoints enter the output or Git.
Use scripts/qualify_native_stance_probe.py with the committed protocol, which
requires ALL136fresh target cases before native proof/trials. Actual teacher,
normalizer, reference and runtime hashes remain H31's; no new optimization.
