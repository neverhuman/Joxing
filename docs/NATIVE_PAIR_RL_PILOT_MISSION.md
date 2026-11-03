Completed checkpoint2026-11-02: the small actual native PPO/replay/recording
witness passes on its explicit Macprofile.32joint transitions/64actor samples/
4updates, all32independent engine replays exact, points0–0. No skill or competitive
qualification. Read NATIVE_PAIR_MAC_PROTOTYPE_RESULTS.md; CURRENTnext is
NATIVE_PAIR_MOTOR_TRAINING_CONTINUATION.md. Larger H31pilot remains pending.
Original prospective plan below retained.

Implementation update2026-11-02: native bridge/paired PPOVecEnv and small witness
implemented;22Mac development checks pass. Initial prototype uses EXPLICITlocal
Mac runtime/profile and fresh dedicated checkout; protocol native-pair-mac-
prototype-protocol.json freezes32jointsteps/64actor samples/4updates/native replay/
fixed clips. Source/protocol public before formal outcomes. This is a small
prequalification witness, not the larger two-seed Linuxpilot or competitive
boxing qualification. See current STATUS.md for exact caps/hashes/outcomes.

# H31 — Native two-humanoid motor RL is the critical path

Status: prospective worker-ready implementation mission, NOTimplemented/run.
User steering2026-11-02: get actual two-humanoid RL running ASAP. H30fake-worker
benchmark is secondary; it does not gate writing or qualifying this native path.
No more infrastructure-only missions unless an actual measured fault blocks this
pilot. Keep H30drafts/faults, no formal H30rows have run. No agent delegation.

## Objective and honest boundary

Deliver an executable small RL pilot with TWO intact700muscle humans in ONE
physical world, the compatible pretrained walking motor as baseline, causal own/
opponent motion history, learned muscle residuals and fixed before/after clips.
First prove real native transitions and optimizer changes; measure balance and
reach/return independently. Then use legal contacts/points and frozen opponents
for competitive boxing once existing motor/contact/rules/replay gates pass.

This first pilot is **motor learning toward boxing**, not a qualified boxing
match or100k-game result. Dense balance/reach learning may precede stable motor
competence: that competence is its objective. A physically valid falling baseline
is a measured training problem, not a reason to postpone motor RL for another
fake benchmark. Numerical, control/replay/ownership faults still stop execution.
Competition remains false until sporting attribution and replay gates pass.
Do not relax face/impulse/point predicates, delete falls or rename motion clips
as bouts. No kinematic pose writes, root support, gravity compensation, pruning,
modified muscles, invented scoring, package/service/driver changes or purchases.

Existing evidence: frozen native walking supports one short3.52s shared parallel
source; it is not stationary guard/opposed boxing. Latest actual close clip is
0–0/severe lean. H10arm-mask/source wrapper and H29source readiness qualify
interfaces, not learned behavior. H27native64transitions/12updates prefix is
invalid on shutdown100.58ms; retain it. Single persistent shared-world training
avoids worker creation/destruction per episode; lifetime still audited at close.
Do not inherit an unqualified native or broad regression receipt for changed code.

## Read first and ownership

Read in order: AGENTS.md,/Users/bentaylor/.codex/RTK.md,README.md,docs/GAME.md,
MISSION.md,STATUS.md,IMPLEMENTATION.md,research/README.md,H1/H2/H4/H8/H10/H27/
H29/H30/H31; SELF_PLAY_EXECUTION_PLAN.md,UPPER_RESIDUAL_MISSION.md,
NATIVE_BALANCE_STRIKE_NEXT_PACKET.md,SHARED_MOTOR_PHYSICAL_RESULTS.md,
UPPER_RESIDUAL_RESULTS.md,PORTABLE_LINUX_SOURCE_READINESS_RESULTS.md,
STATE_HISTORY_SELF_PLAY_AUDIT.md,TRAINING.md and the current v11experiment book.

Then src/joxing/{env,sensed_env,observations,history,timing,sensing,shared_motor,
shared_motor_task,shared_motor_inputs,upper_residual,upper_residual_task,
motor_residual,native_transfer,reference_seam,shared_motor_evidence,
shared_motor_saved_audit,native_tape_storage,physical_motor,wave}.py;
existing matching tests and v10/v11native source/checkpoint protocols.

Single owner, new opt-in files initially: src/joxing/native_pair_motor_rl.py,
native_pair_motor_vec.py,native_pair_motor_train.py,native_pair_motor_evidence.py;
tests/test_native_pair_motor_rl.py,test_native_pair_motor_vec.py,
test_native_pair_motor_replay.py; new bounded entry/auditor/protocol/results.
Do not edit third_party, frozen checkpoints, prior protocols/results/roots,
scene/referee schemas, old default collectors/decoders or other projects.
An explicit new scene/model variant may be GENERATED under artifacts/generated;
retain all1400actuators/activations. One owner for any integration changes; no
parallel writers/agents. Reference-body reset writes are initialization only;
there may be no pose/velocity assistance during an evaluated interval.

## Concrete ordered implementation

1. Inventory executable baseline: BoxingEnv/SensedBoxingEnv already have native
   bodies, causal observations/history, simultaneous commands and point telemetry.
   SharedMotorTask/FrozenPairTeacher provide native walking inputs/phase/decoder.
   UpperResidualTask currently accepts ONLYDiagnosticArmPolicy; that is not a
   learned policy. Do not pretend it already exposes a generic RLaction API.
   Record exact actor/normalizer/reference/group/model/source hashes and names.
2. Implement NativePairMotorEnv as an explicit motor-learning wrapper. Keep the
   original BoxingEnv/SensedBoxingEnv and SharedMotorTask defaults unchanged.
   Choose ONEphysics owner per world; never step both loops over the same data.
   A NativeWalkingPairDecoder builds native teacher3601features from the same
   completed physical state using qualified NativeMotorMap and reference phase,
   two original scalar teacher calls, and expands the learner's138group actions
   per actor into all700commands via existing .1residual transform/sigmoid once.
   The pretrained actor/normalizer/reference stay frozen. Do not mix teacher
   observations with actor/opponent observations or freeze gait to imply standing.
3. Reset first in the teacher's actual valid moving task. Qualify zero-residual
   wrapper against unchanged shared teacher on identical unassisted state. Then
   use a separately named mutually facing/striking-range scene/reset profile;
   derive actual native yaw/root/keypoint transforms and reachable glove-to-target
   gap, not guessed pelvis spacing or an unqualified180degree checkpoint port.
   If opposed transfer fails numerically/observationally, isolate THATnative
   adapter in a small bounded witness; do not silently swap back and call walking
   a bout. Retain both conditions and their footage.
4. Actor observation uses observations.observe()/TimedHistory: own joint position/
   velocity, all700activation states/previous commands and visible own/opponent
   body poses, linear/angular velocities; four recent samples plus validity,
   sample/receipt ages, current public goal/phase/clock. Keep independent histories
   and state per A/B; no opponent activations, hidden reference or future solver
   state leaks into actor features. Both samples from the SAMEcompleted state
   before either policy or engine mutation. Start with zero sensor/action delay
   in a named timing profile; preserve queues/age fields and test fixed delay/
   jitter source invariants before claiming realistic reaction robustness.
5. Expose two logical PPOslots backed by ONEworld: NativePairMotorVecEnv.step_async
   validates/stores BOTH138actions; step_wait samples/commits both700commands
   simultaneously then advances exactly20one-ms intervals. No extra teacher/
   policy inference inside a hold. Reset both logical slots together on motor
   failure/time limit; final observations must be genuine pre-reset states for
   BOTHslots. Maintain terminated/truncated distinction and logical masks.
   One physical episode is ONEmini-game, not two independent games. PPOmay learn
   from both slots with one shared frozen behavior bundle per rollout; preserve
   paired-world dependence in evaluation and do not count slots as independent
   statistical trials. Historical-opponent actor data later stays out of the
   designated learner's policy-gradient loss.
6. Use explicit motor objective native700-paired-balance-reach-return-v1, separate
   from sporting utility. Define reset goals from real reachable glove/torso
   geometry; scheduled guard/reach/return TARGETS, never scripted joint motion.
   Register exact distance/face/up/support/effort/physical-fall definitions and
   reward coefficients before outcomes. Include goal/phase in actor observations.
   Do not repeat unchanged v10alive-only objective or train on points until legal
   contact/attribution gates pass. Log head2/torso1/guard0candidate points without
   altering weights; unqualified events cannot be learning wins or proof ofboxing.
7. Native PPOpilot: zero initial residual mean and small registered exploration,
   shared actor with separate A/Bhistory, updates only after a complete paired
   rollout; correct action transform/logprob and termination/bootstrapping masks.
   Keep optimizer/normalizer/RNG/checkpoint/held commands and histories exactly
   resumable. Use existing SB3pinned runtime where possible; do not introduce a
   second handwritten RLalgorithm just for this pilot. Prove optimizer tensor
   changes, all1400applied muscle excitations and reconstructed source transitions
   came from this native world. A surrogate environment is never native proof.
8. Tests: neutral zero-residual bit-equality, actor isolation/side symmetry,
   per-body limb velocity response, history causality/age and future sample
   rejection, independent/reset histories, atomic paired action rejection,
   all700muscles/138groups and single excitation, unchanged frozen teacher,
  20heldsubsteps/noextra decisions, actual midhold203restore with17steps left,
   native commands/integration/contact/reference/RNG continuation; paired VecEnv
   reset/final-observation/masks/episode counts; PPOsample/update positive control
   on deterministic fixture separately labelled, no reward from invalid contact.
   Run new focused tests and a fresh complete Linux regression using bounded
   fresh-module processes if needed. Keep exact node IDs, known strict xfail,
   every failure and fixture physics separate from formal native counters.
9. Register a small finite physical pilot AFTERimplementation/source checks:
   one persistent native world/thread1,<=8GiBowned RSS, no more than16aggregate
   owned workers (this pilot uses ONE). Proposed whole1800s/source<=900/training
   <=600/render<=120/audit<=120, first limit;200000formalphysics intervals incl
   preparation/source/replay/evaluation/render physics if any;1GiBraw,16MiBsmall
   receipts, free RAM>=16GiB/disk>=10GiB. Final numerical budgets and exact phase
   allocation must be internally consistent/frozen before outcome. Use actual
   fresh4host capacity/admission/new ~/joxing root/readonly runtime; no other jobs.
   No per-episode process churn; count physics BEFOREevery actual mj_step.
10. Stages: (a) intact native model and actor provenance, zero-residual command/
    engine equivalence and midhold replay on two seeds; (b) bounded native PPO
    for two seeds, proposed2048JOINTcontrol transitions/seed (8192total actor
    samples,81920physics before prep/evals); (c) fixed before/after paired-seed
    evals and full fixed clips including falls/0points. Freeze exact seeds and
    goal/eval splits before run. A valid falling baseline may proceed to motor
    learning; parity/assistance/NaN/alias/lifetime/resource faults stop first.
    No automatic large sweep. If time/physics exhausted, preserve partial actors/
    counts/invalid/not_run, never manufacture a completed result.
11. Frozen eval: compare balance duration/root lean/foot support/fall, reachable
    glove-target distance, face orientation/reach-return success, command effort,
    collision/contact/point predicates against the unchanged pretrained baseline
    and an unchanged opponent panel. Separate true replay validity, optimizer
    progress and performance improvement; seed failures remain in denominators.
    Publish first complete before/after clip from preregistered eval ordering,
    not the prettiest chosen bout. Store full compressed outcome tables/H31notes
    in Git, checkpoints/raw/videos through existing bounded artifact path only.
12. If motor gate improves and stable reach/contact/replay/rules qualify, freeze
    complete actor/decoder/normalizer/timing/history bundles, start historical-
    opponent sparring and fixed panel/swapped-side evaluation. That later sport
    stage uses canonical terminal match utility; it is not an alias for this
    dense motor task. Scale measured valid games/s1→2→4→8→16 only after single-
    world proof; target1000→10000→100000valid games. GPUphysics still requires
    exact-model feature/parity benchmarks. Do not let optional H30block native
    work unless an actual native RSS/retirement fault demonstrates the dependency.

## Validation, artifacts, completion and handoff

Use RTK-prefixed .venv/bin/python with PYTHONPATH=src:.:scripts, numerical thread
variables=1, pytest --junitxml under new runs and Ruff on new modules/tests.
Keep development fixtures distinct from actual checkpoint/native evaluation.
Expected: fresh source receipt, exact schemas/commands/versions/source/model/
actor/normalizer/reference/group/rule hashes, PID/start ownership and progress,
counted entered/completed physics/decisions/actor samples/updates/episodes,
complete valid/invalid/not_run rows and every fault, learner before/after hashes,
actual replay/contact tapes and fixed clips, held-out performance summary and
independent saved-array/transition audit. No credential/large objects in Git.

First completion milestone: real native two-humanoid rollout→PPOupdate→frozen
policy replay/recording, honestly measured improvement or failure. No competitive
promotion/tag from source checks, optimizer changes or a falling video. Fix only
measured native bottlenecks, keep prototype vs qualified status explicit, update
README/MISSION/STATUS/IMPLEMENTATION/book/H31/index and push public Git. Leave an
active bounded task or this clear ready mission at every handoff. H30drafts stay
secondary/dispatch-disabled until independently completed; next unused ID:H32.
