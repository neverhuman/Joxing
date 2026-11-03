# H35: learn an upper-body guard while preserving the pretrained walking motor

Status: worker-ready prospective plan; NOTimplemented, registered or run.
Depends on H34's qualified actual-start and separated-walking controls. This is
the next real learning experiment. Single integration owner; no agents.

## Objective, assumptions and boundary

Train a shared two-player PPOpolicy to raise/hold both gloves near the head while
the released walking controller continues to actuate the complete native body.
Isolate arm skill from the unqualified stopping/close-contact task. H31's full-body
residual pilot failed; H33's400msreference warp failed; H34corrects initial overlap
and preserves BOTHwalkers on4/4three-second separated starts. Those failures
and all old criteria remain unchanged. This is a new task, not a larger retry.

All700muscles/body remain. Reuse H10's46active groups/122arm muscles, preserve
578protected teacher commands BIT EXACT and the frozen .075decoder/normalizer/
native walking phase/excitation. Arm torques can still disturb balance.
Non-goals: stationary stance, stop/turn controller, close-range match, legal
strike/score, competitive league, GPUphysics, jitter qualification, changed anatomy,
muscle speed, external forces, evaluated pose/velocity writes or upstream edits.

Completion separates (a) exact learner/physics execution, (b) target-error gain,
and (c) qualified moving guard. Policy hashes changing is not any skill gate.
Do not call control transitions, falls or motor episodes boxing games.

## Read first and ownership

Workspace:/Users/bentaylor/Documents/ChatGPT/JOXING. Read AGENTS/RTK; README,
GAME, MISSION, STATUS, IMPLEMENTATION and current experiment book/research index.
Then H10,H31–H35 and NATIVE_RESET_PLACEMENT_RESULTS.md; UpperResidualSpec/compose/
HeldUpperCommand in upper_residual.py, UpperPairTeacher in upper_residual_task.py;
native_reset_placement.py, native_pair_motor_rl.py/vec.py/pilot.py/train.py,
native_stance_probe.py, native_tape_storage.py and their source/receipt tests.

New write scopes: src/joxing/native_arm_guard.py, native_arm_guard_vec.py,
native_arm_guard_evidence.py, native_arm_guard_pilot.py; matching tests;
scripts/run_native_arm_guard.py, audit_native_arm_guard.py and a bounded qualifier;
experiments/v0.0.11/native-arm-guard-pilot-01.json. Reuse qualified components
without changing their defaults, closed protocols or referee/history schemas.
No third_party, sealed H31–H34roots, raw credentials, services/drivers/packages.
One writer for wrapper/episode/trace contracts. Raw tapes/weights stay outsideGit.

## Ordered implementation and hard interfaces

1.Build UpperResidualSpec from the ACTUALcanonical native model/source, checking
122selected/578protected/46groups/0mixed and all original bundle/source hashes.
Use compose/HeldUpperCommand, not the ordinary138decoder: adding zero to protected
channels can erase signed-zero bits. Preserve the original excitation exactly once.

2.Compose GuardMotorWorld around H34's PlacedNativePairMotorEnv in corrected LANES.
Install UpperPairTeacher(FrozenPairTeacher,spec) BEFOREconstructing the base env.
For each external paired46action: atomically stage both in the upper adapter,
call base.begin_control with two ZERO138vectors, capture adapter.prepared's two
HeldUpperCommands, then clear the one-shot adapter before the next decision.
Base's zero decoder passes composed700commands unchanged; never add another
residual or sigmoid. Store raw frozen teacher700 and composed700 separately.
The base decoder record's `teacher` is now POSTcomposition and must not be
misreported as the frozen actor output. Reject missing/stale/bad action pairs
before teacher inference or engine entry; quarantine inference/solver faults.

3.Use ONEfixed native training reset phase71000, declared openly. Two optimizer
seeds are independent runs; they are not two native motion phases. A training
world keeps one immutable H34scene and reinitializes native states at episode reset.
Evaluate71000/71016/71032/71048; label71000seen, the other THREEunseen. No new
phase chosen after results. Each seed worker reuses its own loaded frozen actor/
donor, ONEactive paired simulator at a time. Run the TWOindependent optimizer
seeds concurrently only after the registered serial/parallel startup check below.
Construct/close separate fixed-seed eval worlds inside each worker.

Keep logical episode ownership unique despite the fixed physical seed. A NEW
LogicalEpisodeNativeReferenceDriver may add a bound run/episode nonce to snapshot
metadata ONLY; inherit unchanged walking sample arithmetic/reference values.
Core's existing game-id digest then distinguishes episodes before histories/
referee are created. Bind nonce scheme in the wrapper contract; restore its
counter/nonce before reconstructing core. Do not mutate an existing game_id,
scene or contract after reset, or silently feed logical seeds into native phase.

4.GuardActorV1 exposes7901float32features: the existing four causal own/opponent
history frames, validity/sample/receipt ages, plus CURRENTfixed guard goal9.
Use current samples, not future opponent/reference data. Replace the old final
clocked reach/return goal with phase0guard: head+.12mforward±.15mlateral−.12mZ,
glove centers using the verified native offset. One-hot goal=(1,0,0) throughout.
Check world/ego goal consistency against scratch FK. Leave legacy actor_goal and
old observation/schema defaults unchanged; hash this new actor goal contract.
Opponent history is available, but this task does not prove learned opponent response.

5.Define a NEWmotor reward and1ms GuardMetrics; never reuse/report the base's
clocked Gaussian reward as the learner reward. Keep base support ending unchanged.
Prospective reward `native-arm-guard-linear-distance-v1`, each completed1ms:

    e = mean(right_glove_target_distance_m, left_glove_target_distance_m)
    h = clip((head_z - initial_head_z + .10) / .10, 0, 1)
    g = clip(1 - e / 1.5, 0, 1) * h
    f = clip(dot(right_glove_face_world, current_own_forward_world), 0, 1)
    r = .001*(.2 + .5*clip(up_world_z,0,1) + .3*g + .1*f
               - .01*mean(applied_action46**2)) - 1.0*first_support_failure

Linear distance supplies signal when hands are low; nonnegative alive shaping
avoids rewarding early termination. The head-height factor limits target gains
from bowing the head toward low hands. All units/frame definitions and values
must be frozen before outcomes. Sporting2/1points remain telemetry, never reward.
Keep old base diagnostic returns explicitly labelled separately if retained.

Measure every1ms. A guard sample requires BOTHdistances≤.15m, face cosine≥.5,
pelvis tilt≤20degrees, head_z≥initial_head_z−.10m. A successful hold is≥40continuous
samples, resetting on any violation; do not infer persistence from20ms endpoints.
No changing these gates to turn a negative pilot into a win.

6.NEW GuardPairVec preserves the qualified TWOlogical-slot/ONEworld semantics,
but action shape is(2,46). Both observations/actions precede either integration.
Any physical ending resets BOTHslots, retains genuine terminal observations,
failure/truncation masks and final engine state before reset. Cached/history/
logical-native seed ownership and last/pending actions belong in continuation.
Do not pad action densities/logprobs to138 or700 dimensions. Raw Gaussian46 and
clipped applied46 are separate; PPOevaluates the actual raw46latent policy.

7.Snapshot owns the base's exact engine/reference/decoder/history/referee/support
state PLUS held upper records/spec/raw teacher/input hashes, unique logical
episode ownership, new cached actor goals, guard reward/metrics and RNG/counters.
Upper adapter pending/prepared must be empty outside an action transaction.
Validate ALLwrapper fields before committing base restore; restore held commands
without inference. Wrong epoch even with identical fixed native state must fail.
Prove actual203+17held continuation ANDnextdecision, zero/signedzero parity to
H34walking and578protected commands for nonzero actions before training.

8.Reuse SB3PPO/PilotAuditedPPO and bounded full checkpoints, adapting shapes and
wrapper state explicitly. Preserve initial, every128joint-step rollout boundary,
final actor/critic/Adam/RNG/current obs/masks/Vec world; restore RNG LAST. No
candidate checkpoint selection. Real optimizer minibatch steps and both action
slots must be counted. Close/reconstruct evaluation contexts without altering a
live training world or losing its held command/clock. If lifecycle differs, qualify
that adapter with native continuation first, rather than bypassing its hashes.

## Evidence and source validation

Prefer a simple NEWfull-observation lossless trace over prematurely adapting the
138action causal codec. Chunk≤64joint rows into qualified ZIP_STORED archives,
each≤64MiB, preflight exact bytes and include temporary/fault reserves. Keep both
native before/after states, full7901actor inputs, raw/applied46actions, native3601
teacher inputs, raw/composed700commands, values/logprobs, rewards/masks, reset/
nonce/behavior-boundary metadata and ordered contact hashes. All source histories
and actual reset states remain reconstructible; no bounded-tail substitution.
The larger prospective2GiBraw allowance buys simple complete evidence, not more
physics. No raw arrays/weights/media in research/ and no large objects in Git.

All current163controls + relevant UpperResidualSpec/UpperPairTeacher controls
stay green. New cases cover signed-zero/protected channels, atomic46actions,
current guard goals/causality, head-bow/pose/continuous-hold anti-cheating cases,
reward units/no suicide incentive, both Vecslots/terminal masks, cross-reset nonce
rejection, full trace/checkpoint corruptions and held/next-step replay. Keep native
source fixtures bounded; never eagerly retain eight scenes again. Collect exact
node IDs and entered/static physics reserves before registration.

## Prospective finite learning study; reconcile before public registration

Two independent optimizer seeds761/983,2048JOINTsteps each =4096joint/8192actor
samples total,32rollout updates if all finish. CPU/thread1; shared policy64×64,
n_steps128,batch64,n_epochs4,lr3e-4,gamma.99,log_std_init−2, initial action means
EXACTLYzero. This is the named new guard task/exploration setting, not an unchanged
larger H31sweep. Stop if prerequisites fail; never silently reduce scope or raise caps.

Before/after: ALLfour fixed resets for BOTHseeds,3smaximum =16eval rows, same
untrained zero-mean baseline and final policy. First independently qualify zero
guard-action native parity/held continuation. Baseline must retain4/4paired3s
survival per seed; a failed baseline is a failed prerequisite, not a new phase search.

Primary motor gain: EACHseed/EACHcorner improves mean target error≥.05m over
the THREEunseen starts, with ALLthree completing3s and0invalid evaluations.
Separate full moving-guard qualification: additionally≥2/3unseen starts have the
40msqualified hold. Report seen71000separately; it cannot rescue unseen failures.
No pooled-seed or best-checkpoint win. Report both negative/partial outcomes.

Candidate caps: TWOaggregate simulation workers, one paired world/thread1each;
8GiBper worker and16GiBAGGREGATEincluding coordinator,1800swhole; setup60/
source180/native600/audit600/render4×30;300000TOTALformal physics including all
source/preparation/proofs/replay,2GiBaggregate raw/16MiBaggregate text, firstlimit.
This is a NEWtwo-seed parallel study; H34's completed8GiBsingle-owner result and
all earlier failed caps remain unchanged. Fresh host admission must budget both
workers before launch; the user allowance remains≤16aggregate owned workers.

Before concurrent learning, qualify TWOowned native processes with a fixed32joint
control prefix each against a single-process oracle, SAMEfixed commands/reset/
reference:32×20×3=1920extra entries. Require exact complete state/contact/command
parity, identity, healthy overlapping progress and full PID/start/exit evidence.
Compare numerical engine/contact/command bytes; validate ownership nonces and
contracts against each worker identity rather than requiring equal episode IDs.
This checks the new isolated-process path, not a throughput win or Linux100msgate.
If startup parity/capacity fails, stop before either optimizer starts; retain all
rows. Stop only the registered owned peer on a first global fault, retaining its
partial/not_run rows and charges. Do not silently switch the registered worker
configuration or launch another coordinator. The source/native-preflight reserve is12288, reconciled to exact cases.

Training81920physics;16evals≤48000; twoheld/nextdecision proofs554. Primary≤130474
and audit≤130474; combined+source/preflight12288=273236<300000. Each seed worker
has its own bounded ledger/namespace and≤134000total primary+auditentries; together
with source/preflight this stays≤280288. Register how failed/preparation prefixes
consume these credits; no work or source reserve is silently reset after a fault.

Keep writer scopes disjoint: output/seed-761 and output/seed-983, each≤896MiBraw/
6MiBtext. Coordinator/source/shared reports get≤256MiBraw/4MiBtext: totals2GiB/
16MiB. Checkpoint state remains in bounded binary continuations; do not duplicate
full world/observation arrays as JSON. Forecast real full traces/checkpoints/evals/
text/media/temporaries within each shard and global ceilings before registration.
If the partition needs adjustment, reconcile prospectively inside those aggregate
caps. No dropped cases, unregistered writer, post-failure cap increase or source
mutation while workers execute. One coordinator owns the global watch/manifest.

Use fresh dedicated ~/joxing on admitted xbabe4, exact qualified runtime/input
hashes, spare host capacity, no oldjob/duplicate. Frozen actor/normalizer/decoder/
reference hashes must remain unchanged. Do not impose or relabel failed Linux
100ms/cross-host parity as this Mac qualification. No services/driver/purchase work.

## Audit, completion, stop and handoff

Fresh independent reader reconstructs ALL4096training joint transitions, all16
evals, all1400commands/578protected-per-side bits, raw46policy values/logprobs,
metric/reward totals and actual midhold/nextdecision states. Retain every attempted,
valid/invalid/not_run row, entered counters, commands, hashes, PID/start/exit and
first resource/solver/identity faults. Any invalid blocks skill promotion.

Export ENTIREfirst UNSEENreset71016before/after for both optimizer seeds, four
fixed recordings including falls/0points. Use H34full-body camera, allframes;
hash/decode/inspect first/mid/last. Independently conclude execution, motor error
and guard criteria separately. Update README/docs/book/H35/index with allnegative
outcomes, push public small summaries/manifests/clips; full tapes stay outsideGit.
No fullv11tag/competitive promotion from a guard prerequisite alone.

On success, define stopping/turning/contact guard and legal-strike micro-games,
then frozen historical-opponent evaluation before claiming self-play progress.
Scale beyond the two independent seed workers to4→16paired collectors only after
exact serial parity, learning/evaluation validity and capacity
benchmarks; count valid boxing games separately from simulator/control throughput.
On failure, choose the measured control/reward/authority bottleneck for the next
small hypothesis. No unchanged huge sweep. Leave a bounded active task or an
exact next-version mission at every handoff.
