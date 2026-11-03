# H34: phase-independent, non-overlapping native boxing starts

Status: CONCLUDED.02qualifies initialization/walking control;01sourceRSSinvalid
retained. [Results](NATIVE_RESET_PLACEMENT_RESULTS.md). Current next mission:
[NATIVE_ARM_GUARD_RL_MISSION.md](NATIVE_ARM_GUARD_RL_MISSION.md)/H35.
Original packet/failed attempt/repair intent below retained for history.

## Objective, evidence and completion boundary

Make the ACTUAL initial pelvis positions and interactor clearance independent of
the pretrained trajectory's global root displacement. Preserve its native pose,
velocity, activation, references, actor and all700muscles/body. This establishes
usable game initialization before further opposed motor training.

H33 passes136target controls and all12native replays. Both walkers survive3s on
4/4separated starts, while the400msstopping reference fails every stance gate.
The post-hoc reset audit finds actual opposed pelvis XYgaps1.148465,0.806951,
0.280446,0.084457m as native reset phase advances. Seed71048already has interactor
penetration up to0.119165m and force at the FIRSTphysics interval. Fixed attachment
frames±0.8m are not fixed actual pelvis positions when native qpos has root travel.

Completion means correct, reproducible starts plus an unchanged separated-walking
control. It does NOT establish stopping, guard, a punch, scoring, a bout or RL gain.
All prior diagnostic trials remain in their original denominators; do not relabel
H33's valid physical negatives or rescue its failed stance criterion.

Non-goals: no new PPOsweep, reference-clock tuning, competitive promotion, changed
rules/weights, pruned anatomy, modified muscles/vmax/mass/gravity/solver, active
root steering, external assistance or evaluated pose/velocity writes. Do not
repeat checkpoint inventory: H10found no compatible released standing checkpoint.

## Read first and ownership

1.AGENTS.md,/Users/bentaylor/.codex/RTK.md; README, GAME, MISSION, STATUS,
IMPLEMENTATION, experiments/v0.0.11/README.md, research/index,H1,H10,H31–H34.
2.docs/NATIVE_STANCE_REFERENCE_RESULTS.md and original H33protocol/receipts;
artifacts/v0.0.11/development/h33-posthoc-reset-geometry-01.json.
3.scene.py compile-then-fixed-translate recipe; shared_motor.py NativeMotorMap;
shared_motor_task.py NativeReferenceDriver and restore; native_pair_motor_rl.py
constructors/reset; native_stance_probe.py evaluator/audit; H33qualification/
guard/counter/render scripts and all matching tests.

Likely NEWfiles: src/joxing/native_reset_placement.py,
tests/test_native_reset_placement.py, scripts/run_native_reset_placement.py,
scripts/qualify_native_reset_placement.py, experiments/v0.0.11/native-reset-placement-01.json.
Additive named-placement dispatch in native_pair_motor_rl.py only if needed.
One owner for scene/core placement interfaces; referee/history schemas untouched.
third_party, all old protocols and sealed H31/H32/H33roots are read-only. Generated
XML/runs stay outside third_party; no large arrays, weights or caches in Git.

## Implement in this order

1.Add frozen NativeResetPlacementSpec with explicit names for corrected opposed
and corrected separated-lane starts. Desired actual world pelvis XY:
opposed A(-.8,0),B(.8,0); lanes A(-.8,-1.25),B(.8,1.25). Keep original A0/Bπyaw
and each native reset height. Reject unknown modes, malformed/nonfinite states,
wrong native model/seed or missing owners before creating an evaluated world.

2.Add build_reset_scene(canonical_model, initial_by_player, *, spec, seed).
Build the ORIGINALopposed scene/constants first. Use NativeMotorMap and scratch
MjData with each actual native reset state to measure its world pelvis origin.
No mj_step. Compute deltaXY=desiredXY-actualXY, then translate ONLYtwo fixed
root body/frame XYpositions, synchronizing model and spec. Preserve every other
compiled numeric array BIT EXACT as H33does, including derived constants. Do
not recompute constants with mj_setConst or recompile only the translated XML.
No actual joint-state/velocity zeroing, root rotation surgery or height changes.

3.Bind mode, seed, desired XY, computed fixed offsets, original native reset
qpos/qvel/activation hashes, native-axis/order identity and compile recipe into
the scene report/hash and episode contract. Reconstruct with the named builder;
XML alone cannot reproduce the preserved compiler-cache profile. Keep frames
immutable after construction; validate the two root positions before every
entered step/snapshot/restore. Wrong-seed or changed-frame restore must fail.

4.For this finite probe, construct ONEactive fixed-seed NativePairMotorEnv per
trial. Its driver factory recreates original NativeReferenceDriver for the SAME
seed and rejects other seeds. Reuse the actual frozen actor/normalizer/donor
serially; close native trial data before constructing the next world. Existing
base constructors/default contracts and walk arithmetic must stay unchanged.
Do not add a mutable per-reset scene behind an already frozen environment hash.
A future training adapter may require a separate registered lifecycle design.

5.Record actual reset pelvis XYZ and all interactor geometry distances after
scratch forward. Require XYwithin1e-9m of declared targets, native height/state
bits unchanged, zero interactor penetration and minimum interactor surface
clearance≥.02m. Enumerate ALLcross-player physical surfaces, not scoring shells
only; use nearest-distance queries with a bounded search beyond this threshold
where no collision contact exists. Contact-list absence alone does not prove
clearance. Keep self/floor interactions and collider masks unchanged.

6.Source tests: four original phases×two placements, exact root/axes/native-state
mapping, all other compiled arrays/muscles/masks/material/equality/solver bits,
complete geometry coverage, bad mode/NaN/foreign seed/hash/corrupt fixed offsets,
atomic paired actions, reset-only writes and203+17held/next-decision replay.
Old136controls remain green; actual checkpoint and command reconstruction must
remain strict within each new profile. Cross-placement dynamics can change with
contact geometry; do not demand or claim an unchanged physical trajectory there.
Never loosen old default/replay gates to admit a changed reference or model.

## Freeze a small physical protocol before outcomes

After source passes and exact case IDs/fixture physics are known, publish protocol
and source, anonymously verify the protocol, then fresh target source gate.
Use a NEWdedicated ~/joxing checkout on eligible xbabe4, its qualified read-only
H31runtime/teacher inputs, one world/thread1. Check all four hosts/owned processes,
load≤16on32CPUs, available pressure estimate≥16GiB and disk≥10GiB before launch.
No duplicate jobs, services/driver/package changes or new compute purchases.

Eight NEWtrials: corrected opposed/walk and corrected lanes/walk, each seeds
71000/71016/71032/71048,3smaximum, zero learned residual, original walking goal.
No more400mswarp trial. Record unchanged SupportMonitor, all1msforces/contact
hashes/failure onsets/root motion/points and initial geometry. Require no positive
interactor normal force in the first100ms for either placement; freeze this before
outcomes. Later contact/falls in the opposed condition are measured motor outcomes,
not automatically an execution fault or a sporting win. Do not require opposed
three-second survival to call initialization correct; do require4/4paired3s
separated-walking completions to preserve the already observed transfer control.

Audit every trial and both203+17proofs in a fresh process, with exact native
commands/engine/contact/metric/frame replay. Proposed finite bounds to reconcile:
600swhole/setup60/source120/physical180/audit180/render2×30,8GiB/12process,
one world/thread1/60000TOTALphysics/256MiBraw/16MiBtext, firstlimit.
8×3000primary+8×3000audit+4×237holds+4096source reserve=53044<60000.
If source needs more than4096, shrink scope or explicitly reconcile BEFORE
registration; do not extend a failed running packet. Forecast archive, text,
temporary, media and fault reserves; reuse checked writes and pre-entry ledgers.

Export the ENTIREfirst fixed trial from both conditions, including falls/0points.
Fix camera framing prospectively: frame both whole bodies throughout walking,
using a declared bounding-sphere/FOV margin from archived geometry. H33's full
walking time sequence crops one athlete at the last frame; retain that original.
Render0physics; hash/decode/inspect first/mid/last and frame count of both exports.

## Stop, publish and next learning mission

Stop at first source/mapping/geometry/assistance/identity/numerical/resource fault;
retain every entered step/prefix and attempted/invalid/not_run row. Valid falls
remain negative motor outcomes. Record exact commands, source/config/runtime/
model/actor/normalizer/reference hashes, original epoch, PID/start/exit, resources,
all eight trial outcomes and independent replay/geometry decisions. Keep full
native tapes on the host; commit only compressed text receipts/manifest and small
fixed recordings. Update README/docs/book/H34/index; no release tag from this
prerequisite alone. Leave a next worker-ready mission or a bounded active task.

If initialization/controls qualify, the next learning task is bounded **upper-only
guard acquisition while using the qualified walking motor**, before close-range
opposed play. Reuse H10's46groups/122arm muscles, protect578teacher commands exactly,
retain causal own/opponent history and independent held replay. Freeze a named
guard-only objective, two training seeds, all fixed before/after evaluations and
survival/hand-target criteria before fresh PPO. This changes the measured H31
failure mechanism: it removes uncontrolled initial overlap and separates arm
skill learning from unqualified stopping. It is still motor learning, not a
competitive bout. Only later qualify stop/turn/guard under interaction, legal
strikes and frozen historical-opponent self-play before parallel game scaling.


## Implemented2026-11-03; target outcome pending

NativeResetPlacementSpec/build_reset_scene preserve original compiled constants,
measure source reset in scratch only, and translate two fixed root XYframes.
PlacedNativePairMotorEnv binds one seed/model identity and checks immutable frames
before decisions/steps/snapshot/restore. ResetPlacementFactory loads the actual
frozen controller once and constructs serial fixed-seed worlds. The existing
H33lossless recorder/replay format is reused; reset-specific geometry and gates
are retained separately in the new study wrapper, not relabelled old outcomes.

Signed clearance queries enumerate729cross-player physical pairs per reset,
including meshes; NativeCCD is mandatory. Queries capped at.05m return a lower
bound, not an exact minimum. Analytic clipped/near/penetrating sphere controls
and the original native-overlap control test signs/coverage. See
[official3.8API](https://mujoco.readthedocs.io/en/3.8.0/APIreference/APIfunctions.html#mj-geomdistance).

Development:27cases/1fixturefailure (missing A/Bpelvis body names), then27/27,
thenall163/163pass74.840s.674+674+3348=4696static sourcefixture steps across ALL
attempts; no scientific pretrained rollout/optimization. All source/fault receipts:
artifacts/v0.0.11/development/h34-development-01.json. No old criteria altered.

Success source loops3348 fit4096reserve; native max24474 primary and24474audit,
total53044. Exact maximal tape bytes78229424; source64MiB/text16MiB/media32MiB/
temp-fault16MiB reserves give212447152B<256MiB. Final protocol is
experiments/v0.0.11/native-reset-placement-01.json. Render derives a bounding
sphere from all visible athlete geoms and fits verticalFOV with1.6margin/min5m,
leaving HUD bands clear and preserving all archived frames. Target/source/run
must remain within the prospectively stated firstlimits; no extension on failure.


## Attempt01failure and prospective02 (2026-11-03)

01remainsINVALID: source RSS8655896576B>8589934592B at98.605s,0scientific
entries, all8trials not_run. No completed sourceJUnit. All identities exited.
[Retained fault](../artifacts/v0.0.11/native-reset-placement-01/summary.json).
The eager fixture held8native models. Repair retains2at most and full-byte SHA/
dtype/shape/solver/bodyposition receipts captured during ALL8geometry cases;
no model-array criterion is weakened. The numeric oracle also passes standalone.
All252production src/scripts files match the original74906d5 byte-for-byte.

New02root/protocol, SAME163sourcecases/scientificgates/resourcecaps. Keep failed
sourcecost4096 and currentsource4096 in combined guardreserve8192, so theoretical
formal total57140<60000. All local development12066static fixtureentries remain
separately reported as preregistration development, never called scientific data.
No oldclock/cap/result changes or native rerun of unexecuted data. See
[repair receipts](../artifacts/v0.0.11/development/h34-memory-development-01.json).
