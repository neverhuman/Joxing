## Sensor foundation implemented; Linux qualification next

Added sensing.py SensorPacket/SensorQueue/ActorInput and an explicit
SensedBoxingEnv variant. Default env.py/replay.py and the D0diagnostic controller
remain unchanged. Sampling is every20ticks; actual delivery is polled every
physics tick. Actors see delivered vectors, causal8sample history, availability
and sample/receipt ages. Empty input stays None/invalid; prescribed diagnostic
hold does not call/advance actor RNG. Private seed-bearing diagnostic match IDs
are redacted from new actor-facing metadata. Snapshots/info remain privileged.

Sensed replay binds pending/delivered values to their actual past sampled engine,
clock/commands and schedule, validates on scratch state before live mutation,
and preserves monotonic attempted physics. Fixed drops/reordering/late-history
and protected old receipts are bounded. No random jitter/noise robustness claim.
New core29tests plus related regression subset62passed12.61s. Final full local
suite271passed202.62s; supervisor203.480325s. Ruffpasses. Zero-delay120tick fixture
matches default commands/history/engine/base application exactly; a57step actual
pending-sensor+held-command continuation matches all future physics/receipt state.
These are development fixtures, not formal D1outcomes or motor qualification.

New regression-only Linux protocol canonical39f948644b6004363eaa5426d8c7d21fe49789bd3987f2f0c716151be1e7bfdf,
experiments/v0.0.11/sensor-foundation-protocol.json. Freeze/public/anonymous-verify,
fresh dedicated xbabe3 checkout,271Linux tests required;180ssync/420stests,
1numericalthread and max16fixtureprocesses;8GiBRAM/10GiBdisk floors twiceatstart.
No automatic formal sensor trajectory. Exact next implementation packet:
V0.0.11_D1_PROOF_PACKET.md,8rows/2,748plannedphysics (951sensor-only +1,797D0oracle
reserve), new evidence/auditor/adversarial tests and source protocol still needed.
No new D1physics; cumulativeformal2642. No model/rules/learning/source-host-gate
change or v11tag. Old executed checkouts/outcomes stay immutable.

## D0 concluded with separate video repair; D1 sensor delivery ready

The first preregistered diagnostic MP4 is public in artifacts/v0.0.11/D0. All7
physical rows/2,642steps and3exact replays passed, saved-evidence auditpassed.
Original decoder MemoryError remains a failed supervisor stage; a separately
frozen0physics repair passed6Linux tests and decoded the same26frames/25fps/
1.04svideo. Visual inspection confirms both athletes lose upright stance within
1second,0opponent-glove touches/points in the source. This is reproducible negative
motor evidence, not boxing or learning. See V0.0.11_D0_RESULTS.md.

Full57file/55,053,851byte rawcopy is SHAverified locally; compact outcomes/audit/
manifest/video/visual/fault/exit receipts stay in Git. All30original+8repair
recorded identities exited; all-owned-CWD process scanempty12:48:00UTC. Both
source checkouts unchanged. Source-hoststrict replaypassed; extra Mac sigmoid
bit-exact recomputation failed by up to7.4505806e-9 and is retained separately.
No cross-host execution qualification or changed gate follows.

No compute active. Ready next worker packet: V0.0.11_D1_SENSOR_TIMING.md. Implement
bounded sensor delivery/availability and full pending-sensor replay, preserve the
D0zero-delay oracle and complete v11four-seed/adversarial battery before claiming
whole-version completion. Stable guard/reach in this actual shared model is the
motor bottleneck before competitive learning. No unchanged walking sweep, more
falling-video sweep, v11tag or competitive training. Current formalphysics2642,
post-repairnewphysics0; retain cumulative v11caps120k/900s/256MiB/max2simworkers,
global16andone numerical thread/process. Public-push new source/protocol before
formal next outcomes; inspect shared-host ownership/capacity afresh.

## D0physics/audit/render complete; original decode failed, bounded repair next

All7registered rows completed2,642physics/17.022529s, all3replays exact. Independent
saved-evidence audit passed.236Linux tests passed301.30s. MP4render completed,
but decode failed with MemoryError at list(reader); original supervisorhalt is
retained. No whole-D0pass yet. All30recorded identities exited, ownedscanempty,
141source+288upstreamfiles verified; source unchanged/actualscene matches compiled
model at12:38:29UTC. Full57file/55,053,851byte rawcopy independently verified.
Source90101had0opponentglovetouches/points. Artificial overlap source had56positive
glove touches, peak33.351243N (model),0.848797Ns summed per-glove impulse;
replay52touches on the19step tail. These are diagnostic stress numbers, not punches.

Repair source and tests are ready: bounded iterator never requests video length
hint, retains only first/last copies, rejects27thframe, uses1decoderthread. Local
9QA/recordingtests passed3.13s; Ruffpasses. New zero-physics protocol canonical
2df19cdea2a530d1c9dd1ac84a044712944a15ba5cfbd75650a20e06453dae2e.
Read V0.0.11_D0_VIDEO_QA_REPAIR.md. Public freeze/anonymous verification then fresh
Linux6tests and bounded decode of the exact338,092byte video, no physical rerun.
Original run immutable. No v11tag or learning claim. Conditional next packet
V0.0.11_D1_SENSOR_TIMING.md closes the measured ideal-sensing gap after D0closure.

## Active bounded recording supervisor —2026-10-27 12:30UTC

Public frozen source d2b582ee05041c21129b393945240b42e51153f8; protocol canonical
baec85f1208721b66d8476dcfbd850afc82c471701aa7ec247ca60f24ea99456, anonymous
public bytes verified. Fresh xbabe3:/home/ubuntu/joxing/v011-recording-20261027.
Supervisor3596577/start_ticks920622447; pytest3596803/start_ticks920622753.
141source+288upstreamfiles verified, startownedscanempty. Syncpassed2.09s;
236test gate advancing at12:30:53UTC, pytestCPU18.66s/RSS1.45GB, availableRAM
102.18GB/disk35.18GB, source diffempty. D0/audit/render/decode allnot_run pending
that gate; no formal outcomes or video yet. No duplicate/executing-source edits.

Compiled exact native model (0physics) retained outside Git and the64MiBD0raw
credit under artifacts/generated/v0.0.11/deployment-model: modelSceneHashd31ab280
92832ba5bdb21cc89c62fd8b6f85286f380dd401fb30723c3d93ed0d, full contiguous hash
in recording-launch.json. XML1,459,109bytes and self-containedMJB180,809,620bytes
have exact hashes; dimensions170qpos/170qvel/1400ctrl/1400activation,700muscles/
corner. One inline preflight quoting attempt failed before compilation/supervisor
or any physics; retained attempt-01. Corrected preflight passed without source
changes. No credential or model bytes committed.

Next check actual stage/result/events/progress/PIDstartticks/resources/source.
Require236Linux passes,7/7formal rows,2,642physics, all saved-tape audit checks,
26decoded frames/25fps/0renderphysics and all recorded owned exits. Decode pass
still needs visual inspection of both athletes/clock/honest diagnostic label.
Retrieve/checksum complete raw, preserve failures and not_run denominator, compare
actual scene hash with deployment model. Keep failed/partial footage classified;
never retry unchanged silently or claim learned boxing. If healthy, independent
work only. On completion update README/docs/book/results/public push; no v11tag
until its larger registered mission concludes. Receipts:
artifacts/v0.0.11/development/recording-{publication,capacity,launch,check-01}.json.

## Recording packet frozen locally; public deployment next

Local235test suite passed189.93s (supervisor190.753645s), then34focused tests
passed12.96s after the integer-clock video endpoint fix and final evidence checks.
Ruffpasses. New complete Linux count236must pass before formal outcomes.
Protocol canonicalbaec85f1208721b66d8476dcfbd850afc82c471701aa7ec247ca60f24ea99456;
7case rows/2,642plannedphysics. Source/protocol will be public and anonymously
byte-verified before a fresh xbabe3 bounded supervisor starts. No formal D0
outcome yet. Preserve the old218test source checkout. No v11tag.

## Linux foundation audited; D0 recording harness implemented —2026-10-27

The frozen foundation gate passed218Linux tests292.50s (stage296.655539s).
Independent post-exit check verified all133source+288upstream files,22unique
recorded identities exited, ownedscanempty and source diffempty at12:17:47UTC.
Receipt artifacts/v0.0.11/development/foundation-completion.json. No formal v11
physics or optimizer updates occurred in that regression gate.

D0 implementation now exists: boxing_evidence.py captures actual chosen/applied
700commands, decoded excitation, packet decision/application ticks, raw actor and
history arrays, decision-engine states, every typed signed contact and per-glove
physical/staging/event evidence. boxing_audit.py checks saved arithmetic and exact
replay tails. qualify_boxing_env.py/audit_boxing_env.py implement the seven-row
packet; bounded supervisor gates it behind a fresh full pinned236test Linux regression.
The renderer remains diagnostic/no-qualified-score, scratch-only,0physics, with
an optional8MiB inherited encoder file ceiling and pre-write raw reservation.

Development tests30passed9.94s including11semantic corruptions, actual-contact/
boundary/mid-command replay and missing/invalid attempt rejection. A further
fault-preservation test is included in the 235test full regression, now passed189.93s.
The small development contact tape passed20source+19replay interval audits;
this is regression fixture evidence, not a formal D0 result. Same-source FK/
actor/controller/HitTracker code is reused explicitly; force signs, arrival/
history/decoding and impulse arithmetic are checked from saved arrays. Saved
geometry frames/clearance are source measurements, not an independent physical
model. No sporting qualification follows from these checks.

Next freeze/public-push recording-protocol.json after local validation, then
fresh xbabe3 deploy:236Linux tests required before D0. Only then one bounded
untrained diagnostic run:7rows/2,642plannedphysics,12kphysics/180srunner/300swhole
post-gate/64MiBraw, firstlimit, one simulation process and one numerical thread.
Supervisor sync180s/tests420s; floors8GiBRAM/10GiBdisk, twiceatstart. Preserve
allfailed/incomplete/not_run rows and rawfaults; progress records20step hard-crash
bounds. Audit then source-hostEGLrender/decode; visual inspection remains required.
No formal D0outcome, new video, learned bout or v11tag is claimed yet.

## Frozen Linux foundation gate active —2026-10-27 12:01UTC

Public sourcef22cf8b338e00d179dd88df77456cb96ee2c24c7; foundation protocol canonical
bbcbf084d0169f1c3c869fb34195f1fd861a9876da8fe0c5cbef7fa5d8ea50d2 anonymously
byte-verified. Dedicated xbabe3:/home/ubuntu/joxing/v011-loop-foundation-20261027.
Supervisor3355366/start_ticks920445093; pytest3355576/start_ticks920445336.
133source+288upstreamfiles verified. Fresh launchscanownedempty,99.27GBavailable
RAM/36.43GBdisk. Syncpassed2.09s; testlogadvancing at12:01:13, pytestCPU13.31s/
RSS1.28GB, source diffempty. This is regression fixtures only, not a formal v11
rollout or learner. Bound180s sync/420s tests, one numerical thread/process.

Next heartbeat: inspect actual PID/startticks, full result/testlog/resources and
source, require218passes and all recorded processes exited. Preserve any fault
or negative result. If gatepasses, implement D0_SHARED_RECORDING runner/auditor/
new protocol and adversarial tests; freeze/public-push before its formal outcomes.
No autoD0launch, duplicatejob, executing-source edit or v11tag. Local docs/receipts
may advance while the frozen checkout remains immutable. Receipts in
artifacts/v0.0.11/development/foundation-{publication,capacity,launch,check-01}.json.

## Shared-loop foundation implemented; Linux qualification next

Current authority: docs/MISSION.md. The shared BoxingEnv now freezes both players'
observations before either command, owns all700muscles per corner, schedules
bounded delayed commands and stages all four gloves before a shared ending.
ReplaySnapshot restores engine, held command, queue, history, referee, RNG and
explicit JSON policy state atomically, including mid-command and active contact.
No competitive readiness or learned bout is claimed. Sensing remains ideal and
zero-delay; history supports received samples, but sensor-delivery queues and a
stateful learned controller are not integrated.

Local full development regression:218passed194.29s; supervisor195.236931s.
Receipt: artifacts/v0.0.11/development/shared-loop-full-tests.json. Targeted
queue/referee/env tests14passed6.83s; Ruffpasses. The original close-overlap-v1
fixture failed (1failed/12passed): measured glove clearance0.04843827953m, not
contact. Its result is retained. Separately named close-overlap-v2 shifts reset
roots0.727m each, producing an artificial overlap; positive native glove force
and exact mid-contact replay pass. This is collision stress, not a punch or a
valid boxing start. No root/pose writes occur during evaluated steps.

Next: public-frozen regression-only Linux foundation qualification, then implement
[the detailed D0 recording packet](V0.0.11_D0_SHARED_RECORDING.md): seven formal
case rows,2,642plannedphysics, first preregistered untrained1s recording, typed
application/contact evidence and independent all-attempt audit. Runner, auditor
and D0machine protocol are still missing. No formal v11physics, optimizer update,
new video, v11tag or native valid bout exists. Do not auto-start D0 from test success.

## v10published; diagnostic v11observations implemented

v0.0.10tag/source1c6796385754c13494f1aff59d64587ee9840874 and complete234,850,618byte
raw asset are public. Anonymous full-stream SHA256matches296a30a1d070102a3f7683b3062fb97c5661a80007ffbe5581cf086020b84cfc.
An initial anonymous API quota403was retained; authenticated release creation and
anonymous direct download verification succeeded. No training/source/gate changed.
Publication receipt artifacts/receipts/v0.0.10-publication.json. All35owned PIDs
exited; no native compute active. Negative v10motor result remains unchanged.

v11ActorObservation diagnosticv1 now has1,970named rawfloat32features: own79joint
positions/velocities,700activation, previouscommand, own/opponent11body pose/
linear/angular velocity, publicclock/points/height/ring. Opponent privatejoint/
activation/commands/goals/RNG/bundle/critic excluded. Ego yaw preservesheight/tilt;
shared-sample timestamps enforced. Native adapter now includes torso/upperarm/
forearm positions, rotations and angular velocities; source/root/anatomy unchanged.
Observation/body/history subset7passed5.78s; Ruffpasses. Prior full201tests188.09s
preceded this extension, not a claim of a new full204test run. Frame arithmetic
is not dynamic teacher equivariance; ideal sensors are explicit. No model uses
this new history/observation yet, and no BoxingEnv or learned bout exists.

Next exact packet: timing.py CommandPacket/ActionQueue and tests/test_timing.py,
then env.py/referee.py/replay.py and first registered diagnostic recording.
Read OBSERVATION_CONTRACT.md and current MISSION. No formalv11physics or v11tag.

## v0.0.10 concluded; v0.0.11 active implementation

Both fixed residual seeds completed8,192transitions/448updates each,32/32held-out
evaluations valid,2/2exact replays. Independent saved-evidence audit passed and
all35recorded PIDs exited/sourceunchanged/ownedscanempty10:58:23UTC. Neither
motor gate passed:81011mean−0.06625s,81029mean+0.01625s; each1/8finalfive-second
survivors versus required4/8 and+0.5s. No larger unchanged walking sweep.
Main251,144physics/495.045196s; replayreserve4,494physics/18.962701s;896updates,
64completed/8partial training motor episodes,0boxinggames. All668rawmanifest
files/276,715,588bytes verified after local copy. Sourceb1d1690/protocol72531389
and every attempted evaluation/model/update/fault receipt retained in
artifacts/v0.0.10/D2. See V0.0.10_RESULTS.md and final-accounting.json.

Next: V0.0.11_MISSION, shared two-player env/referee/application replay and the
first diagnostic recording. Body/recording/history foundations implemented;
actor observation filter/action queues/sharedloop remain. Current dev runtime
experiments/v0.0.11/runtime; old source/runtime/protocol remain immutable.
Publication/tag receipt follows in artifacts/receipts/v0.0.10-publication.json.

## Causal history foundation —2026-10-27

Added history.py with bounded per-game/per-corner/schema ownership, delayed/reordered received samples, causal decision-time selection, explicit padding/validity/sample-age/receipt-age fields, immutable copiedfloat32features and snapshot/restore. Three new pure history tests plus body/recording fixtures:6passed5.64s; Ruffpasses. No new formal physics/training and no changes to frozen v10source. This stores already filtered actor features; it does not expose private opponent body bundles safely by itself. Actor observation assembly, action queues, shared env/referee/application replay remain next. Receipt artifacts/v0.0.11/development/history.json.

## Linux gate passed; bounded D2 now active

Frozenb1d1690 on xbabe3 passed all196Linux tests275.02s (supervisor277.642415s), all recorded test processes exited. Formal D2uv2786514/Python2786532 started under supervisor2726271. First initial heldout evaluation completed, secondrunning, no invalids; confirmed mainphysics12,754including prior10,394, optimizerupdates0 at this check. Source diffempty. This is actual native motor evaluation, not a bout. Full32evaluation/two training/two replay rows retained; inspect live result/events/private workerprogress before any continuation. Exact source/protocol/launch in artifacts/v0.0.10/D2/deployment/progress-check-02.json. Current development additions do not change executing source.

Next heartbeat: inspect fresh counters/PIDstartticks/RSS/capacity/faults, then independent completed-attempt audit and raw retrieval/publication. Never launch a duplicate or another unchanged walking sweep. On v10conclusion activate V0.0.11_MISSION and recorded shared-loop goal BOUT_RECORDING_MILESTONE. No v10tag or learned boxing claim yet.

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

# Status and execution log

Date: October 26, 2026 (America/Denver).

## Initial source delivery

- Studied all four supplied specs; preserved originals under docs/references and reconciled decisions in SPEC_REVIEW.md.
- Created private `neverhuman/Joxing` using an existing authenticated neverhuman account; no token was copied into files or output.
- Inspected xbabe1–4 read-only; selected xbabe1 for bounded first runs.
- Pinned anatomy and msgym read-only submodules at supplied commits.
- Compiled full dual anatomy with 1,400 muscles, 84 equalities, 170 qpos/qvel, 165 bodies including four added gloves, 715 geoms. Total mass 121.565046 kg matches expected. Tested actuator dynamics/gains/bias/control ranges match upstream exactly.
- Implemented staged force capture, independent signed-impulse/momentum fixtures, pure candidate hit referee and snapshot/restore, canonical reward shaping, CPU benchmark and constrained muscle-reach search.
- Initial local validation: 27 tests passed in 2.88 s. Fixture momentum error ~1.3e-15 in the axis-aligned case; not empirical glove calibration.

Production scene fit/self-contact, complete simultaneous environment, foul/fall/boundary adjudication, unassisted motor competence, Warp parity and tactical self-play remain incomplete.

## Completed execution and receipts

- Source bootstrap committed as `464ff5b`; provenance fix/probe at `6a1ee06`; GPU variant/replay at `1c0c58c`; physical-scoring integration and formatting at `b0f3772`.
- An initial remote deployment exposed a provenance bug: `git rev-parse` in a copied submodule directory found the parent repo. Fixed by checking the repository root and falling back to committed gitlink pins; added regression test. Reran learning rather than publishing the mislabeled receipt.
- Verified learning command on xbabe1: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/joxing train-reach --workers 4 --generations 16 --population 24 --out runs/reach-seed7-verified`. Completed, no background training left. 384 training episodes plus 32 evaluations each for initial/trained/random controls. Error 0.338643 → 0.043747 m; wall 34.921 s; zero invalid episodes. Code receipt references `6a1ee06` and correct upstream pins.
- Run location: `/home/ubuntu/joxing/bootstrap-20261026/runs/reach-seed7-verified`; local copy `/Users/bentaylor/Documents/ChatGPT/JOXING/runs/reach-seed7-verified`. Every generation stores candidates, rewards, scenario seed, mean/std. Final checkpoint and before/after replay arrays remain there; curated result/checkpoint in `artifacts/receipts`.
- Rendered and visually inspected `artifacts/scene-inspection.png`, `reach-learning.png`, and before/after MP4. Replay shows assisted motor task, not an autonomous boxing match.
- Added real solver-to-referee fixtures: head 2 / torso 1 / guard 0; all emit exactly one event with physical impulse. Full two-human region/ownership integration still pending.
- GPU runtime probe rejected Warp 1.18 on existing driver. Warp 1.12 sees 4090; exact model rejected native mesh margin/MULTICCD combination. Named disabled-MULTICCD variant runs eight worlds but misses short-state parity (6.49e-4 > 1e-4). Preserve all three receipts; no reference change or host driver update.
- Native/locked local tests and lint pass. Final test output and benchmark receipts are stored under `artifacts/receipts`; latest counts appended before handoff.

Final source verification: **31 tests pass on both local Mac and xbabe1 Linux**; Ruff passes. Final xbabe1 single-world diagnostic median: **534.2 physics substeps/s** (three 500-step repeats). Local initial diagnostic median: **876.7 substeps/s**. These are not end-to-end training rates. Controlled Linux-to-Mac replay validation and all failed GPU probes are preserved alongside successful receipts. No training or GPU probe remains running.

## Next implementation gate

Follow P1 in IMPLEMENTATION.md: complete external shell/native-overlap review and explicit self-contact; build fitted anatomical frames and geometry-derived face/clearance/motion evidence; investigate scalar-root fall singularity. In parallel only through disjoint write scopes: isolate GPU divergence on small muscle/contact fixtures. Next learning milestone is unassisted stance and independent reach/return under gravity, before any tactical PPO self-play run.

## v0.0.2 — launched iteration

User requested a concrete next proof iteration and named it v0.0.2. Plan: `docs/V0.0.2_PLAN.md`. This deliberately integrates learned muscle motion with a physical glove target, geometry-derived strike evidence, one-event referee and full mid-contact replay before full-body competition.

Development mechanical check: an explicit conversion of the prior reach controller makes a real torso-target impact, earns exactly one point despite 165 force-producing contact steps, but does not withdraw/rearm. This is the baseline that v0.0.2 training must improve. Candidate rules remain unchanged. Fixed pelvis/zero gravity remain explicit diagnostic assistance.

Package version set to 0.0.2. New strike rig keeps all 700 muscles, adds exactly one inertial glove and a fixed spherical target. Source exposes strike-pilot/train-strike; frozen proof suite uses training seeds 7/19/41, 64 held-out reset scenarios each, and a four-worker CPU cap. Full unassisted stance and self-play remain later gates.

Pilot completed on xbabe1 under source 0eba6fe: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/joxing strike-pilot --out runs/v0.0.2/pilot-01 --workers 4`. 128 training episodes, 16 evaluation attempts, 34.809 s, zero invalids. Initial hit/withdraw rates 100%/0%; fitted policy 100%/100% on 8 development resets. First-contact snapshot at tick 368 reproduced the full final engine state exactly and matched the ledger. Pilot evidence remains in that run directory and local runs/v0.0.2/pilot-01.

Before proof launch, froze a smaller 12×24 training budget per seed based on pilot speed/success. Final evaluation IDs 20000–20063 and diagnostic/replay ID 20000 are disjoint from pilot's 10000–10007. Acceptance remains 80% hits and 75% strike-withdrawal per seed, no invalids, exact ledger/≤1e-9 state replay; physical role controls also require head=2, torso=1, guard=0 and passive=0 for the fixed diagnostic scenario. No final data has been used to select policies or alter thresholds.

## v0.0.2 — completed proof

- Executed source `3f2be02`; frozen protocol `36c229d690f79a60edeab302948529342980c2ae6fa766c40785a3512641848a`.
- Launch command on xbabe1: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/joxing train-strike --out runs/v0.0.2/proof-01 --workers 4`. Main PID was 1920776. Completed in 256.813 s; no worker or training process remains.
- Three seeds 7/19/41 each trained 288 episodes, then evaluated initial/final policies on 64 paired fresh resets 20000–20063. Total 864 training episodes, 384 evaluation attempts, 21 diagnostics, zero invalids. Every seed achieved 64/64 legal single hits and 64/64 strike-withdrawal successes. Baseline achieved 64/64 hits but 0/64 withdrawals for every seed.
- Role controls passed (head 2, guard 0 with actual physical impulse, passive 0). They are target-proxy variants, not anatomical zone or guard-occlusion validation.
- All three mid-contact restores matched the entire final engine integration state and event ledger exactly. Independent Mac replay/audit also passed for all three checkpoints, including raw evaluation accounting and event identities.
- Labeled plot/video inspected; fixed video is seed 7 / scenario 20000, not a selected favorable clip. Curated proof, pilot, protocol, raw archives, local/Linux test logs and replay audit are in `artifacts/v0.0.2/`. Full raw runs remain in both local and xbabe1 `runs/v0.0.2/`.
- Final validation: 40 tests and Ruff pass. The added preservation test confirmed native actuator parameters, body masses and inertias remain unchanged. A missing test import was caught during development and corrected before final verification; no physics/rules change resulted.

Read `docs/V0.0.2_RESULTS.md` for measured scope and reproduction commands. Next iteration should remove assistance through a validated stance/reset and balance gate, not jump directly into self-play. Existing Warp parity failure remains open.

## v0.0.3 — test plan prepared

User requested the next version of tests and a Git push. [V0.0.3_PLAN.md](V0.0.3_PLAN.md) defines the sequential implementation handoff against released source `88b76604e87e62e2681e2adc15af0651636183ac`: native-root stress tests, support/assistance audits, validated resets, a bounded full-body feedback learner, three-seed quiet-stance/push evaluation, replay and compute receipts.

The plan separates a completed test platform from a passed motor gate and measured learning improvement. Formal targets are 61/64 quiet stance and 58/64 recovery successes per seed, zero invalids, with all attempts retained. Initial compute cap is four CPU simulation workers; total planned cap is 40 million physics substeps/four hours, subject to fresh host availability. GPU simulator parity remains unresolved.

This delivery contains planning/documentation only. No v0.0.3 implementation, new physics measurement, training job or evaluation has started. Package/tag v0.0.2 and its evidence remain current. Next action is plan step 1, then the root/support/reset probe before learner work.

Planning validation: `rtk proxy uv run python` checked all local Markdown links and balanced code fences in the four changed documents; passed. Confirmed 768 planned evaluation attempts and unchanged package version 0.0.2. The first link-check invocation used unavailable bare `python`; rerunning through the locked uv interpreter passed. `rtk proxy git diff --check` passed. No executable code changed; the prior 40-test result remains the latest physics/code verification. The planning commit is pushed to `origin/main`; its Git commit identifies this handoff, with no v0.0.3 release tag.

## Public repository and training-guide clarification

At the user's explicit request, changed [neverhuman/Joxing](https://github.com/neverhuman/Joxing) from private to public using `gh repo edit neverhuman/Joxing --visibility public --accept-visibility-change-consequences`. Authenticated metadata confirms `visibility=PUBLIC`, `isPrivate=false`. Independent requests without authorization headers returned HTTP 200 for repository metadata and raw README. HTTPS `git ls-remote` with an empty credential helper and disabled prompts succeeded. Both upstream submodule repositories were independently confirmed public. Sanitized timestamped verification is in [repository-public.json](../artifacts/receipts/repository-public.json); no credential was stored or printed.

Added [documentation index](README.md) and [training guide](TRAINING.md), expanded the project README with the ordered v0.0.3 training/test gates, and linked the current iteration from GAME, COMPUTE and IMPLEMENTATION. The detailed plan now preregisters paired recovery-success improvement as the primary learning metric; quiet stance and stable seconds remain secondary. This prevents selecting a favorable primary metric after final evaluation. No final v0.0.3 evaluation has occurred.

v0.0.3 remains the next implementation version: root/support/reset validation, feedback environment tests, bounded SAC pilot, protocol freeze, three-seed training and 768 planned paired stance/recovery attempts. Pilot seed is 73; continuing to formal training requires at least 16/32 development starts to survive 5 seconds, with zero invalids. Bootstrap seed is fixed at 61003. Package/release remains v0.0.2. No training or compute job was started by this documentation/visibility update.

Validation: nine changed/new Markdown documents passed 61 local-link/anchor checks and balanced-fence checks; the public receipt JSON passed its visibility/HTTP assertions. `rtk proxy git diff --check` passed. No executable code changed, so the source/physics regression results above remain unchanged. Push the documentation and public-access receipt together to `origin/main`, then verify that the public default branch points to the pushed commit.

## v0.0.4 — implementation and pilot launch

User requested a major implemented push. v0.0.3 was planning only; [v0.0.4](V0.0.4_PLAN.md) implements its native-gravity prerequisites and a bounded SAC development pilot. New balance/support/group/environment/training modules preserve 700 muscles, 42 equalities, segment inertia and two 0.283 kg gloves. CPU-only Torch wheels are locked for Linux; simulation remains CPU MuJoCo.

Initial source inspection measured native keyframe equality error 0.010506. Reset-only Newton projection, 200 ms gravity preparation and final position/velocity projection produce accepted bilateral-foot states at <=2 mm penetration and <1e-8 equality error. Both projections and their correction magnitudes are logged; no state writes occur after the measured episode begins. Native-root coordinate singularities are not assumed resolved by a short finite probe.

Development checks: `rtk proxy uv run pytest -q` passed 57 tests; Ruff passed. A 64-transition/two-worker smoke run (`runs/v0.0.4/smoke-01`) completed with four paired evaluation attempts, zero invalids and no policy improvement, as expected before the 10,000-transition learning warmup. The local development probe (`runs/v0.0.4/probe-dev-01`) accepted all 32 resets and completed twelve 0.8-second root-stress cases at 2/1/0.5 ms without numerical faults. These pre-commit development receipts carry the source hash and are not the published pilot proof.

xbabe1 fresh inventory: load ~5.75, 41 GiB available RAM, 328 GiB disk free, RTX 4090 idle. Use the existing dedicated checkout and at most four CPU simulation workers, one BLAS/OpenMP/Torch thread. Commit source before Linux validation/probe/pilot. Planned command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run joxing train-stance --out runs/v0.0.4/pilot-01 --steps 100000 --workers 4 --seed 73 --wall-s 1800 --eval-count 32`. All 64 development evaluation attempts remain in the denominator. Formal three-seed motor proof and self-play require the stated continuation gates; their completion is not implied by launching this pilot.

Source `92db82688ee9930ed40aae60d6510ef4b6d210a1` passed 57 Linux tests (20.70 s) and all 32-reset/twelve-case root probes. Pilot launched on xbabe1 as PID 2868420; log `runs/v0.0.4/pilot-01.log`, live progress/checkpoint in `runs/v0.0.4/pilot-01/`. Source uses CPU PyTorch 2.14.1+cpu / stable-baselines3 2.9.2. Source/wall/physics/seed identity is persisted before learning.

Independent smoke audit caught JSON key-order leakage in support observations: a sorted file changed restored right/left/non-foot channel order. Fixed observations/trace to use explicit semantic order, added a sorted-JSON observation regression, and verified closed-loop smoke continuation now has zero full-state error. This corrects snapshot decoding; normal unsorted live training observations and the running pilot's learning path are unchanged. The failed development audit is preserved before its corrected audit. Snapshot guarantees in this release cover control boundaries (including during a push), not arbitrary interruption inside a Python control step. Learner checkpoints are policy/optimizer bundles; full replay-buffer/RNG training resume is not implemented.

Additional verification: malformed snapshot state, shape, unknown fields and clocks are rejected before mutation. Optional motor integration exercises actual SAC gradient updates, checkpoint reload and external-horizon bootstrap/final observations. Latest Mac suite: 63 tests pass with motor dependencies (16.92 s); Ruff passes. Linux initial suite was 57 tests before these additional restore/learner cases; rerun the full final suite after the pilot exits.

Initial-policy Linux-to-Mac audit preserves fall outcome but misses the predeclared 1e-5 full-INTEGRATION-state tolerance: max state error 2.115e-4 (solver warmstart), qpos error 4.707e-8, qvel error 1.316e-6. Same-host sorted-JSON continuation is exact. Preserve the cross-host failure, do not relax its tolerance or claim closed-loop platform parity. The bounded pilot is also update-bound: warmup rollout throughput is materially higher than throughput after SAC begins gradient updates. Physics-only throughput is not reported as learner throughput.

## v0.0.4 — completed implementation and measured pilot

Pilot source `92db826`, frozen protocol `4591a775b84bb2a278df9a8699fb85edd46f8020e0452c6e58b61774844e0a9c`, seed 73. PID 2868420 finished. Training stopped at its 1,800 s wall cap with 39,040/100,000 control transitions and 388,573 training physics substeps; 60,960 scheduled transitions remain unrun. Final evaluation then completed all 64 planned paired attempts, zero invalids; total wall 1,848.367 s. Initial/final mean survival 1.017531/1.023000 s; paired gain 0.005469 s, 95% reset interval [-0.000281, 0.011625] includes zero. Initial/final five-second survivors 0/32 / 0/32; final twenty-second successes 0/32. No convincing motor improvement. Continuation gate failed; zero of the proposed 768 formal evaluations were attempted.

Source fixes/audit tools at `675caf8`, `de2c09b`, `91a2573` preserve model/action/observation identities and the live training path; source on xbabe1 was deliberately left at `92db826` until the run completed, so run provenance remains accurate. The later code fixes sorted snapshot decoding and validates it before mutation. Final suite: 63 tests on Mac (16.92 s) and Linux (26.54 s); Ruff passes. Linux initial/final policy snapshot continuation has zero full-state error. Mac independently confirms checkpoint/row/summary identities and zero repeated local-restore error, but strict cross-host full-state parity remains failed: trained max error 2.412e-3 (qpos 8.102e-7, qvel 1.600e-5); outcome and tick still match.

Post-run command on xbabe1: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python scripts/benchmark_balance.py --run runs/v0.0.4/pilot-01 --out runs/v0.0.4/benchmark.json`. Three repeats at 1/2/4 workers measured medians 65.71/131.81/240.52 control transitions/s. This includes policy inference/support/reset and excludes optimization/referee. Gradient-phase SAC throughput is roughly 17–18 transitions/s. An in-flight reset diagnostic found a large entropy/reward-scale mismatch; a preregistered ablation is required to establish its effect. No GPU compute was used.

Published portable evidence/policies, raw rows, progress, model/reset/root audits, all failures, test logs, benchmark, independent audits, inspected fixed-scenario plot and video are in `artifacts/v0.0.4/`. The run, evaluation and benchmark are complete; no iteration job remains active. See [V0.0.4_RESULTS.md](V0.0.4_RESULTS.md) for exact reproduction and next gates. Remaining work: reward/entropy and optimizer-throughput experiments, qualified motor skills, full boxing geometry/referee and self-play. Game points are unchanged. Final source/evidence/docs are pushed together and tagged v0.0.4 with this limited scope.

## v0.0.5 — preregistered controlled wave

User requested study, the strongest next test wave, execution/conclusion and a public GitHub push. Plan committed first as `09cbacb`; [V0.0.5_PLAN.md](V0.0.5_PLAN.md) and [experiment book](../experiments/v0.0.5/README.md) define exact interventions, splits, conditional replication and stop rules. No model, muscle, observation, physical failure or boxing-point rule changes.

Implementation adds strict immutable protocol, constant-control sanity test, real optimizer profiling, matched 2×2 reward/entropy SAC screen, conditional two-seed confirmation, independent raw-row/actor/action-replay audits and report curation. Four workers, one thread each; 40,000 transitions/900s per run, at most eight training runs. Confirmation scenarios stay unopened until each scheduled final actor is frozen. The five-second and formal motor gates remain unchanged.

Development smoke: 64transitions, seed149/armB, four workers, 12 optimizer updates, IDs62010/62011, four valid paired evaluations in11.95s, no motor gain. First invocation failed before execution because the parent run folder did not exist; fixed creation while preserving overwrite rejection. Both attempts are retained in `artifacts/v0.0.5/development/smoke.json`. Latest pre-final suite:69passing tests; an additional fault-accounting test is being validated. Source/protocol will be committed before shared-host execution; formal outcomes remain pending at this entry.

v0.0.5 live execution: source `6ac2bad`, protocol SHA256 `7a4ca4a6b841ca39ca8141db83d1e0e05b01d49054408d71197ac28ecff97fe5`, frozen source SHA256 `8b45216daf642dc2ec9e76b37d0e76533310666d26d7c8943e6e9bf33ba20ba0`. Mac70tests/34.72s and Linux70tests/54.91s passed before launch. Main PID3802532, log `~/joxing/bootstrap-20261026/runs/v0.0.5/training-output.txt`, registry/events under `runs/v0.0.5/wave-01/`; source checkout must remain frozen until exit.

B0 completed80valid evaluations; selected constant0.05 averaged1.0414s on32screen starts. P0 passed: optimized91.99vsreference19.62updates/s,4.69×. S-A completed40,000transitions, initial/final1.01775/1.01156s, gain−0.00619s. S-B completed40,000, final1.09094s, gain0.07319s, reset95%CI[0.06841,0.07766]; both arms0/32five-second survivors. At this entry S-C is running, S-D/selection/conditional replication/X0 remain pending. No learning claim is finalized from these partial screen results.

User explicitly requested increased host limits. Registered resource addendum `b3ec872` raises aggregate allowance4→16workers without changing the active scientific protocol. H0 used4/8/12workers alongside the four-worker wave; nine timed samples/14,400transitions, zero invalids, medians220.48/436.01/646.79rollout transitions/s; excludes optimization. H0 completed;16-worker standalone test is reserved until the wave exits. Partial Linux actor/action continuation is exact. Mac action replay errors<7e-11 but closed-loop full-state errors3.58e-5/9.21e-4 exceed unchanged1e-5; repeated local continuation is exact. Failed platform parity is retained. Mechanical reset validity does not imply rest: B0 starts tilt8–9degrees with pelvis-tilt velocity~1.1rad/s, so describe this task as initial-fall recovery.

Original v0.0.5 wave completed in2,358.383s:160,000training transitions,336evaluations, zero invalids,2,148,064accounted physics/preparation steps. S-D finalmean1.120156s/gain0.102406s was best; no arm passed0.25s improvement or five-second continuation. Four original replication slots not_run. Linux independent accounting/actor/action continuation audit passes; Mac accounting passes but strict cross-host closed-loop parity fails. Original job PID3802532 exited. H0 completed all4/8/12/16pool sizes,24,000timed transitions, zero invalids;16-worker median844.67transitions/s with optimization excluded. Both plots and fixed scenario60000/armD video were inspected; instability is visible.

User explicitly authorized increased workers AND training budgets. Extension plan/protocol/source `76f6e5c` was committed before extension outcomes.72tests pass Mac55.40s/Linux83.23s, including two parallel trials, deferred initial evaluation and peer-stop preservation. Separate frozen checkout `/home/ubuntu/joxing/v005-extension-20261026`; source SHA256 `ed1379dccbcbe57fe558f99643f637b4aa8224411ee2dfeae5aac5512ef9d37b`. Launch UTC19:46:43, PID4172953; log `runs/v0.0.5/extension-output.txt`, per-cell logs/registry under `runs/v0.0.5/extension-01`. Four concurrent learner jobs × four simulation workers, one thread each;160,000transitions/2,700seconds per cell. New screen seed547/IDs64000–64031; conditional confirmation seeds761/983/IDs65000–65031. All original physics/learner conditions and gates remain fixed. Preserve both waves, do not edit active extension source, and finish independent audits/results/public push after its final decision. Deployment initially referenced unavailable FETCH_HEAD in the clone; explicit76f6e5c checkout succeeded before any extension experiment.


### v0.0.5 extension screen freeze and state/history review

Source `76f6e5c` remains frozen on xbabe1. All four extension screen trials completed 160,000 transitions and 39,500 updates each. Means A/B/C/D: 0.893156/1.313469/1.245156/1.126563 seconds, initial 1.059063; no five-second survivors. Selection B cleared the registered thresholds (gain 0.254406s, versus A 0.420313s). Before confirmation outcomes, curated every completed screen actor, row, progress record and baseline; independently recomputed paired summaries/selection and wrote `artifacts/v0.0.5/extension/screen-freeze.json`. Screen denominator: 288 new attempts, zero invalids; eight reused diagnostic rows excluded. Confirmation A/B jobs for seeds 761/983 are running under the same parent PID 4172953, four learners / sixteen simulation workers total. No source or action/observation contract changed.

User requested opponent state, motion history, responsive muscles, jitter handling and many short games. Read executable control/contact/rule/scene code and published `docs/STATE_HISTORY_SELF_PLAY_AUDIT.md` with a worker-ready handoff. Actual stance observations include 85 generalized velocities and 700 muscle activations; SAC has no recurrent state, opponent input, tactical referee or league. Offline command audit retains all eight original and extension screen traces. C showed sharp command changes and D saturation in the original preselected trace; these are diagnostics, not causal limb-jitter claims. A new isolated-actor timing script will measure inference separately from the unqualified wall-clock physics loop; no additional training or task alteration is involved.


### State/control diagnostics and upstream checkpoint inspection

Measured eight extension screen actors on Mac with `scripts/benchmark_actor_latency.py`: one Torch thread, 100 warmup calls and 1,000 timed calls per actor/batch case. Single-observation medians 0.062–0.064ms, p99 0.067–0.124ms; no >10ms calls among 16,000 batch-one/batch-four samples. Repeated fixed observations, no physics/transport/optimizer; not a full-game deadline guarantee. `scripts/profile_control_path.py` ran three unprofiled and three profiled recorded-action continuations through FaultRecorder on selected B scenario64000. Median unprofiled139.65control steps/s; middle profiled repeat ~76%native step1/step2, ~6%snapshot, ~2%point velocity. Same physical outcome/tick, maximum integration-state reference error1.997e-11, 6,498diagnostic physics steps within the reserved1million. Both JSON receipts retained under `artifacts/v0.0.5/extension/`; no active remote code changed.

Read pinned msgym observation, DynSyn layer/actor/config and evaluation source. Its public release has a 223,978,082-byte full-body checkpoint archive with SHA256 `2b2f0f1007d530db76463e7d3d5fb309b52c7b2a53649920ddfd9cf8a6880c57`. Downloaded to ignored `runs/upstream-checkpoint-inspection/LocomotionFull.zip`, checked the published hash, inspected only ZIP/JSON (no model/pickle/tensor deserialization). Saved metadata:45million upstream steps,3,601observation values,700outputs,138synergies, separate normalizer. These are upstream counters, not Joxing measurements. Source/metadata indicate scheduled DynSyn amplitude may be lost on load (saved top-levelNone vs schedule0.075); require reproduction before transfer interpretation. Added reproducible `scripts/inspect_upstream_checkpoint.py`, acquisition/metadata receipts and `docs/UPSTREAM_MOTOR_TRANSFER_PLAN.md`. No transfer rollout or upstream training was launched; v5confirmations continue unchanged. Local diagnostic scripts passed Ruff and their actual full-data executions.


### Confirmation-bank timing disclosure before fresh-seed outcomes

Review of `wave_extension.py:run_extension` identifies a distinction from broad preregistration wording: the frozen constant baseline evaluates confirmation IDs before the confirmation learners launch. Therefore the bank was not unopened by every process until final actors froze. Initial/final learner-policy evaluations are explicitly deferred until final actor save; training uses disjoint IDs110000–110063, and selectedB/config/criteria stayed frozen. This is disclosed in results before confirmation outcomes; historical protocol/addendum text is not rewritten. No execution/source mutation or retuning occurred. A future sealed-bank formal proof must defer the constant comparator too, or preregister that exception precisely.


## v0.0.5 completed and audited

All eight extension trials completed 160,000 transitions and 39,500 updates each. Confirmation seed 761: initial 1.022875s, A 0.803000s, B 1.293531s; B minus A 0.490531s [0.452125, 0.532286]. Seed 983: initial 1.010531s, A 1.165406s, B 1.097188s; B minus A −0.068219s [−0.096314, −0.040187]. B improves over own initial and constant on both seeds, but fails registered superiority versus A on 983. Every final policy has 0/32 five-second survivors. Full extension: 576 new valid evaluations / 1,280,000 transitions. Both waves: 912 new valid evaluations / 1,440,000 transitions / 354,000 scheduled updates. The 14,736 completed stochastic motor episodes are not boxing mini-games. No formal motor proof or self-play was launched.

Extension wall time was 3,555.388s, with 13,841,563 formal physics/preparation steps. Both waves total 15,989,627; adding the one-million supplementary reserve stays below 40 million. Every source-unchanged assertion passes. Independent Linux audit passes raw pairing/counts/summaries/selection/actor hashes/accounting and exact policy/action continuations. Mac accounting and repeated local restore pass; strict cross-host closed-loop parity fails (maximum 4.247 in mixed integration-vector units), while recorded-action continuations agree within 1.20e-10. Outcomes/ticks match on the examined snapshots. No tolerance was relaxed; zero invalid evaluations does not erase this failed portability gate.

After completion, ran source-host `scripts/verify_wave_extension.py --run runs/v0.0.5/extension-01 --original /home/ubuntu/joxing/bootstrap-20261026/runs/v0.0.5/wave-01 --out runs/v0.0.5/extension-audit-linux.json`. Ran the same auditor on Mac against copied raw outputs; exit 1 and the failed cross-host receipt are preserved. Ran final `scripts/report_wave.py` against completed extension data. A copy in ignored remote `runs/v0.0.5/report_wave_final.py` used `MUJOCO_GL=egl` for source-host video; source/script/input hashes are retained. Inspected screen/confirmation figures, the stochastic training figure and fixed first-scenario 64000 video preview; falling motion is clearly visible. Registered PIDs 3802532/3901039/4126632/4172953 exited, and no Python job remains in either dedicated checkout (`artifacts/v0.0.5/process-exit.json`). No unrelated process or service was changed.

Final diagnostics found a small accounting omission: latency/profile environment construction performs 200 preparation steps before timing; each control-state inspection performs 400. Preserve published raw measurements and add `diagnostic-accounting-correction.json`: these four diagnostics total 7,698 control/preparation steps, within the supplementary reserve. Future latency/profile scripts count preparation explicitly. Timing conclusions are unchanged. Corrected derived-plot bottom margins; scientific rows are unchanged.

Updated README, documentation index, game/master implementation/training/compute guides, results, changelog, experiment book and evidence index. Curated the machine-readable release summary, every raw attempt/log/policy, audits, visuals and manifests. The frozen extension source passed 72 tests on Mac/Linux; final scripts pass Ruff and actual-data execution. Next work: feasible stationary/recovery resets and proposed upstream teacher compatibility/load tests, then P1/P2/P3/P5 causal opponent/history/jitter/game/league integration. Exact commands, caps, stop rules and read-first files are in `docs/UPSTREAM_MOTOR_TRANSFER_PLAN.md` and `docs/STATE_HISTORY_SELF_PLAY_AUDIT.md`. No active job remains to monitor or resume. Final public push/tag verification follows this completion record.


### v0.0.5 publication verified

Pushed final evidence commit `f737dcb494f87825447937904619251978896bf5` and annotated tag `v0.0.5` atomically to `neverhuman/Joxing`. Anonymous GitHub API confirms `private=false`, `visibility=public`; anonymous raw reads of the tagged README, results, state/history audit and release-summary match local bytes. Tag target was independently resolved and checked. `artifacts/receipts/v0.0.5-publication.json` records this post-push verification; main adds only that receipt/status after the immutable scientific release tag. All requested experiments are concluded, all owned jobs exited, and no follow-up training is running.


## v0.0.6 mission started

User authorized the next value-add phase and continuing20-minute progress checks. Created active heartbeat `advance-joxing-every-20-minutes` in this chat after checking no matching Joxing automation existed. Current mission/ownership/state transitions are in docs/MISSION.md. Registered v0.0.6 loader/normalizer/native-task qualification protocol before loading the released checkpoint or running its motor trajectories:20paired seeds71000–71019,default/restored decoder conditions,40attempts,one CPU world,900s/1Mphysics/1GiBraw first limit. Separate runtime targets checkpoint SB3/Torch/Gym/NumPy versions; root runtime remains separate. xbabe1 preflight load5.80/128logical,~40GiBavailable RAM,~318GiBfree disk. No v6job has launched yet.


### v0.0.6 implementation/preflight

Separate checkpoint-compatible runtime locked and installed. Native path-only adapter preserves source timing/options and700muscles; global SAC alias restoration, immutable normalizer, explicit decoder schedule and corrected save/load are implemented. Eight target tests pass; initial full pinned-runtime Mac suite80passed/64.41s. Initial7pass/1fail was a toy replay harness stepping after valid termination; preserve receipt and correct loop, no physics threshold changes. Float32 action arithmetic is checked against the actual pinned upstream wrapper. Native snapshots retain the real last actor observation and trajectory/RNG state in addition to engine integration state.

Release-source check initially hit anonymous GitHub API rate limit; authenticated read succeeded without storing credentials. Release modelpin c049 differs from Joxing2d, but complete trees differ only in README; no physical asset changed. Native environment patch only changes model-file lookup. Receipts are curated. No large-scale upstream training, native formal evaluation or transfer has launched yet. Current phase is source freeze and Linux deployment/test.


### v0.0.6 formal launch

Frozen source `97565e90012b59bb16db8c45e4aa7f095a905c3c`, source SHA256 `880ff9eecb52b4507ec1cd4e9b6cbb63eb5ffefd06dc36345bb2c54dea19211a`. Linux complete suite80passed/94.03s before launch. Dedicated checkout `/home/ubuntu/joxing/v006-20261026`; previous checkouts unchanged. Archive transfer/hash verified; fresh capacity load4.85/128logical,46GiBavailable RAM,319GiBfree disk. Registered one-world qualification launchedUTC21:30:56,wrapperPID700993. Exact command,source/config/resources/log ownership in artifacts/v0.0.6/launch.json and docs/MISSION.md. No model, reward, physics or criteria changed after launch.


## v0.0.6 completed and audited; next mission ready

Registered native qualification finished all 40 attempts in 77.2294137 seconds with 70,400 physics steps, zero invalid/incomplete/not_run and unchanged source/input hashes. Default and schedule-restored modes both complete 20/20 upstream three-cycle tasks (3.52 simseconds each), zero sampled support failures. Native diagnostic criterion>=16/20 restored passes; this is a frozen upstream walking controller, zero new optimization and no boxing/stance claim. Both modes reach the ceiling, so no schedule-restoration completion benefit is established.

L0 confirms default layer amplitudeNone versus scheduled 0.075. Synthetic normalized-action probes change by mean absolute 0.030902829,max 0.074400783; corrected save/load error 0 and actor tensors unchanged. Normalizer stays frozen. Linux independent audit checks all rows/pairs/physics/actor/normalizer hashes and first-seed 71000 step 10 action/closed-loop/repeat continuation in both modes: all state/observation errors 0, strict<1e-9 passes (9,960 audit steps). Mac accounting/hashes pass but strict compiled contract differs by5.55e-17body mass/9.76e-19inertia; preserve failure before reference replay, no relaxed tolerance or portability pass.

Source-host report initially failed Gym's 480 pxframebuffer; a display-only 640 x 360 repair in an ignored copy generated the final recorded-action video without changing frozen evaluated source. Video adds 3,520 physics steps; plot and fixed first-scenario preview were inspected and show physical walking. Curation checks 135 raw files byte-for-byte, retaining original pending-replay result and using the later audit for final status. All development failures and limitations remain in artifacts/v0.0.6. PIDs 700993/701010 exited; read-only /proc scan found no Python jobs in the dedicated checkout. No unrelated jobs/services changed.

Full 80 tests pass Mac 64.41 s/Linux 94.03 s, final 8 target Mac checks 8.45 s. Updated README, game/master/training/compute/docs index/changelog, experiment book, results and evidence indexes/manifests. Public v 6 push/tag verification follows. Current mission is the worker-ready v0.0.7 glove/timestep/longer-native-motion plan; no v 7 outcomes or job yet. The active same-chat 20-minute heartbeat must inspect actual process/progress/faults/Git and advance the current mission, preserving ownership and bounded resources. It notifies meaningful changes/completion/blockers and stays quiet on unchanged healthy state.


### v0.0.6 publication verified

Pushed final source/evidence commit `bd48f96913b7184121034d54500667cff1e3b4f0` and annotated tag `v0.0.6` atomically. GitHub repository metadata confirms private=false/visibility=public; anonymous tagged README/results/next mission/release-summary reads return200 and match local bytes. Remote annotated tag resolves to the completed release commit. Post-push receipt: `artifacts/receipts/v0.0.6-publication.json`. Main adds this verification after the immutable scientific tag. The 20-minute heartbeat remains ACTIVE, verified from saved configuration, with v0.0.7 as its next bounded mission. No owned simulation job remains active.


## v0.0.7 preregistration and implementation

Heartbeat inspected clean Git and actual /proc ownership: no Joxing Python jobs. xbabe1 load8.31/128logical,~29.6GiBavailable RAM and311GiBfree disk. Registered four variants and80paired long-task attempts before outcomes in experiments/v0.0.7/protocol.json. Native body-ID/reference-index mismatch confirmed by reading both observation and reward paths; implement explicit canonical-name map and exact native oracle first. Maximum4worlds/one thread,1800s/1.7Mrunner physics/1GiBraw plus0.3Mreserved audit/video; no competitive learning or reward changes. Sole integration owner in this chat; third_party and previous evidence remain read-only.


### v0.0.7 development validation

All four MjSpec variants compile with700muscles/85qpos/85qvel, unchanged native masses/inertias/joint/actuator/tendon/equality ordering and explicitly separate0.566kg glove equipment. Reference body-name mapping handles shifted IDs in observations AND reward; native feature/state/reward oracle is exact. Development smoke is retained; no formal72000bank opened. Mac full98tests passed in126.46s. Final targeted checks add parent-only shared-stop ownership after review caught a potential concurrent temporary-file rename race; parent raw-byte sampling now tolerates atomic receipt renames. No scientific thresholds changed. Separate checkpoint-compatible v7runtime locked. Source freeze/Linux full regressions and real v6snapshot continuation remain launch prerequisites.


Timing clarification: bundled117frames at100Hz gives1.17s per cycle and21.06s for18cycles, not the plan's approximate21.12s (which multiplied rounded v6endpoints). Protocol18cycles and native termination remain unchanged. Report sampled endpoint/timestep rounding explicitly; do not change a gate to align durations. This was inspected from saved v6reset metadata before interpreting v7long outcomes.


### v0.0.7 formal launch

Frozen source9e3ee335c3995c860b40237bf6f00ba155d5d62d passed99Linux tests/151.18s. Validation wrapper955614completed and launched finite qualification UTC22:12:19,wrapper972834/Python972838 in dedicatedv007checkout. Fresh load14.90/128logical,41.4GiBavailable RAM and318GiBfree disk. Source/protocol were committed and pushed before outcomes; no executing source is changed. Prerequisite oracle and H1precede T1. Sourcehost command/capacity and final tests retained.


## v0.0.7 completed and audited

All 80 registered long native tasks completed in 141.4419 s with 0 invalid/incomplete/not_run. Native 2 ms/gloves 2 ms/native 1 ms/gloves 1 ms all 0/20 horizon completions; means 4.129/4.171/4.145/4.178 s versus 21.06 s threshold. Original native baseline also fails; no glove-only causal failure or new training gain is established. T 0 native feature/state/reward and actual v 6 continuation are exact. H 1 median 1/4 worker loops 91.24/256.50 TPS, startup-inclusive 4.64/15.53 TPS. Linux independent audit all 80 pairing/actions/hashes/metrics/physics passes, all 12 first-seed replay paths have 0 state/observation error; completion diagnostic remains false.

Post-hoc firstseed 72000 replay ends at native mean joint error 0.06027–0.06139, with foot support, 0 nonfoot load,10.2–11.9 degree tilt and 2.1–3.2 cm drop. Knee/ankle/shoulder deviations and body spatial velocities retained; actual predicted commands match recorded actions exactly. These are sampled tracking failures, not demonstrated physical falls or long physical survival. No gate/reference/reward/normalizer/anatomy was changed.

Counted physics:249,460 formal+7,460 prerequisite/benchmark+38,730 audit+13,510 video+13,510 posthoc=322,670, within 2 Mcap. No optimizer/GPUphysics. All owned PIDs exited and dedicated checkout has no Python jobs. Full 98 Mac tests plus final 27 target,99 Linux tests pass. Plot/video preview inspected. Every outcome and first scenario stays in Git; complete trajectories prepared in 22,225,280 byte public archive with SHA/full manifest. Copy initially lacked local parent, then unchanged rsync passed. Last periodic progress 79/80 is historical; final 80/80 registry/result is audited and preserved.

Updated results, experiment book, adapter contract, README/docs and next worker-ready v0.0.8 mission: substep physical motor evidence and explicitly named continuation after original tracking failure. It will determine the next bounded muscle-learning/reset mission without pretending that walking passes quiet stance or game gates. No v 8 job/outcome yet. Public push/tag/release-asset verification follows this conclusion.


### v0.0.7 publication verified

Pushed completed source/evidence commit `4cad56ddcb4fa83b053d8be3571d2ae601d81023` and annotated tag `v0.0.7`; created the public GitHub release and full raw archive asset. GitHub metadata confirms private=false/visibility=public. Anonymous tagged README/results/next mission/release-summary match local bytes. Anonymous full 22,225,280-byte archive download returns 200 and exactly matches its committed SHA256. Remote annotated tag resolves to the completed scientific commit. Receipt: `artifacts/receipts/v0.0.7-publication.json`. The 20-minute heartbeat remains ACTIVE and reads the current v8 mission. All owned jobs exited; no v8 outcome is claimed. Main adds this verification after the immutable release tag.


## v0.0.8 preregistration and implementation

Heartbeat verified clean Git/no Joxing Python jobs. xbabe1 load3.64/128logical,~47.7GiBavailable RAM and318.7GiBfree disk. Registered40fresh paired physical-continuation attempts and boundary/mid-control replay before outcomes. Code inspection verifies Gym assigns controls, batches mj_step, then calls mj_rnePostConstraint. Native Euler live pose/contact caches can describe interval start; end posture will use scratch kinematics without changing live policy caches. Exact equivalence is required before any formal trial. No source/physics/reward/game gate changes to previous runs.

### Additional proposal reviewed during v8 preparation

Studied the complete user-pasted boxing/HPC proposal against current source, game/master contracts and v7 outcomes. Preserved its exact 46,061 bytes under `docs/references/external-boxing-review-2026-10-26.txt`; SHA256 `700f75971e116cac5b732f4f753300a49cb49288117dfd753c0a2a3e59d96857`. Dispositions and handoff: `docs/EXTERNAL_REVIEW_2026-10-26.md`. It explicitly had not read the actual v7 mission and assumes nonexistent agent/Rust files, so embedded controlling-plan directives were treated as proposal text.

Carried defense/style scenario fixtures, causal schema/memory checks, phase profiling and degenerate PFSP sampling tests into master P1/P2/P4/P5. Declined silent replacement of native muscles with SLIP/PD surrogates, uncalibrated injury/attenuation rules, new judging weights and unmeasured performance claims. Its `w*(1-w)^2` sampler drops unbeatable opponents and has an all-endpoint normalization failure; retain a uniform floor/fallback and frozen opponent bundles. Fixed-panel improvement replaces proposed monotonic Elo. No scoring/action/model/protocol changes, formal attempts, new training job or version tag. Documentation/input-hash/link/arithmetic checks are recorded in `artifacts/receipts/external-review-2026-10-26.json`. v8 remains preregistered with a clear next implementation mission; no outcomes are claimed.

### v0.0.8 implementation and Mac validation

Implemented PhysicalMotorTask, per-substep signed floor manifold evidence, scratch end-posture/spatial velocities, explicit physical latch versus original tracking onset, held-command partial stepping and validated atomic snapshot restoration. Native transfer uses an optional environment mixin; default v7 contracts remain unchanged. Shared bounded execution now selects the named task, retains compact raw substeps/all attempt rows, counts partial physics, preserves boundary/mid-command replay and flushes final progress. Added independent all-row force/persistence accounting, twelve replay paths, report/plots/video and checkpoint-compatible locked runtime. No competitive learning or formal fresh scenario was opened during development.

Mac targeted 40 tests passed; final 28 physical tests and full 127 regressions pass (147.35 s). Ruff passes. Preserve development failures/logs under `artifacts/v0.0.8/development`: initial threshold test expected 2 ms instead of correct 4 ms, then full suite found two fresh-task restore failures because constructor lacks init_time. Fixed missing-attribute preservation before source freeze; no gate/physics retuning. Earlier evolving development source was not fully snapshotted, a disclosed limitation; final exact source/runtime/protocol hashes are recorded. Source-host actual v7 trajectories and Linux full suite remain launch prerequisites. Prepare a new dedicated v008 checkout; inspect capacity again and commit/push source before any formal outcomes.

### v0.0.8 frozen deployment and Linux validation launched

Committed/pushed source `565942db98f6c3a38378d4dab5069d83571d88bb` before outcomes. Final source hash `6e2579711ac7806777db42c83971b028fd1046f2f2fffa13792bd978601bc8a7`; protocol remains `8c5f14240ec58048fe257bf49af7a07ed138474633096a84f8624553954b20e4`. Preserved raw failing-test output byte for byte, including its original trailing whitespace; source/doc whitespace check excluding those raw `.txt` logs passes. Created new `/home/ubuntu/joxing/v008-20261026` from public GitHub and copied exact read-only pinned assets from v7. No old executing checkout was changed.

Owned wrapper1392737 launched UTC23:13:01, locked sync completed in1.015s, Linux pytest child1392874 running. Fresh prelaunch scan found no Joxing jobs; recent load6.52/128logical,~39.7GiB available RAM and312GiBfree disk, then observed load3.98 during validation. Driver reinspects capacity after Linux tests and only then starts registered T0/H1/T1, audit and report/video sequentially. Two worlds/one thread maximum, original900s/700k/1GiB caps and950kmission cap; failed stage stops the pipeline. Exact commands, PID/source and hashed driver retained under `artifacts/v0.0.8/deployment/`. No formal outcome yet; MISSION lists live files and the next audit action. No v8tag created.

### v0.0.8 completed and audited

All40 registered episodes completed with zero invalid/incomplete/not_run. Native2ms/gloves1ms physical20s completions0/20 each; mean final boundaries5.033/5.026s. All physical latches are persistent excess tilt; original tracking fails at mean4.185/4.203s and physical latch follows0.839/0.813s later. Ignoring tracking termination does not repair stable walking or establish stance/boxing. Frozen native action/state/feature/reward oracles and actual v7 trajectories pass exactly. No new optimization.

Independent signed-manifold/load/persistence accounting passes all40rows. All12 boundary/mid-command fixed-action/closed-loop/repeat continuations have zero engine and cached-observation error with matching application/RNG/monitor/latches. Final progress40/40. Full127 Mac tests147.35s/Linux211.69s pass; Ruff passes. H1 one/two-worker median loops70.33/130.16 controlTPS, startup-inclusive5.26/9.92 on short40step cases, optimizer/referee excluded. Runner167,050physics/205.31s includes16,200prerequisite/benchmark, audit46,662/68.11s, video8,080/21.05s: mission total221,792 below950k. Wrapper completed UTC23:21:34; all stages exited, fresh /proc scan empty and frozen source diff empty.

All184 raw-manifest copies and70 curated raw files hash-verified. Full raw archive47,279,704bytes, SHA256 `5b5dbacf4f113f04551616eb4c2498bd5357c86fc49f340981ba4a440a5b60ab`; Git retains every outcome/start/spec, firstseed actions/substeps/replays, audit and inspected plot/video. Earlier development failures remain intact. Results/book/README/docs updated; public push/tag/release verification follows. No scientific gate was loosened.

### User-directed HPC and proper self-play route

Refreshed all four hosts read-only: xbabe1 idle4090/24GiB and47.1GiB available RAM, xbabe2 idle3090/65.7GiB available, xbabe3 idle3090/97.6GiB available but46.8GiB disk, xbabe4 low load about4 with512GiB physical RAM (free RAM not measured). Receipt `artifacts/receipts/hpc-inventory-2026-10-26.json`; no services/drivers changed. Added worker-ready `docs/SELF_PLAY_HPC_PLAN.md` and next `docs/V0.0.9_MISSION.md`: actual-actor batching/parity, complete-loop profiling, CPU scaling to16 subject to capacity, shared node-local actor inference and isolated CUDA learning/inference before any GPU-physics assumption. Motor/reset qualification and authoritative simultaneous state/history/referee/league remain prerequisites for valid games.100,000ten-second games require50Mworld transitions;60s games300M, arithmetic not a forecast. Current valid boxing games remain0. Current mission is ready H0 implementation/registration; no new job yet.

### v0.0.8 publication verified; v9 H0 source ready

Pushed completed scientific commit/tag `f30fa4a68372d36db9d5c8341ae0e69bf68b62ec`/v0.0.8 and public release asset. GitHub confirms public/private=false; four anonymous tagged files match local bytes and the complete anonymous47,279,704byte archive hash matches. Remote annotated tag resolves correctly. Receipt `artifacts/receipts/v0.0.8-publication.json`; tag remains immutable.

Implemented H0 `scripts/benchmark_inference.py`, isolated locked CPU runtime and preregistered32 cases/eight fixed retained observations. Scalar/batched prediction validates all700 outputs, ownership/order, float32 parity<=1e-6 and exact self-repeat. No physical state or existing CPU replay gate changes. Exclusive lease/fresh output, parent case registry, faults, raw samples, model/normalizer/source hashes, startup/RSS and120s/64MiB/zero-physics cap. Seven targeted tests and Ruff pass. Freeze/push before timing; deploy a fresh v009 checkout and retain the old checkouts/archives. No new simulation or training claim.

### v9 H0 attempted: batch parity gate failed, evidence retained

Frozen/pushed source `6a32a1dc872c0cfcf03eea4ce4990355231d155c` deployed to fresh v009 checkout. Prelaunch load3.56/128logical,42.1GiB available RAM,317.6GiBfree disk and no Joxing jobs. Wrapper1595614, H0uv1598351; locked sync and all7Linux helper tests pass. H0 stops as preregistered after9.6585s:3 completed cases,1 invalid backend-qualification case,28 not_run. Batch2 output max difference7.152557373046875e-6 exceeds1e-6. Scalar/batch1 repeat exactly. No qualified batching speedup or full-game throughput claim; zero physics/games, peak808,396KiB RSS. Independent all32row/sample/fault-tensor accounting passes. Original protocol/limit retained; no unchanged retry. All PIDs exited and /proc owned scan empty; source diff empty.

Retained raw inputs/fault arrays/timings, source/protocol hashes, launch/supervisor/capacity, Linux tests and independent H0audit under artifacts/v0.0.9. Current ready packet `docs/V0.0.9_H0_NUMERICS.md` locates tensor divergence without changing output/normalizer and preregisters full scalar CPU profiling before selecting shared inference, hot-path buffers or scaling. GPU inference/learning remains a separately qualified candidate; no driver changes or v9tag. Full self-play/HPC route and current mission updated; push follows.

### v9 N1 preregistration —2026-10-26

Numerical localization implemented as a standalone script with two frozen float32 variants (default/MKLDNN disabled), exact saved-fault reproduction, output-preserving hooks and unchanged actor/normalizer checks.11 targeted Mac tests pass; initial import-order lint corrected. Source/protocol will be pushed before Linux outcomes. Caps120s/64MiB/zero physics, one thread. No rollout source changed, no H0 tolerance relaxation or batching win. H1 full-loop profiler remains next independent implementation.

### v9 N1 complete; H1 source/protocol preregistered —2026-10-26

N1 source615c02a:2/2completed diagnostic rows,0physics/8.219s,11Linux tests, independent saved-array/hash audit passes. First divergence is first MLP linear `latent_pi.0`; identical normalization/features, same7.152557373046875e-6 action error with/without MKLDNN. Hooks output-preserving, model/normalizer unchanged; neither backend passes H0. All owned PIDs exited, no source diff. Evidence/artifact manifest in artifacts/v0.0.9/N1. No low-level kernel-cause claim.

H1 complete scalar physical pipeline profiler implemented/preregistered:6paired cases/3development seeds/300controls each, strict state/features/actions/telemetry equality, CPU1thread/native2ms/all700. Caps180s/30kphysics/128MiB,1simulation worker+zero-physics IPC helper. Separate startup/reset/RSS/cProfile attribution/serialization/disk and diagnostic IPC; no actor-service or game-throughput claim.15targeted Mac tests and Ruff pass; source push precedes physical outcomes. Next: fresh shared capacity scan, dedicated checkout, Linux tests, bounded H1 run, independent paired-array audit and choose one measured bottleneck for H2/H3.

### v9 H1 completed and independently audited; H2 implemented —2026-10-27

H1 source5b52d77, wrapper1725427 and all stages exited.6/6completed,18kphysics/45.875s,0invalid/incomplete/not_run;15Linux tests pass. All three pairs exactly match every input/action/integration state/reward/telemetry/reset and final snapshot. Median baseline62.481control transitions/s; physical step/evidence63.4–69.9%,actor+normalization22.7–29.3%,snapshot3.56–3.84%. Profiles implicate MuJoCo linear stepping, Torch linear and Python per-substep/contact work; snapshot removal alone cannot meet1.2x. Raw32,932,666byte archive/all rows and independent audit retained; first seed pair raw and all timing/function rows in Git. No motor or game win.

H2 CPU scaling implemented/preregistered in scripts/benchmark_physical_scaling.py and experiments/v0.0.9/scaling-protocol.json; worker-ready docs/V0.0.9_H2_SCALING.md.16serial anchors+62wave attempts,120controls,1/2/4/8/16workers twice with common start gate and exact per-control hashes. Caps360s/120kphysics/128MiB,2GiB reserved per worker+8GiB floor,1thread/16aggregate workers.19targeted Mac tests and Ruff pass; source push/new checkout/full Linux regressions precede formal H2. Fullmission H1+H2 planned111,600physics, remaining H3/audit allocation unchanged. Next: bounded fresh-host launch; audit every planned case/resource/exit and choose the next measured bottleneck.

### v9 H2 bounded supervisor launched —2026-10-27 00:09:55UTC

Frozen sourcea8f47d84d21cd8dd84cd7147989d7a2bc02d06c0 pushed before formal outcomes. Dedicated xbabe1:/home/ubuntu/joxing/v009-scaling-20261027; wrapper1799276. Fresh scan no owned jobs, load3.72/3.85/4.02,39.70GB available RAM/339.91GB free disk. Sync passed; full Linux regressions running uv1799404/Python1799409 with actual CPU/progress confirmed00:10:35UTC; formal H2 gated on return0. Source diff empty. See artifacts/v0.0.9/H2/{launch,launch-monitor}.json.

No H2 throughput outcome yet. Formal cap360s/120kphysics/128MiB,up to16workers/one thread.2GiB/worker+8GiB floor stays fixed; launch RAM may prevent16worker allocation if it does not recover above40GiB after tests. Retain that stopped/not_run outcome and use a separately registered roomier-host attempt if necessary; no relaxed guard. Next heartbeat reads actual tests/supervisor/private worker progress and audits all78attempts/physics/exit. Do not change executing source or launch duplicates.

H2 launch health recheck00:11:53UTC: full-regression log advanced41→65dots and actual owned pytest CPU ticks4278→12468; no formal H2 started yet. Supervisor remains active, no observed fault. Latest actual monitor retained in artifacts/v0.0.9/H2/latest-monitor.json.

### H2 attempt01 memory stop audited; roomier-host attempt02 preregistered —2026-10-27

Full146Linux tests passed237.35s. H2 sourcea8f47d8 stopped as registered at16worker reservation42949672960bytes before launching any16worker process.31completed cases (16anchors+15parallel),47not_run;4completed waves/1invalid allocation/5not_run. All15completed parallel cases exact against source anchors, no invalid physics/trajectory case;37,200physics/110.2765s, counters exact. First-repeat1/2/4/8steady43.82/82.21/155.66/226.44controlTPS; startup-inclusive9.85/17.98/34.41/56.18. No second-repeat median/16worker/game/training claim. Independent scripts/audit_physical_scaling.py verifies all rows/fingerprints/private counters, expected source hash, runtime test receipt and process exit; audit passed. All22recorded PIDs exited, owned scan empty/source diff empty. Complete small raw receipts/manifest in artifacts/v0.0.9/H2/attempt-01.

Preregistered new independent host attempt02 in experiments/v0.0.9/scaling-attempt-02.json (canonical66caf5eb8e90b3775d6894c7d3e86ca54e411eebd1ff514c5ea1797145a0bec2), docs/V0.0.9_H2_ATTEMPT02.md. Fresh xbabe3same3995WX,~102GB available RAM/49GBdisk/load8.76; no owned job. Existing uvbinary matches exact hash. Read-only pinned asset archive/source model/normalizer/config copied and verified under ~/joxing/deployment-cache, no physics run yet. New host must pass full153Linux regressions then4strict native source-v8 boundary/mid-control recorded/closed-loop state/action/all-contact/posture continuations (60s/12kphysics/32MiB), before unchangedH2(360s/120k/128MiB/16workers/1thread; fixed2GiB+8GiBfloor).11targeted Mac audit/qualification/worker tests pass, Ruff passes; source push precedes new formal outcomes. H1+attempt01 actual55,200physics/156.152s; conservative extra132k/420s stays within500k/900s plus150kaudit reserve. Preserve attempt01; next fresh dedicated checkout and gated bounded launch, then independent all-row audit and one measured H3 candidate.

### Roomier-host bounded attempt02 launched —2026-10-27 00:29UTC

Pushed source37aa36cf6577d453fbb9b367ab3a7963184f8540 before outcomes. Fresh xbabe1 owned scan empty; xbabe3 owned scan empty,104.87GB availableRAM/48.97GBdisk/load9.08/9.45/9.91. New dedicated /home/ubuntu/joxing/v009-scaling-attempt02-20261027; verified exact asset/model/normalizer/runtime/uv hashes. Wrapper2219857; locked sync passed9.63s; full153regressions running uv2220549/Python2220553 with49dots/6734CPUticks at00:30:14UTC,2.30GBRSS. Source diff empty. Formal qualification then H2 strictly gated; no newhost formal result yet. Source/protocol anonymously readable on public GitHub; wrapper exact bytes/hash and launch/process receipts in artifacts/v0.0.9/H2/attempt-02. Limits/worker cap/reservation remain unchanged; no unrelated jobs/services/drivers or upstream source modified. Next: actual fresh progress and all-row/host replay/exit audit, retain all failed/not_run outcomes; no v9tag before remaining H3scope.

### H2 attempt02 completed and independently audited —2026-10-27

Source37aa36c on xbabe3:153Linux tests pass215.75s. All4source-v8 candidate-host native boundary/mid fixed/closed-loop final state/cache/application/RNG/monitor and signed-contact/posture tails exactly match;10,394physics/20.2136s, independent raw-tail/final-JSON audit passed with zero new physics. H2completed78/78cases/10waves,all62parallel cases match serial histories,0invalid/incomplete/not_run,93,600physics/159.4054s.1/2/4/8/16median steady54.77/106.98/201.20/358.85/520.66controlTPS;startup-inclusive12.20/23.09/45.24/79.90/124.88. Both16worker repeats qualified. All71recorded PIDs exited,/proc owned scan empty,source diff empty. Allsmall raw receipts/arrays/host replay/audits/manifests retained in artifacts/v0.0.9/H2/attempt-02; xbabe1's31completed/47not_run attempt remains unchanged. No optimizer/game/learning claim.

Implemented one opt-in H3 allocation-reduced collector subclass in physical_fast.py; original physical_motor.py/default unchanged. Same warning/control/assistance/finite/substep/contact ordering/matmul/monitor/scratch logic; reusable explicitly-zeroed wrench and owned Python-float contact scalars reduce temporary/deepcopy allocations.9Mac physical invariants pass12.04s incl native/gloves exactengine/features/reward/contact,held-action cross-implementation restore,forces/sign/NaN/ownership.3harness tests pass;165collected/Ruffpass. No formal H3outcome. Protocolcollector-protocol.json canonicalb4ad191f93a6645557ebeb23e8a5cd80e643370195cc0a8423de4a150cc41373 freezes4source-v8 candidate replay gates then24timed cases/6counterbalanced4worker waves/300controls. Primary steady parent pipeline includes output/finalization;>=1.2x median and>=.95startup ratio,zero faults,strict rawtape/evidence equality. Caps240s/100kphysics/256MiB/4workers/1thread;planned82,394physics. Previous formal159,194physics/335.771s;maxH3fits500k/900s plus150kaudit reserve. Source push/full165Linux suite before formal H3; worker-ready docs/V0.0.9_H3_COLLECTOR.md. Decline an insufficient candidate without changing gates; v9not tagged yet.

### H3 bounded gated deployment active —2026-10-27 01:06UTC

Source4aea332e85051a2669c1973f71101f488393cd5c pushed before formal outcomes. Fresh dual-host owned scan empty; dedicated xbabe3:/home/ubuntu/joxing/v009-collector-20261027,wrapper2496537. Load9.72/10.03/9.95,103.01GB availableRAM/47.02GBdisk. Pinned assets copied into fresh checkout, oldcheckouts/default reference unchanged. Sync passed1.466s; full165Linux tests underuv2496660/Python2496671 show71dots/11,706CPUticks/2.69GBRSS at01:08:10UTC,source diffempty. Onlyafter all165pass does H3run four source-v8 candidate replay gates then equal4worker timing. Limits240s/100kphysics/256MiB/4workers/1thread and1.2x/.95performance gates unchanged. Wrapper exactbytes/hash/commands/public anonymous source verification and actual processes in artifacts/v0.0.9/H3/deployment. No formalH3outcome yet; next heartbeat reads realprogress then independently audits all28rows/rawtapes/contactarrays/source/PIDs. Keep insufficient/failed candidates and no v9tag until fullscope concluded.


### v9 H3 concluded; v0.0.9 complete and v10 mission ready —2026-10-27

H3frozen4aea332e85051a2669c1973f71101f488393cd5c completed onxbabe3 at01:11:58UTC, wrapper2496537/all29recorded PIDs exited and ownedscan empty01:13:31UTC,source diffempty.165Linux tests passed233.65s. Four candidate source-v8 replay paths exact then24/24timed cases/6waves,82,394physics/103.1101s,0invalid/incomplete/not_run. Independent scripts/audit_collector.py reads every raw action/obs/integration/contact/load/position/posture/dtype, trace/reset/initial/final snapshot and private counter. All24histories exact; no newauditphysics. Scope limit: intermediate full-snapshot hash lists checked but cannot fully regenerate unsaved application snapshots; frozen source asserts sourcepolicy/resource/deadline guards. Reference/candidate median steady204.1549/207.8888TPS,1.01829x fails1.2gate;startup85.6083/86.6889,1.01262x passes. Candidate declined, default unchanged; no gate tuning. Fullraw165,973,903bytes beforeaudit receipt,all24tapes archived; allcompact result/private/qualification/firstpairedraw/audit/exits/test receipts inGit. Two auditor corruption tests (right/left semantics, signed loads, NaN,dtype/value mismatch)+priorcollector/physical subset14passed13.11sMac, Ruffallpass. Audit-development mistakes/fixes explicitlyretained; no formalcode/results modified. Current167tests afterauditor additions, notclaiming167Linux rerun.

v9complete outcome:228case rows152completed/1invalid/75not_run,0incomplete; H2attempt01memory-wavefault separatelyretained.241,588physics/438.8812s mainH1/H2/H3;456.7588s includingH0/N1,allwithincaps. Exact CPU scalar rollout qualifiedto16workers/520.6553steadyTPS (124.8751startup), no batching orcollector promotion.0optimizerupdates/0boxinggames. Public resultsdocs/book/accounting/rawarchive/manifests completed; packagev0.0.9 metadata advanceswithout rewritingoldruntime locks/protocols. Reproducefromfrozenexecutedcommit. Next ready docs/V0.0.10_MISSION.md: existing-datafailure audit, exactzero-residual deterministic138/all700adapter, then onlygated two-seed8192-transitionphysical-survivalpilot with32heldout evaluations. Upstream stochasticdecoderlogprob omission explicitly motivatesnew latentSACcontract. No v10training started orownedjobactive; no competitivelearninguntilmotor/contact/rules/replay/history/leaguegates.


v0.0.9publication verified2026-10-27 01:29UTC: annotated publictag points6d901706586c328be23cc0c5991fb446443305f5; release424941535published, repositoryprivate=false/visibilitypublic. Anonymous tagged-results GET matches localhash; complete214,411,789-byte asset streamed without credentials and matches SHA256df87afe90e27237e8528156b6a8ec3af69dd84a97bcddeb53ec562d9a285c45f. All388archive members independently checked. Publication receipt artifacts/receipts/v0.0.9-publication.json livesoutsideimmutabletaggedmanifest. Publicmain includes readyv10mission; no v10run launched.


### v10 D0 input audit implemented and preregistered —2026-10-27

Fresh owned-job scans on xbabe1/xbabe3 empty; xbabe3load9.57/9.73/10.09,102.33GBavailableRAM/46.30GBdisk. D0 source scripts/audit_motor_inputs.py/protocol experiments/v0.0.10/input-audit-protocol.json implemented; seven unique saved H1/H3baseline seeds, per-block normalization/reset/reference/failure audit plus finite unwrapped seam velocity and scratch FK cycle-translation witnesses. No newphysics/policy/optimizer. Fixed1e-4velocity/translation thresholds; a proved reference inconsistency stops future residual learning, without causal-blame claim. Two synthetic tests/Ruffpass, independent v10CPU runtime lock created; no formal outcome before public source freeze. Exact protocol/source/test receipt artifacts/v0.0.10/D0/development-validation.json. Next bounded fresh-checkout Linux targeted tests/D0, independent raw arithmetic/accounting audit; full residual/pilot remains unimplemented/gated.


### v10 D0audited, learning stopped; R1named correction/parity preregistered —2026-10-27

D0source8471fe6a5baea8b79af500497685c9b70398d8a1 ran xbabe3:/home/ubuntu/joxing/v010-inputs-20261027,wrapper2761809.2Linux tests pass0.45s; all7retained cases/2,100controls audited2.917922s,0newphysics/actor/optimizer. Coordinate2seam−119.4601059qvel versus0consecutive-positiondifference; native resetgridexcludeslastframe. Allcurrent/futurecached goals/reset/normalizerstats independentlyexact, cycletranslationmax1.79e-7m. No fallcausality/learningclaim. Independent scripts/verify_motor_inputs.py reconstructs rawNPZarithmetic/all7case stats,passed; its initial weak-scalar division mismatch corrected/retained without changing frozen result. Source/PIDs:all4recorded exited,ownedscanempty/sourcediffempty. Initialrsyncmissing-owned-parentdirectory failure fixed beforelaunch and retained. Fullreceipts in artifacts/v0.0.10/D0. D0gate stopsD1/D2; originalpilot remainsunrun,notfailedlearning.

New separate named docs/V0.0.10_R1_REFERENCE_SEAM.md within unfinishedv10; source reference_seam.py opt-in deep-copy managercorrectsonlylast-frame translated-root0/2velocity. Allpositions/nonseamvelocity/default/third_party/700muscles/teacher unchanged. Four synthetic tests0.29s/Ruffpass. Frozenmachineprotocol1c6f405215cba2e49f18afd083311e39f0a0ad45974352b59ecec0043b5071f9 registers42referencecases/4,914samples,cycles0…20scalar/batch, allnative58reset-grid outputs exact, correctedtemplate1e-4tolerance. Full171Linux regressions before formal60s/32MiB/zero-physics R1, newdedicatedcheckout. Source/publicpush before outcomes. R1passstillrequiresD1neutralphysicalreplay/likelihood/masks andD2frozenprotocol beforetraining. No v10release/tag orperformancewinclaimed.


### R1bounded qualification deployment active —2026-10-27 03:11UTC

Source38eb93c5814912011e32b9c21c4ea2cf20d98927 pushedbeforeformaloutcomes. New xbabe3:/home/ubuntu/joxing/v010-seam-20261027; fresh100.21GBRAM/44.53GBdisk/load12.12, noownedjobs. Wrapper3411580, syncpassed1.167s; full171tests uv3411695/Python3411699shows12dots/1,219CPUticks/1.08GBRSS03:11:48UTC. Source diffempty. Only literal171passed+exit0permits R142cases,60s/32MiB/zeroformalphysics/onecomputeprocess. Exact source/wrapper/command/PID/publicprotocol receipts in artifacts/v0.0.10/R1/deployment. NoformalR1outcome/learningyet. Nextindependent all42rawarray audit andsource/process exits; D1stillgatesD2. OldD0checkout/stop/results unchanged, no v10tag.


### R1completed/audited; D1neutral residual implemented —2026-10-27

R1source38eb93c5814912011e32b9c21c4ea2cf20d98927:171Linux tests pass230.68s; all42cases/4,914samples completed0.779255s,0formalphysics. Independent scripts/verify_reference_seam.py reads everyrawarray and reconstructs native/correctedvelocity templates/resetgrid:passed,0invalid/incomplete/not_run. Positions/nonseamvelocities exact, changes confined todeclaredseamroot; all58native reset-grid outputs exact.14,011,146rawbytes, all5recorded PIDs exited/ownedscanempty/source unchanged. Allresult/audit/statuses/manifest+cycles0/20scalar/batch inGit, full42rawhost/localruns. No actor/optimizer/game/causalbalancewin.

D1MotorResidualEnv/harness+protocol implemented,7Mac tests pass11.80s. Neutral138toall700command mapping, immutableteacher/normalizer/reference bundle, physicalalive reward, separate bootstrap/trace masks, ownedterminalobs, pendingcommandmetadata/RNG andnorefreshdecision. Newtestcaught oldrestore dtype loss (~1.18e-8native rewarddifference): typedreference metadata/reconstruction added onlytothenewwrapper, oldbase/default unchanged; failure retained. SyntheticSAC138finitegradient/replay/target/save-load proof passes, notphysicaltraining. D1protocol136427ced9b436e6ff3ed13b2859e618db4d17f4e85c816480dacc8f914200d1 freezes4boundary/mid×fixed/closed paths/10,394plannedphysics,60s/12k/32MiB/oneworker/one thread. Full178Linux regressions before source-v8 action/finalstate/contact/telemetry/clipping proof. Source/publicpush before outcomes. Nextboundedfreshcheckoutlaunch afterfinal checks; no D2training orv10tag.


### D1bounded regression-gated deployment active —2026-10-27 03:32UTC

Source7acd207ca69784935657d5f6b2933aff5a23e3f9 pushedbeforeformaloutcomes, dedicated xbabe3:/home/ubuntu/joxing/v010-neutral-20261027,wrapper3601585. Fresh99.67GBRAM/44.60GBdisk/load10.25,ownedscanempty; model/n/config/archive exacthashes verified. Syncpassed1.216s; full178Linux suite uv3601771/Python3601775 advances54dots/4,643CPUticks/1.88GBRSS03:33:37UTC,sourceunchanged. Literal178passed+exit0gates fourneutral source-v8physical replaypaths under60s/12k/32MiB/oneworker/thread,10,394expectedphysics. Liveownership/source/wrapper/publicprotocolreceipts artifacts/v0.0.10/D1/deployment. NoformalD1outcome/trainingyet. Nextindependent4rowtypedwrapper/action/contact/telemetry/reward/mask/accounting/PIDaudit; retainfaults. D2runner/evalprotocolstillgated, no v10tag.


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
