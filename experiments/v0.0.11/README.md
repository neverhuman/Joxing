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

# v0.0.11 experiment book

Registered next mission, formal physics not started. Current v10frozen pilot takes precedence. Independent development: NativeBodyAdapter/BodyMeasurements, passive dual-state recording and visual playback. Read docs/V0.0.11_MISSION.md and docs/BOUT_RECORDING_MILESTONE.md. No boxing game, trained match or v11release is claimed.

Body/recording tests:3passed5.58s. Body test uses native joint velocities and Jacobians, no mj_step; recording fixture takes two diagnostic physics steps, separate from formal outcomes. Provenance nested-source mutation test1passed0.17s. Static MP4encoder smoke attempt01failed a PIL/ndarray boundary; fixedattempt02decoded one1024×576frame/25fps/0newphysics and was visually inspected. All failed/successful smoke receipts retained under artifacts/v0.0.10/D2/recording-smoke.json. No static snapshot is presented as a bout.

Next code: observations.py/history.py causal schema and simultaneous env.py/referee.py/replay.py per mission. Retain exact muscle ownership, source/rules/bundle identities and full contact/application evidence; only after freeze/Linux tests use the small v11qualification protocol. No native learning/promotion until contact/rules/motor/replay gates. First diagnostic footage must be produced from the first preregistered shared-world scenario, never selected as a favorable take.
