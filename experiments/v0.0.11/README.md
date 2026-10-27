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
