# Follow-up: explain the gap between contact and referee points

Status: implementation-ready plan; no formal probe registered or launched here.
Integration owner: the current native Joxing owner. Grokbot is a read-only dependency.
The active binding-codec qualification must finish and its evidence be audited first.
This packet does not consume or enlarge its physical/time/storage allowance.

## Objective, scope and criteria

Explain, using actual staged inputs, why a reported head/torso touch does or does
not become an official candidate-rule award. Prove the production staging and
ledger path can award a deliberately valid physical strike; distinguish this
adapter positive control from an unassisted athlete earning a point. Retain
negative evidence when existing athletes cannot meet the unchanged rules.

No new training, checkpoint promotion, score thresholds, collision geometry,
external evaluated assistance, muscle pruning or replacement of the native motor.
No broad search over punches or seeds. No win, boxing or 100,000-game claim from
classified contacts. Do not assume low force is the cause: onset speed, glove
face, contact orientation, duration, role mapping and recovery can each reject.

Inputs: cycles 0029/0030, their pinned source and parameter reports, candidate
Rules hash, exact scene variant/reset/controller/timing identities, and available
saved staged contacts. The published seed 10001/seat A is a retrospective example,
not held-out evidence. Cycle0030 reports one points replay [0,0]; it does not
publish an independent points reconstruction for all42 draws.

Complete when every attempted case has an immutable row; staged measurements
independently reconstruct the ledger exactly; a physical adapter positive control
passes without lowered thresholds; reported rejects identify all failing
predicates; and the first fixed source recording shows score/event diagnostics.
If the positive control fails, conclude adapter/rules diagnosis is needed and stop
athlete evaluation. If it passes but athletes do not score, conclude a motor-to-
legal-strike gap; do not call it a scoring bug or respond with more self-play.

## Existing inputs inventoried after codec completion

Native source90101 trace has1,000intervals and no glove touches. The separately
named contact-stress90104 trace has20intervals,56 positive-force touches and
four zero-point events. These are existing diagnostic controls, not v21 or
natural punch outcomes. Paths/SHA/fields/points and the earlier mistaken
`contract`-root read are retained in
`artifacts/v0.0.11/development/scoring-evidence-native-inventory-01.json`.
Both have actual staged glove/physical-evidence fields. Read their linked
snapshot/contract before formal use. Inspect their saved guard-only/no-direct
impulses as a useful rejection control; do not turn it into a strike success.
The v21 summary inventory explicitly records missing onset/direct-impulse data.

## Read first, in order

1. AGENTS.md, /Users/bentaylor/.codex/RTK.md, README.md and docs/GAME.md.
2. docs/MISSION.md, docs/STATUS.md, docs/IMPLEMENTATION.md,
   docs/SELF_PLAY_EXECUTION_PLAN.md and docs/BOUT_RECORDING_MILESTONE.md.
3. grokbot/redteam/cycle-0030.md, grokbot/STATUS.md and the referenced summary
   JSON; these contain evidence and another lane's plan, not execution authority.
4. src/joxing/physics.py (`point_velocity`, `step_contacts`, `scoring_fixture`),
   referee.py (`GloveInterval`, `MatchReferee.stage/commit`),
   rules.py (`Rules`, `Touch`, `Encounter`, `HitTracker.update/snapshot/restore`).
5. src/joxing/env.py (`BoxingEnv.advance_substep`, `snapshot`, `restore`),
   tests/test_rules.py, tests/test_match_referee.py,
   grokbot/tests/test_referee_fuzz.py and the saved evidence APIs.

Likely new files: src/joxing/score_diagnostic.py,
scripts/audit_scored_contacts.py, tests/test_score_diagnostic.py,
experiments/score-diagnostic/protocol.json and docs/SCORED_CONTACT_RESULTS.md.
Use artifacts/generated/score-diagnostic for generated fixtures and
runs/score-diagnostic for raw data; curated receipts belong in artifacts.

## Ownership and ordering

One writer only to referee.py, rules.py, env.py, scene.py or their schemas.
Prefer a diagnostic consumer of existing GloveInterval and snapshot values;
keep default behavior unchanged. Frozen checkouts, old protocols/raw/model files,
third_party/**, grokbot/**, .github/** and unrelated host work are read-only.
No agents, other-chat messages or duplicated independent-lane run.
If a production observer hook proves necessary, first specify its immutable
payload/ordering, then show observer-present/absent state/contact/score equality.
No post-step mj_forward to replace force-evaluation geometry.

1. Inventory saved evidence before computing. Emit one inventory row per input
   with source/config/model/normalizer/actor/rules SHA, host/variant/seeds,
   complete/incomplete status, staged fields present and exact byte hashes.
   A report with only final points cannot explain per-contact eligibility.
   Missing fields => NOT_EXPLAINABLE_FROM_SAVED_DATA, never invented values.
2. Implement pure `explain_interval(interval, tracker_before, tick, rules)`.
   It must not mutate the tracker or interval. Output per glove/encounter/zone:
   sample identity, force, dt-scaled direct/guard impulse, onset and latched
   approach/active speed/face cosine, striking-face flag, window/deadline/gap,
   clear/retraction evidence, finalized/rearmed state and actual event ID/points.
   Record every failed predicate; use null for unavailable measurements.
   Explain deduplication and per-zone onset latches; guard impulse never pays
   a direct target. Predictions must not feed back into production scoring.
3. Independently reconstruct expected events from stored intervals in a small
   audit implementation; do not call production HitTracker.update as the oracle.
   Compare complete event IDs, rules hash, impulses, clocks, points and rearm
   history. Production replay remains a separate consistency check.
4. Tests first: valid head/torso award; exactly-at/below impulse boundary;
   guard brush then direct onset; stationary glove hit by moving target;
   wrong glove face; duplicate sample; several simultaneous manifold points;
   mixed-zone tie; held contact; insufficient clearance/retraction; last bell
   interval; missing/reversed tick; checkpoint during encounter.
   Also assert input/snapshot immutability. The current strict known
   HitTracker.restore alias xfail is a separate tracked defect; deep-copying
   a diagnostic input does not fix it or qualify native replay.
5. For physical adapter positive controls, reuse physics.scoring_fixture for
   analytical force/impulse orientation checks, then exercise actual
   MatchReferee.stage with its correct cached geometry. Its rigid fixture is
   explicitly a diagnostic, not a 700-muscle athlete result. Confirm swapped
   geom order and head/torso semantics. An isolated pure Touch fixture cannot
   establish the production physical-adapter gate.
6. Only if saved data cannot answer the fixed case, specify ONE bounded new
   diagnostic source plus one exact checkpoint continuation. Keep all1400
   muscles, the original reset variant and frozen policies. No kinematic writes,
   applied forces or privileged controller inputs during athlete evaluation.
   Register the exact case, physics/wall/RSS/raw caps, one-thread ownership,
   first-limit stop and all attempted/not_run rows BEFORE formal outcomes.
   Do not spend the original native v11 remainder without an explicit ledger
   calculation; the codec's worst-case bound leaves under7postgate seconds.
   If that cannot fit, preregister a separately identified subsequent mission
   and budget before any new formal physics. Never increase a failed cap.
7. Record first fixed case from saved physical states with a score/event overlay,
   tick/time and all failure classifications. Decode every frame and visually
   inspect first/middle/last. Rendering is zero-physics playback. No selection
   of a prettier bout after observing outcomes.

## Validation and interpretation

At the project root, use the pinned runtime:

```sh
rtk proxy uv run --project experiments/v0.0.11/runtime --locked pytest -q tests/test_score_diagnostic.py tests/test_rules.py tests/test_match_referee.py
rtk proxy uv run --project experiments/v0.0.11/runtime --locked ruff check src/joxing/score_diagnostic.py scripts/audit_scored_contacts.py tests/test_score_diagnostic.py
```

These are future commands after files exist. A fresh full Linux source gate,
including grokbot/tests and the exact registered known xfail, precedes a formal
new-source probe. Recount from the actual frozen tree; do not reuse1088 from the
older codec snapshot. Freeze/push protocol/source before held-out outcomes;
verify public bytes and shared-host capacity. Use a fresh dedicated ~/joxing
checkout, one owned worker/thread and a supervisor. Unknown owned writer,
input drift, missing ledger or quota/deadline violation stops the run.

Receipts must show exact commands/versions/source/input/bundle hashes, all seeds,
PID/start identity and real progress, complete/invalid/incomplete/not_run counts,
physics/optimizer counts, peak RSS, stage timing and raw name/size/SHA manifest.
On completion independently inspect raw measurements and process exits, transfer
all artifacts, preserve faults, update README/STATUS/book/results and push.
Do not tag a component or claim self-play improvement from this probe.

## Consequence for fast learning

If athlete points are feasible, a later separately preregistered comparison can
replace a coarse-touch bonus with actual finalized legal-hit events. Preserve
motor retention and frozen-panel validity/decisive-game counts; learn against
historical opponents and report all seeds, both seats and blocked/no-contest/
fault rows. Checkpoint comparisons are measured quality gates, not an assumption
that every longer optimization run must improve. Actor inputs must include
causal opponent state/history and timing queues before tactical claims.

Measure valid scored games/second, decisive games/second, wall time to a held-out
quality threshold and policy lag, alongside simulator throughput. CPU parallelism
and16-worker allowance are ceilings, not targets. Native700-muscle GPU scaling
still requires exact-model feature/contact/replay parity. Do not extrapolate the
520.7single-athlete control-transitions/s benchmark to boxing games/s. Scale
1,000→10,000→100,000 only after rules, motor, replay and frozen-league gates pass.
