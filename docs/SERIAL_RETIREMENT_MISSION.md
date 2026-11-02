# Next: bounded serial retirement of four native workers

Status: CONCLUDED FAILED native prerequisite.75Linux controls pass, but the2s
serial deadline expires on worker2 after0.85/0.92s first-two exits. Forced cleanup
then triggers105.9ms retirement hold fault. No normal worker3 close/parent initial
hash receipt; four-way state parity unavailable. Preserve original gates below.
[Results](SERIAL_RETIREMENT_RESULTS.md); next [H18mission](LIVE_RESOURCE_RELEASE_MISSION.md).

Historical preregistration: opt-in adapter/runner and source controls implemented. Registered packet:
`experiments/v0.0.11/serial-native-teardown-protocol.json`, canonical SHA256
`27dea5bd160a8a781001576978359d84b605d534da583c797bf13e2d04d1ff13`.
Requires75exact Linux controls then ONE four-worker constructor; formal outcomes
not_run at registration. Local71pass/4Linux-only skips, Ruffpass. Legacy default
and100ms reader hold unchanged. Freeze/public verification precede target launch.
Integration owner: the current Joxing agent, one writer, no delegation. Work in
`/Users/bentaylor/Documents/ChatGPT/JOXING`; RTK for all shell commands.

## Objective, assumptions and completion

Test H17: closing four workers one at a time may avoid overlapping address-space
retirement and meet the existing100ms reservation hold. The third H15packet
faulted at122.75ms; the one-worker constructor previously passed at70.99ms.
This motivates a test; it does not establish contention as the cause.

Deliver a named opt-in transport-close implementation, meaningful source/fault
controls, then ONE fresh four-worker native constructor/close attempt. Every
worker must retain700muscles and exact initial/close state, consume exactly200
preparation steps, and run0rollout/actor/inference/update operations. All terminal
proofs and nonzero charges that occur must release within100ms; no pending
retirement, resource/ownership/source/counter fault or live owned process remains.
Require an observed native retirement path, not a vacuous absence of records.
This qualifies only this constructor configuration if it passes, not trained
workers, learning parity, balance, boxing or throughput.

No model/rules/reward/observation/reset/RNG/optimizer edits, kinematic assistance,
default change, cap increase, private-RSS/PSS substitution, old-result repair or
learning sweep. Preserve three invalid reference attempts/nine not_run slots.
Serial shutdown is a declared new lifecycle variant; a later parity result must
identify that variant on both sides and cannot silently qualify original close.

## Read first, in order

1. `/Users/bentaylor/.codex/RTK.md`, AGENTS.md, README.md, docs/GAME.md,
   docs/SPEC_REVIEW.md, docs/MISSION.md, docs/STATUS.md, docs/IMPLEMENTATION.md.
2. research/README.md, H15.md, H16.md, H17.md, TEMPLATE.md;
   docs/NATIVE_RETIREMENT_RESULTS.md and its three native failure receipts.
3. docs/EXIT_AWARE_RSS_READER_MISSION.md, docs/DEFERRED_PARITY_EXECUTION_PACKET.md,
   docs/DEFERRED_EVALUATION_ALLOCATION_MISSION.md; experiments/v0.0.11/README.md.
4. src/joxing/native_teardown.py, deferred_trial.py (`VectorOwner`),
   deferred_qualification.py (`instrumented_worker`, `wait_ready`, trace schema),
   physics_credit.py; scripts/qualify_deferred_trial.py and terminal_rss_reader.py.
5. tests/test_native_teardown.py, test_deferred_trial.py,
   test_deferred_qualification.py, test_terminal_rss_reader.py and watchdog tests;
   existing native-teardown-v3-protocol.json and frozen runtime lock.
6. Read-only inspect the installed pinned SB3 `SubprocVecEnv.close` and worker
   close branch, record file/hash. Last measured file SHA256
   `e3acbeb465d5ca1288a7b9fa812eeacb9b03e3c52e757a1130b5d57a04baf942`.

## Ownership and likely files

Create `src/joxing/serial_retirement.py`, `tests/test_serial_retirement.py`,
`src/joxing/serial_native_teardown.py`,
`experiments/v0.0.11/serial-native-teardown-protocol.json` and small receipts.
Extend scripts/qualify_deferred_trial.py through an explicit new schema/mode;
preserve existing packet behavior. Reuse instrumentation/ledger/guards rather
than duplicate physics. Update research/H17.md/index, README, docs/STATUS.md,
MISSION.md, IMPLEMENTATION.md and experiments/v0.0.11/README.md at each conclusion.

Forbidden: third_party, old protocols/roots, native XML/muscle data, referee/scene
schemas, host jobs/services/drivers and executing source. Generated outputs stay
in runs/artifacts. Do not patch installed SB3 on disk. No simultaneous source
writer or experiment; inspect every actual owned PID and shared capacity first.

## Implementation sequence and hard parts

1. Add a local adapter factory, e.g. `serial_vec_class(base, receipt_path,
   close_deadline_s)`, returning a subclass of the pinned SB3 class. Keep module
   import stdlib-only; create/import the base only after independent guard ready.
   The adapter changes `close()` only. It must not intercept reset/step or write
   poses. Use this class only in the new constructor runner at first.
2. Bind all worker PID/start identities before close. Preserve SB3's already-
   closed and pending-step behavior. If `waiting`, bounded `poll`/`recv` drains
   each outstanding response before shutdown; no new simulation command. For
   each paired remote/process, send exactly one original `('close',None)`, wait
   for verified process exit/join, then proceed to the next. Never send to worker
   i+1 while worker i remains alive. Record ordered start/exit events, PID/start,
   exit code, elapsed times and first error. Close connections after their owned
   worker exits. Mark closed only after all are verified. Use one overall close
   deadline, e.g.2s, within the preregistered slot bound; this is NOT an increase
   in the reader's100ms hold. Outer watchdog stays independent and ready.
3. Fault edges: dead/reused/missing worker, mismatched endpoint/process count,
   pending-step failure, broken pipe, nonzero exit, join timeout, repeat close,
   partially constructed vector. Fail explicitly. `VectorOwner` may clean up
   owned live processes on a fault; preserve the originating error and mark the
   native attempt invalid even if cleanup succeeds. Cleanup may retire several
   processes concurrently on failure; never report that as a qualified serial
   sequence. No signals to foreign/reused PID identities, unbounded join/recv,
   suppression of live-missing-RSS or reader-policy edits.
4. Fake controls assert exact send/join/event order across four workers,
   waiting drain, no reset/step, repeat close idempotence and all fault edges.
   Add a Linux synthetic process integration with positive RSS charges, owned
   cleanup and foreign-process survival. Tests must fail if implementation
   broadcasts all close requests before joining. Retain all initial development
   failures and commands; do not qualify by fake results alone.
5. New runner mirrors native_teardown.run but creates the original FOUR factories
   with worker IDs0..3/seed1097/reward1 and scenario30000 constructors. Use
   persistent guards for each worker with correct role IDs. Query all four
   specification/snapshot/counts, no `vec.reset`/`vec.step` after construction.
   Verify actual model/action/observation/normalizer identity, all700outputs,
   constructor reset trace per worker and exact close-state digest. Total ledger
   exactly800preparation/0rollout, no unknown/pending reservation or actor files.
   Observe actual retirement charges; no sleep/retry-until-pass sampling trick.
6. Extend the supervisor with a new explicit schema/module branch and row
   denominator, no implicit selector/default changes. Add source tests proving
   old packet commands/defaults and legacy function AST unchanged. Freeze source,
   runtime, test IDs, protocol, input hashes and commands publicly before formal
   native outcomes. Use a fresh dedicated root, NEVER mutate/reuse failed roots.

## Registration, limits and validation

Proposed one packet: Linux source/fault tests first, then one four-worker attempt,
seed1097, original dt.001/control.01/preparation.2/horizon20. Keep7GiB planned/
8GiB hard aggregate observed-RSS-plus-charge,100ms gap/probe/retiring hold,
16owned processes/four simulation workers/one thread. Proposed packet limits:
100swhole,50sactive constructor slot,800physics,12MiBfixtures and1MiBreceipts,
first limit. Register exact limits/test IDs after implementation and before run.
No retry on failure. Include controls/deployment receipts in storage budgets.

Carry all six prior packet costs:15458physics,179.25451156683266s,
31750669fixture/1574621receipt bytes. Original240k/600s/1GiB/4MiB envelope leaves
224542steps/420.74548843316734s/1041991155fixture/2619683receipt bytes. This test
consumes that remainder. No reset/enlargement after failure; recalculate actual
remaining credit before any future four-slot worst-case221760step packet.

Local development commands, once files exist (use repo's installed Python env):

```sh
rtk proxy .venv/bin/python -m pytest -q tests/test_serial_retirement.py tests/test_native_teardown.py tests/test_deferred_trial.py tests/test_deferred_qualification.py tests/test_terminal_rss_reader.py
rtk proxy .venv/bin/ruff check src/joxing/serial_retirement.py src/joxing/serial_native_teardown.py scripts/qualify_deferred_trial.py tests/test_serial_retirement.py
```

If local `.venv/bin/ruff` is absent use existing repo Ruff command; do not install
or edit dependencies merely for docs. Linux run uses the locked
`experiments/v0.0.11/runtime/.venv/bin/python`, exact collected test IDs, then the
ready-guarded supervisor. Record full actual commands, versions/seeds, source/
config/model/actor(not_applicable)/normalizer hashes, caps, PID/start ownership,
events/progress and every attempted/valid/invalid/not_run row. Before launch
require no owned writer,>=16GiB host MemAvailable,>=10GiB free disk and capacity
check; do not interfere with unrelated jobs or wait for a lucky repeat outcome.

## Audit, stop interpretation and handoff

Independently audit source/history/vendor/input hashes and full JUnit denominator,
all800ledger steps/sequence-chain, all four initial/close traces, ordered serial
events against actual worker identity/exit evidence, watcher charge arithmetic/
proofs/hold/gaps and all owned exits. No actor/inference/optimizer/extra physics
may occur. Match every raw name/size/hash on transfer; Git gets only small text
summaries, not arrays/checkpoints/media. Include failure evidence unedited.

Any100ms/resource/identity/equality/counter/source fault stops; remaining actions
not_run. Constructor success cannot prove trained-worker retirement or causality
versus broadcast. If constructor fails, do not launch learning. Next mission must
address the measured fault, not widen100ms or repeat unchanged RL.

If constructor passes and independent audit agrees, register a FRESH H15four-slot
packet with this explicitly named close transport on both arms; preserve every
numeric trace/RNG/replay gate, all prior invalid rows and cumulative budget. Only
after exact learning parity may original concurrent-eight-world/full-source
qualification proceed with a separately declared adoption decision. H10physical
balance, real strike/referee/replay and frozen-opponent training still block
competitive self-play and claims of100000valid games. Publish outcome/docs/next
action to public main; tag only an actually completed micro-version. Leave a
bounded task or a clear next worker mission, never an idle ambiguous state.
