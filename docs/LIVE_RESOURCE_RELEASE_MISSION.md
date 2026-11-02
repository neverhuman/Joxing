# Next: measure closed-world resource release before process exit

Status: CONCLUDED one-worker native lifetime/resource prerequisite QUALIFIED.
93Linux controls +ONE700muscle/200prep constructor pass; exact state, dead native
weakrefs/live RSS drop90.66MB, actual reservation96.47ms. Send→join797.80ms leaves
649.66ms after release, so no demonstrated four-worker latency solution. Measured
tail motivates [H19zero-physics mission](PROCESS_EXIT_BACKEND_MISSION.md) before
more native work. [Results/faults/audit](LIVE_RESOURCE_RELEASE_RESULTS.md).

Historical preregistration: wrapper/runner and93case protocol implemented; target NOT_RUN atfreeze.
Local89pass/4Linux-only skips/Ruffpass; Linux toy weakref support check passed
with0physics. Packet experiments/v0.0.11/live-resource-release-protocol.json,
canonical SHA94750277896099fff234bdfbc3f34455e3d027bdca7f59a72c0b4a3edbd173b5.
Native prerequisite still requires actual positive observed retirement/release;
an unobserved fast terminal path is recorded as incomplete, never a gate pass.
80swhole/35sslot/200physics/oneworker/8MiBfixtures/768KiBreceipts. Original
carry includes a conservative0.940936s whole-command charge for the toy probe.
Historical worker-ready plan below. H18 follows the
failed H17constructor, not a new boxing version. Integration owner current agent,
single writer/no agents. Workspace `/Users/bentaylor/Documents/ChatGPT/JOXING`.

## Objective, assumptions, boundaries and completion

H17 first-two native workers took0.85/0.92s from close send to exit; the2s global
serial deadline expired on worker2. Forced cleanup later caused a105.9ms memory
reservation fault. The normal serial exits' reservations were74/76ms. Thus simply
widening the100ms hold would not solve the first fault. Source inspection shows
StanceEnv owns model/data through several aliases and inherits Gym's no-op close.
Hypothesis: dropping the ENTIRE closed environment graph while the worker is
still live may reduce its later teardown cost. No measured benefit yet.

Deliver a named opt-in release-after-close wrapper and meaningful lifetime/fault
controls, then preregister ONE original native constructor (200prep) with persisted
pre-close receipt, child release timing/identity/RSS and parent exit timing. Preserve
all700muscles, exact pre-release state and all existing100ms/8GiB/2s close gates.
Require actual graph-death evidence and readable before/after live RSS; report
numerical change without inventing a speedup or a threshold after outcomes.
One-worker success alone does not qualify four workers, trained workers, parity,
balance, boxing or throughput. A four-worker successor needs its own public freeze.

Non-goals: no default/source-model/rules/reset/step/RNG/learning changes; no explicit
pose writes, reduced muscle counts, fast `os._exit`, allocator tuning/malloc_trim,
host service/kernel/driver changes, cap relaxation or unchanged RL repeat. Do not
claim serial-close H17 passes. Preserve its2s failure and later cleanup hold fault.
Old failed attempts/protocols/roots and third_party are read-only.

## Read first and ownership

1. AGENTS.md, `/Users/bentaylor/.codex/RTK.md`, README.md, docs/GAME.md,
   SPEC_REVIEW.md, MISSION.md, STATUS.md, IMPLEMENTATION.md and current v11book.
2. research/README.md/H15.md/H16.md/H17.md/H18.md/TEMPLATE.md;
   docs/SERIAL_RETIREMENT_RESULTS.md, NATIVE_RETIREMENT_RESULTS.md and their audits.
3. src/joxing/serial_retirement.py, serial_native_teardown.py, native_teardown.py,
   deferred_trial.py (`VectorOwner`), deferred_qualification.py (instrumented
   wrapper close/state journal/physics hooks), physics_credit.py.
4. src/joxing/stance_env.py (`__init__`, reset, snapshot, inherited close),
   balance.py, groups.py, wave.py/TrainingContract/FaultRecorder wrapper graph;
   pinned installed Gym wrapper and SB3 worker close implementation, read-only.
5. scripts/qualify_deferred_trial.py, trial_rss_watchdog.py, terminal_rss_reader.py;
   relevant native/serial/deferred/watchdog tests and existing frozen protocols.

Likely new files: src/joxing/live_resource_release.py (stdlib before readiness),
src/joxing/release_native_teardown.py, tests/test_live_resource_release.py,
experiments/v0.0.11/live-resource-release-protocol.json. Extend supervisor through
an explicit new schema, preserve every old dispatch/default; small receipts under
artifacts/v0.0.11. Generated runs outside Git. Do not edit referee/scene schemas,
installed dependencies, third_party or an executing checkout. No concurrent work.

## Ordered implementation and hard parts

1. Inspect actual ownership first; no live task/writer may overlap. Implement a
   new opt-in wrapper constructed only after guard readiness. Preserve the
   original env/action/observation/reset/step behavior until close. In close:
   snapshot/hash while intact, record PID/start/live RSS and monotonic begin;
   invoke the existing observed env.close so its state/RNG journal is retained;
   only after it returns, remove this outer wrapper's last reference to the
   entire inner wrapper/world graph, then collect unreachable objects. Explicitly
   delete temporary strong references BEFORE `gc.collect`; otherwise the test
   would falsely claim release. Do not individually set native pose/model fields.
2. Before dropping the graph, register weak references to the actual StanceEnv,
   MjData and MjModel. Verify support with a zero-step binding check first; if an
   extension object cannot be weak-referenced, stop the planned native prerequisite
   and document the missing proof rather than invent a freed flag. Do not retain
   bound methods, native array views, audit dictionaries or traceback locals that
   keep the graph alive. Inspect references carefully; saved snapshots must be
   independent copies. Self.env=None is not proof of native resource release.
3. Persist child before/after hashes, wrapper/graph lifetime observations and
   strictly bound live RSS (same PID/start, VmRSS readable, non-PF_EXITING) and
   elapsed collection/close times. Use existing reader/identity rules; no guessed
   zero or metric change. Require native graph weakrefs dead after release; if
   still alive, classify incomplete and stop. RSS can remain high due to allocator
   behavior: report actual before/after and do not equate freed objects to returned
   physical pages. No required positive RSS reduction is silently invented.
4. Mark the released wrapper permanently closed. Reject reset/step/snapshot or
   other world-dependent operations after release; repeated close is idempotent.
   On inner-close/hash/collection/RSS/identity error, write first fault, keep native
   attempt invalid and preserve owned cleanup diagnostics. No forced fast exit
   or bypass of ordinary SB3/multiprocessing cleanup. Wrapper metadata may remain
   as detached scalars; it must not retain a native model/data reference.
5. Fake lifetime tests: one graph shared via multiple aliases, nested wrappers,
   a hidden strong reference preventing release, close exception, weakref support
   failure, identity/RSS-read failure, repeat close and forbidden post-close call.
   Assert last state hash before release and no reset/step/physics; deliberately
   corrupt the final hash and retained-reference condition. Keep existing75case
   H17controls/default tests; do not remove a failed criterion.
6. New native runner initially uses ONE original constructor/seed1097/reward1/
   scenario30000, exactly200prep/0rollout/actor/inference/updates. Reuse guards,
   instrumented worker and PhysicsCredit. BEFORE sending close, atomically write
   `before-close.json` containing actual spec/muscle count/initial snapshot hash,
   worker PID/start and counters. Child before-release hash must equal both that
   initial hash and observed wrapper close hash. Preserve partial stage receipts
   on every failure; H17lost its parent hashes by writing only at final success.
7. Use named serial-close transport for the single worker; preserve2s close bound,
   100ms monitor/probe/retiring hold and observed-RSS-plus-reservation accounting.
   A retirement may become unobserved if release makes exit too brief: classify
   that path explicitly. Mechanistic graph-release evidence can still be recorded,
   but do not claim a new observed terminal-reservation qualification from absence
   of samples. Define exact next-use acceptance in the protocol BEFORE outcomes.

## Preregistration and resource limits

Source/fault/binding controls before native work; resolve exact test IDs and runtime
weakref support, freeze/push/anonymously verify all source/config/runtime/input
hashes before the ONE target attempt. Do not change source while it executes.
Propose80swhole/35sslot/200physics/one simulation worker/one thread/16owned processes,
8MiBfixtures/768KiBreceipts, first limit.7GiB planned/8GiB hard RSS+charge,
100ms gap/probe/hold and2s close remain. Register machine limits prospectively.
Require no owned writer, >=16GiB host available and >=10GiB free disk; inspect
actual load/owned processes, never interfere with unrelated workloads.

Seven-packet carry:16258steps/211.86220482364297s/31955319fixture/1872562receipt
bytes. Original240k/600s/1GiB/4MiB envelope leaves223742steps/388.137795176357s/
1041786505fixture/2321742receipt bytes. New controls/constructor consume that
remainder. A four-worker constructor and221760step worst-case parity must also
fit future remaining credit; no envelope reset after another failure.

Local validation once files exist:

```sh
rtk proxy .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_live_resource_release.py tests/test_serial_retirement.py tests/test_serial_native_teardown.py tests/test_native_teardown.py tests/test_deferred_trial.py
rtk proxy uvx --offline ruff check src/joxing/live_resource_release.py src/joxing/release_native_teardown.py scripts/qualify_deferred_trial.py tests/test_live_resource_release.py
```

Use locked Linux runtime for target tests, exact JUnit denominator, independent
ready guard before native imports. Retain development failures separately.

## Independent proof, conclusion and handoff

Audit every attempted/invalid/not_run row, source/history/vendor/runtime/input/
model/action/observation/normalizer hashes (actor not_applicable), all200credit
steps/chain/no pending work, copied pre-close receipt, exact child/parent hashes,
weakref lifetimes, PID/start continuity, child phase and parent exit chronology,
live RSS observations, all reader limits and every owned process exit. No fresh
physics in the auditor. Transfer all raw names/sizes/hashes; Git gets only compact
text receipts/research notes, no arrays/checkpoints/media. Missing receipt/proof,
100ms/2s/resource/identity/source fault means invalid/incomplete, not a near-pass.

If graph release is actually proved and this one-worker lifecycle meets unchanged
limits, register ONE four-worker constructor with the same explicit new release
variant and2s total close bound, before any learning. If no native graph release
or time benefit is observed, keep that negative and define the next diagnostic
from measured phase timings; do not repeat training or widen deadlines. Only after
four-worker and exact fresh H15paired learning gates may eight-worker/full-source
qualification/adoption proceed. H10balance, legal strike/contact/rules/replay and
frozen-opponent RL remain required for actual boxing. Update README/docs/v11book/
H18/index, push public Git, tag no incomplete v11, and leave a bounded task or clear
next worker mission. No user decision needed for this authorized reversible work.
