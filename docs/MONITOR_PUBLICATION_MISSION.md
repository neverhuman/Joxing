# Next: qualify watchdog publication without blocking its sampling loop

Status: H20 worker-ready plan, NOT implemented/preregistered/run. Single current
agent owns integration; no delegation/concurrent writers. Workspace
`/Users/bentaylor/Documents/ChatGPT/JOXING`; use RTK for shell commands.

## Objective, assumptions and completion

H19's second packet fixes the parent import error and passes71Linux controls,
then stops during the first worker import:120.58ms sample gap,1.71ms maximum
probe,379.72MB sampled peak. Source performs synchronous JSON publication in the
sampler every250ms; a publication point precedes the large gap. The report lacks
write/sleep timing, so storage/scheduling attribution is NOT established.

Instrument those phases, then qualify an explicit private-tmpfs report variant
before another six-slot backend comparison. Keep original disk mode/default,
100ms sample/probe/retirement hold, RSS+charge metric and first-limit behavior.
Do not hide time by resetting sample clocks, averaging away gaps or suppressing
stale_monitor. The measured fault remains invalid. A memory-backed file may also
stall under memory pressure; same gates must reject it if that happens.

Deliver source/fault/storage controls, one bounded Linux publication qualification
packet and ONE guarded H19spawn1097 import-only smoke after controls. No native
model/data/physics/actor/inference/gradient/update. Evidence must show sampled
work stays bounded, delayed durable storage does not block active sampling, full
report copies match by bytes/hash, and all owned processes/private IPC resources
are accounted and cleaned up. Success qualifies only this publication path and
one smoke; it is not a backend, native terminal, balance or training result.

No services/drivers/kernel/mount/fstab changes, global affinity/priority changes,
thread increase, forced fast exit, unrelated-job interference, physics/model/
rules/default edits or cap relaxation. No unchanged H19repeat while unresolved.

## Read first and write boundaries

1. AGENTS.md, `/Users/bentaylor/.codex/RTK.md`, README/GAME/SPEC_REVIEW,
   MISSION/STATUS/IMPLEMENTATION, current v11mission/experiment book.
2. research index/H15/H16/H17/H18/H19/H20/TEMPLATE, docs/PROCESS_EXIT_BACKEND_RESULTS.md
   and both failed H19audits/actual watch reports. The post-observation is not
   in-gap evidence; do not upgrade its pressure readings to causal proof.
3. scripts/trial_rss_watchdog.py (`monitor`, `save`, first ready marker),
   terminal_rss_reader.py, qualify_deferred_trial.py (`stage`, guard readiness,
   resource checks, final copy/cleanup), joxing_process_ownership.py.
4. src/joxing/process_exit_probe.py (stdlib writer/readiness/fresh root/IPC proof),
   serial_retirement.py, physics_credit.py and runtime lock.
5. tests/test_trial_rss_watchdog.py, test_terminal_rss_reader.py,
   test_process_exit_probe.py and existing protocols/input/source hashes.

Likely edit watchdog timing metadata; create
`scripts/watchdog_publication.py` and `tests/test_watchdog_publication.py`,
`experiments/v0.0.11/watchdog-publication-protocol.json`, a small supervisor mode
or dedicated stdlib qualifier and small receipts. Explicitly opt in via protocol;
do not alter old launchers/defaults or rewrite failed roots/protocols. No writers
to referee/scene/native schemas or third_party. Persisted generated artifacts
belong in runs/artifacts; private RAM files are bounded temporary IPC only.

## Ordered implementation and hard parts

1. Add small in-memory timing fields: serialization wall/CPU, report-write wall,
   readiness-touch wall, requested sleep, observed wake delay, probe wall/CPU and
   own rusage context-switch/page-fault deltas. Retain a fixed-size recent ring
   and maxima. Timing of a completed write must survive into the next/final
   report; measuring only before save repeats the current evidence gap. Keep
   sampling timestamps and stale test identical. Avoid slow proc/smaps scans in
   the hot path or any helper that imports native libraries into the parent.
2. Add an opt-in launcher report scope `private-tmpfs-v1`: create one fresh private
   directory with unpredictable nonce/UID ownership/mode0700 on existing verified
   /dev/shm tmpfs. Do not mount/configure it, enumerate/remove others' objects or
   reuse another job's directory. Actual host has existing tmpfs and ample free
   space in the retained post-observation; recheck at launch. Reject unavailable/
   wrong-filesystem/symlink/ownership/capacity cases BEFORE releasing work, without
   falling back to disk. Canonical persisted outputs remain under the run root.
3. Bound the report and readiness IPC files (existing watchdog max256KiB; propose
   <=1MiB aggregate per packet, one active report scope at a time). Record paths,
   ownership, byte credits and lifecycle. Logical run paths may point through an
   explicitly registered private volatile namespace; do not treat IPC as a Git
   artifact or claim its pages are measured process RSS. Keep RSS unchanged and
   separately bound/report tmpfs storage. No extra writer thread/process needed.
4. Run the SAME watchdog/probe logic with output and ready marker in that scope.
   Worker receives that exact readiness path. It must remain blocked until guard
   readiness; no sample-clock reset or omitted startup gap to manufacture a pass.
   Parent still observes guard liveness and all100ms/first-limit stops. All active
   reports, including stopped/fault/partial states, remain available for salvage.
5. Once task AND guard have exited (or been stopped as invalid), copy the exact
   RAM report/ready files to canonical runs output with original bytes preserved;
   verify size/hash/readback before deleting only the owned IPC directory. Hash
   identity must hold even on fault. Durable copy time remains inside whole-wall
   and storage credits; it may not occur in the active sampler. If copy/drain/
   cleanup fails, retain fault/paths, mark invalid, never substitute an older disk
   report. Keep sampling while any evaluated task is active, including teardown.
6. Tests: missing tmpfs/space; unowned/reused/symlink path rejection; bounded file
   size; exact byte/hash copy plus deliberate corruption; cleanup on partial
   launch and guard fault; readiness path agreement; no parent heavy imports;
   resource caps/default disk mode unchanged. Source tests must catch publication
   accidentally directed to disk again during active sampling.
7. Controlled Linux diagnostic: inject a declared150ms durable-publication delay
   in the test-owned disk sink. Disk-mode control should honestly detect a>100ms
   gap; memory-backed mode must sample the same small owned heap workload within
   100ms and perform the delayed durable copy only AFTER task exit. This proves
   separation under injected delay, not the historical fault's cause. Retain
   expected-stop controls as such; never call them successful native monitoring.
   Verify foreign-process survival, positive observed RSS/reservation arithmetic
   where exercised and all actual owner PID/start exits, with one thread each.
8. Only after controls qualify, run ONE H19spawn1097 import-only smoke using the
   new report scope. Reuse exact payload/readiness/cleanup/source guards,2s close,
   100ms gap/probe/hold, one worker and no native work. New explicit qualification
   schema must not call the three-pair conclusion on this one-row smoke. Retain
   any unobserved retirement as not_observed, not a new native qualification.
   If it fails, stop; use the new phase timing to choose the next action.

## Preregistration, caps and validation

Freeze/push/anonymously verify source/runtime/input hashes, exact test IDs,
controlled-delay semantics, smoke slot and acceptance BEFORE formal outcomes.
Fresh dedicated ~/joxing root, no other owned writer; inspect load/capacity/jobs,
>=16GiB host available/>=10GiB disk and privately bounded tmpfs free space.
Keep16owned processes/one payload worker/one thread,7GiBplanned/8GiBhard observed
RSS+retirement charge,100ms probe/gap/hold,2s close. No actor/model/normalizer (N/A).
Propose60swhole/12ssmoke/0physics/4MiBfixtures/384KiBreceipts plus <=1MiB temporary
IPC, first limit. Resolve actual source-test/output bounds prospectively.

Current ten-packet carry incl toy charge:16458steps/273.079175575369s/
32311887fixture/2714816receipt bytes. Original remainder223542steps/
326.920824424631s/1041429937fixture/1479488receipt bytes. Controls/smoke/new metadata
consume it. Receipt capacity is becoming a constraint: register compact JSON
whitespace for NEW deployment/supervisor receipts if needed, with exact decoded
value/float/hash checks; never drop source hashes, fault/denominator rows or raw
watch evidence, and never rewrite old byte-preserved receipts. If the planned
bound cannot fit, define a zero-physics lossless receipt-storage mission first.
Do not reset/enlarge the envelope after failure or launch an oversized repeat.

Local validation after files exist:

```sh
rtk proxy .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_watchdog_publication.py tests/test_trial_rss_watchdog.py tests/test_terminal_rss_reader.py tests/test_process_exit_probe.py
rtk proxy uvx --offline ruff check scripts/watchdog_publication.py scripts/trial_rss_watchdog.py scripts/qualify_deferred_trial.py tests/test_watchdog_publication.py
```

Use locked Linux runtime for actual controls, preserve development failures and
exact commands/versions/seeds/PID-starts/progress/active IPC byte counts.

## Independent audit and handoff

Reconstruct probe/write/sleep timing from raw phase metadata; identify unexplained
time honestly. Verify private tmpfs mount/ownership/readiness/cap proof, exact
volatile-to-durable byte hashes, complete JUnit and every attempted/valid/invalid/
expected-stop/not_run row. Check all source/history/vendor/runtime/input hashes,
reader RSS/retirement identities/charge arithmetic/100ms thresholds, stop/cleanup
behavior,0physics/actors/updates, all process exits and no active writer or owned
IPC leak. Transfer all raw names/sizes/hashes; commit only small text receipts/
research notes. Do not run physics in the auditor.

On qualification, separately register a fresh H19six-slot packet with the named
publication variant, same seeds/payload/latency criterion/limits and every old
failed row/cost retained. No backend promotion from publication or smoke alone.
If smoke fails or diagnostics implicate scheduling instead, conclude that result
and define a small measured follow-up; no blind repeat or100ms relaxation.
Native H18new-backend200prep/state/cleanup proof, four-worker and exact H15learning/
RNG/replay/evaluation parity, original concurrency/full-source and H10motor/game
gates still follow. Update README/docs/v11book/H20/index, push public Git, tag no
incomplete v11, and leave a bounded task or clear next worker mission.
