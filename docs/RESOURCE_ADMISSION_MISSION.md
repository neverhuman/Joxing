# H23: admit a quiet host, expose runtime import progress, retain full receipts

Status: worker-ready plan; NOT implemented/preregistered/run. Current agent owns
integration, no delegates/concurrent writers. Work in
/Users/bentaylor/Documents/ChatGPT/JOXING, prefix shell commands with RTK.

## Objective, assumptions, non-goals and completion

H22locked setup completes, then its unchanged runtime-stage15s allowance (13s
usable +2s cleanup reserve) fails BEFORE198tests or any native worker. Exact
cause/module unknown; high contemporaneous host load and later I/O/memory
pressure are observations, not historical causality. H23must prevent blindly
starting expensive qualification in known pressure, and make any fresh import
failure attributable. Conditional four-world proof retains H22unchanged native
body/seed/state and ALLoriginal2s/100ms/RSS/physics gates. No timeout increase.

Deliver: read-only prospective admission +deferral receipts, staged import
provenance under SAME15s effective timing, qualified lossless NEWmain/terminal
encoding to fit remaining storage, all old/new source controls, then at most ONE
four-native attempt IFall prerequisites pass. Admission success alone is not a
source/runtime/native qualification or a throughput result. Rejection is DEFERRED/
NOT_RUN, not a failed bout or a successful experiment. Preserve every check.

Complete only if new controls pass; imported version tuple exactly pinned;
full provenance/decoded fields/float bits and encoded hashes validated; FOUR
original700muscle workers pass original state/lifetime/context/retirement/WHOLE
serial-close gates. Any missing evidence/fault stops. Native semantics in
src/joxing/four_native_context.py, worker/release/defaults remain unchanged.
H22prerequisite fault and all earlier invalids stay. No learning, new physics/
model/rules, root assistance/pruning, GPU, application fast exit, unrelated jobs,
services/drivers/affinity/priority/thread change or default promotion/v11tag.

## Read first and write boundaries

Read AGENTS/RTK; README/GAME/SPEC_REVIEW; MISSION/STATUS/IMPLEMENTATION/v11mission/
experiment book; research index/H15–H23/TEMPLATE; FOUR_NATIVE_CONTEXT_RESULTS.md,
NATIVE_CONTEXT_TRANSFER_RESULTS.md, failed H22packet/audit/metadata and source
pressure observation. Then docs/COMPUTE.md/SELF_PLAY_HPC_PLAN.md for existing
host qualification; do not assume another host is numerically interchangeable.
Read scripts/qualify_deferred_trial.py, receipt_encoding.py, deployment helper,
source_shard_support.py, joxing_process_ownership.py, runtime lock/version check,
watchdog_publication.py/trial_rss_watchdog.py/terminal_rss_reader.py and source
four_native_context/native_context/live_resource_release/serial_retirement/credit.

Likely create scripts/host_admission.py, scripts/runtime_import_check.py (stdlib
until requested imports), tests/test_host_admission.py/test_runtime_import_check.py;
extend explicit receipt modes/tests and LOCAL supervisor/deployer hooks, fresh
experiments/v0.0.11/resource-admitted-four-context-protocol.json, receipts/docs.
Do NOTedit native four-worker body, original runtime allowance, serial deadline,
watcher/retirement gates, scene/referee/model/actor/wave/third_party, old roots/
protocols/receipts. No raw arrays/checkpoints/media or credentials/SSHconfig in
Git. No duplicate task/writer; use dedicated immutable ~/joxing checkout/lease.

## Resource and storage dependency

All14prior packets remain charged:16658physics/395.40257320619463s/
35531698fixture/4032589receipt bytes. Original remaining223342steps/
204.59742679380537s/1038210126fixture/161715receipt bytes. H22's224KiBmaximum no
longer fits. Do not delete/reclassify old evidence or increase any failed gate.

Proposed first-limit H23envelope100swhole/50sslot/15sruntime (existing2s cleanup
reserve retained)/800prep/12MiBfixtures/156KiB(159744)receipts/1MiBprivateIPC/
4simulationworkers/thread1/16aggregate owned processes/7GiBplanned/8GiBhard/
100ms gap-probe-hold/WHOLE2sserial close. It FITS remaining ceilings arithmetically;
actual storage projection with headroom is a mandatory launch prerequisite.
If it cannot fit, STOP physical launch and leave measured encoding/budget next
action; no assumption that later H15parity will fit. Original accounting retained.

H22four-world prototype plain compact main~57k plus terminal JSON duplicates
~25k; those duplicates cost too much. Implement NEWnamed encoding only, e.g.
`gzip-main-stdout-bootstrap-v2`: bootstrap exact gzip as H22; main authoritative
result.json.gz (no durable plaintext twin), atomic exact compact-JSON bytes with
encoded/decoded size/SHA/CRC and bounded decode; terminal summary is a small
JSONcapsule containing deterministic gzip+base64 of the ENTIRE old terminal
JSONline/bytes +decoded byte length/SHA. All fields/errors/rows/history remain
recoverable; no value truncation/downcast or field/log deletion. Other stderr/
log lines stay fully retained. Capsule compresses duplicate terminal fields,
not a lossy summary. Default and H22wire modes must remain byte/field compatible.

RAM watcher and uncompressed canonical watch copies/readiness/publication proof
stay unchanged. Source/codec controls must prove old H21and H22saved main+terminal
bytes/types/float.hex()/-0/largeints/bools, corruption/partial writes/limits and
old defaults, then actual Linux newbootstrap/main/terminal sizes before native.
Compressed files and metadata count fully as receipts. No uncharged host
plaintext expansion. Independent audit expands only into ignored LOCAL scratch,
keeps original wire files, verifies raw manifest +encoded/decoded hashes. Live
readers choose report name from explicit protocol mode, no stale fallback file.

## Ordered implementation and hard parts

1. Implement read-only Linux admission parser for /proc/loadavg, /proc/meminfo,
   /proc/pressure/{cpu,io,memory}, actual sched_getaffinity count and known owned
   process identities. Existing >=16GiBavailable/10GiBdisk/no unknown Joxing job
   still apply. Proposed PROSPECTIVE conservative policy: load1<=0.5*affinityCPU;
   PSIcpu some.avg10<=10%, I/O full.avg10<=1%, memory full.avg10<=1%; require TWO
   passing snapshots10sapart, fresh <=2s final sample before launching imports.
   Missing/invalid/stale data defers. These are preventive heuristics, NOTproof
   of H22cause or optimized benchmark selection. Register exact policy before
   any fresh attempt; do not change thresholds after seeing a result.
   Parse PSI against the [primary kernel interface documentation](https://docs.kernel.org/accounting/psi.html):
   averages are percentages over10/60/300s, total is accumulated microseconds;
   some and full describe different stall populations. System CPUfull is undefined
   and reported zero for compatibility; do not use it as healthy-CPU evidence.
   This task only reads pressure files; it does not install/write PSItriggers.
2. Admission runs BEFOREnew checkout/dependency install/fullbootstrap to avoid
   consuming most remaining evidence budget while already busy. A final fresh
   admission check after setup catches changed conditions. On rejection emit
   small DEFERRED/NOT_RUNreceipt with every value/reason/time/hash; no simulation,
   no full qualification/unchanged retry/background loop. Retain bounded admission
   history in packet accounting; stop when its allowance exhausted. Heartbeat
   can recheck later quietly when unchanged; do independent source work meanwhile.
   Never stop or tune unrelated jobs. Read-only snapshots of xbabe2/3/4 can inform
   availability per COMPUTE.md, but another host needs exact pinned runtime/native
   parity qualification before substitution, no automatic migration.
3. Replace only runtime-check subprocess body with explicit named stdlib module
   checking SAMEversions and SAME15s stage. Emit compact append-only phase events
   before/after imports numpy,mujoco,torch,stable_baselines3 and final tuple/assert.
   BoundPID/start/monotonic/processCPU/getrusage(minor/majorfaults/contextswitches)
   and /proc/self/io/status plus concurrent PSI/load snapshots at boundaries.
   Keep unknown/fault/cut-off phase; never call a missing end successful. Values
   include instrumentation costs where applicable; whole-stage wall covers ALL
   logging/reads/imports without resetting clocks. No model/data/env/actor/infer/
   optimizer or mj_step calls. Version hash/normalizer/body checks remain.
4. Stage provenance can locate a slow import for THISfresh run; rolling PSI and
   system-wide deltas include unrelated work, not proof of past cause. On timeout,
   preserve originating stage error, last phase, actual wall/CPU/I/O/probe data,
   all partial files/owned cleanup. Never extend15s/relax100ms to get a pass.
5. Implement optional main/terminal encodings above with small stdlib helper;
   same authoritative result values and decoded bytes. Preserve atomic old valid
   output on encode/IO failure; track encoded/decoded bound, reject corrupt/
   truncated/oversized data. A final capsule includes original entire terminal
   payload, no field omission. No output pump processes/threads/driver changes.
   Do not hide runtime/watcher latencies through compression or altered clocks.
6. Unit controls: PSI percentages/affinity edge cases/missing fields/stale times,
   two-pass spacing, pressure/loadingdeferral/no heavy work, source/mode/default
   preservation, phase boundary/timing/fault/partial event handling, no version
   assertion without all imports, exact float/type/byte codec/corruption/partial
   writes/storage partition, decoder caps/unknown modes. Reuse all198H22controls
   +new cases; no native physics in unit tests. Log all intermediate faults.
7. Produce accurate synthetic receipt projection from REALsaved H21/H22data using
   actual NEWreport shapes/encoding/bootstrap source manifest/control count and
   headroom. Label projection only, never persist fake physical result rows.
   If >159744(or remaining ceiling), no launch. Pin saved input/source/runtime/
   CPython/gzip hashes and exact case list. Freeze source/protocol PUBLICbefore
   outcomes, anonymously verify, admission/freshcapacity, unique dedicated root,
   no duplicate/overlapping writer. Source/runtime/codec/preflight controls ALL
   pass before conditional ONEfour-native row. Retain four-world denominator.
8. Native stage calls unchanged H22body. Independently audit all prelaunch and
   phase metadata, runtime tuple, exact JUnit cases, decoded main/terminal/bootstrap
   fields, four native state/context/lifetime/RSS/finalizer/credit/2s/100ms proofs,
   all encoded/raw bytes/storage costs, complete PID/start exits and no writer.
   Missing actual native retirement remains incomplete, no resampling. Transfer
   ALLraw by names/sizes/SHA; no old source/criteria rewritten.

## Reporting, validation and handoff

Use `.venv/bin/python -m pytest -q -p no:cacheprovider` with affected/new +all
H22test_modules and JUnit under runs/v0.0.11; existing offline uvtool Ruff.
Exact final commands/case IDs, versions/seeds/source/config/model/actor-none/
normalizer hashes, all attempts/deferrals/valid/invalid/not_run, PIDs/progress/
faults/limits and actual costs persist. Fresh source qualification precedes
physical work; local tests alone are not Linux gates. Never claim a version
assertion from sync package output or empty runtime.log.

Update research/H23.md/index, README/MISSION/STATUS/IMPLEMENTATION/book and results
with honest scope, carry all14old costs and retain failed criteria. Push public
Git, no v11tag. If admitted job healthy, verify fresh events/actual processes,
advance independent analysis without editing source. If deferred, preserve clear
next action and stay quiet until state changes. If pass, separately register exact
H15lifecycle learner/RNG/trace/optimizer/eval parity BEFOREH10physical motor/bout/
opponent-aware history self-play; observe remaining budget, not a giant sweep.
