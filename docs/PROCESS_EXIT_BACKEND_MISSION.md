# Concluded: third H19packet supports import-only hypothesis

[All attempts/results](PROCESS_EXIT_BACKEND_RESULTS.md). Next [H21ONE-native transfer](NATIVE_CONTEXT_TRANSFER_MISSION.md); no current backend repeat. Historical plan follows.

# Current: H20prerequisite qualifies; fresh H19packet ready

Read [PROCESS_EXIT_BACKEND_PUBLICATION_PACKET.md](PROCESS_EXIT_BACKEND_PUBLICATION_PACKET.md) and new protocol aefeeb4b before execution. Original two invalid attempts retained. Historical plan follows.

# Next: isolate process-exit latency without native physics

Current: TWO packets invalid. Stdlib parent repair succeeds and71Linux controls
pass, but second packet stops during worker import on120.58ms sample gap; max
probe1.71ms. Aggregate2invalid/10not_run, no latency result. Do not repeat until
[H20publication prerequisite](MONITOR_PUBLICATION_MISSION.md). All original gates
and failed rows retained; [both outcomes/audits](PROCESS_EXIT_BACKEND_RESULTS.md).

Historical first repair: FIRST packet invalid on parent provenance-helper imports before worker
creation,1invalid/5not_run,0physics; [results/audit](PROCESS_EXIT_BACKEND_RESULTS.md).
Fresh stdlib-only writer repair registered at process-exit-backend-repair-protocol.json,
canonical f94d81546acc3471dc5e6b06ba72269417b1fa74820cbb426c6eb60d3b3cecf9.
71Linux controls (67local pass/4Linux skips) precede unchanged6slots/seeds/limits/
hypothesis in a fresh root. Preserve original failure, carry costs; no adoption.

Historical first freeze: FINAL prelaunch revision adds raw bound stat/status/maps files for
independent reconstruction;69cases (65local pass/4Linux skips), same slots and
limits. Superseded68case freeze never deployed; no outcomes reused. Exact current
protocol identity is in experiments/v0.0.11/process-exit-backend-protocol.json and
its canonical hash in docs/MISSION.md/deployment receipt. Historical first freeze:
implemented/preregistered, target NOT_RUN atfreeze. Protocol
experiments/v0.0.11/process-exit-backend-protocol.json, canonical
e0e76c36371e66f3ae615de3f8837489bf8f7ca14862c54b6b74f4e3f47339fb.
Registered68case Linux subset: local64pass/4Linux skips; full previous-regression
set114pass/4skip/Ruffpass. Reused supervisor scripts/qualify_deferred_trial.py
through explicit new probe schema (no separate duplicate supervisor). Original
source/defaults and lifetime/physical gates unchanged. Historical plan below.
Integration
owner current agent, no agents/concurrent writers. Workspace
`/Users/bentaylor/Documents/ChatGPT/JOXING`, RTK for shell commands.

## Objective, assumptions, non-goals and completion

H18 proves actual native graph release and the one-worker gate, but0.64966s
remains between release and join. Total0.79780s does not demonstrate a solution
to H17's four-worker2s deadline. Installed Python source exposes different
spawn/forkserver child exit paths. Test whether an import-only equivalent worker
shows a reproducible exit-latency difference, without more200-step constructors
or learning. Mechanism remains unproved; do not call this boxing/HPC speedup.

Deliver local source/fault controls and a preregistered six-slot diagnostic:
three paired fresh-worker probes, seeds1097/1098/1099, one worker at a time.
Each worker loads the SAME pinned Torch/NumPy/MuJoCo/SB3 libraries only after
independent readiness, initializes identical explicit RNG seeds and emits a small
deterministic tensor/array/RNG-state digest. No native model/data, env/actor,
inference, gradient or mj_step. Then receive one close, flush its own receipt,
close its own pipe and return normally. Compare parent close-to-join duration
and child phase times under explicit spawn versus clean forkserver contexts.

Before outcomes register all six rows/order, exact payload/digests/gates and
statistics. Proposed order: spawn1097/forkserver1097, forkserver1098/spawn1098,
spawn1099/forkserver1099; all rows retained, no seed/order selection or retry.
Scientific support requires all six valid, exact paired payload/runtime/seed
contracts, median candidate close time <=0.5*median reference, and no candidate
pair slower than its reference. With n=3 this is a tiny diagnostic, not robust
throughput evidence. Report all actual values and failed/null outcomes unchanged.
Every slot still must meet2s close and100ms guard/resource limits. Protocol must
distinguish a negative latency hypothesis from an invalid mechanics attempt.

No application os._exit/kill-on-success, fork-from-live-Torch learner, custom C
exit hooks, allocator tuning, GPU/CUDA initialization, dependency/service changes,
model/rule/physics/default edits or source adoption. The standard forkserver
internally uses a different exit path; record finalizer/atexit behavior rather
than disguising that difference. H17failed/H18qualified scopes stay unchanged.

## Read first and ownership boundaries

1. AGENTS.md, `/Users/bentaylor/.codex/RTK.md`, README/GAME/SPEC_REVIEW,
   MISSION/STATUS/IMPLEMENTATION, current v11mission/experiment book.
2. research index/H15/H16/H17/H18/H19/TEMPLATE; docs/LIVE_RESOURCE_RELEASE_RESULTS.md,
   SERIAL_RETIREMENT_RESULTS.md and all linked faults/chronology/audits.
3. src/joxing/live_resource_release.py, release_native_teardown.py,
   serial_retirement.py, deferred_qualification.py (`wait_ready`), physics_credit.py;
   scripts/trial_rss_watchdog.py, terminal_rss_reader.py,
   joxing_process_ownership.py and supervisor readiness/source checks.
4. Actual locked runtime `/usr/lib/python3.12/multiprocessing/{spawn,forkserver,process}.py`,
   hashes in H18post-observation; runtime uv.lock/pyproject. Read
   https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods .
5. Relevant existing source/watchdog/ownership tests and H18protocol. Do not change
   old packet roots, protocols, third_party or installed Python/library files.

Likely create `src/joxing/process_exit_probe.py` (stdlib-only top level),
`scripts/qualify_process_exit_probe.py`, `tests/test_process_exit_probe.py`,
`experiments/v0.0.11/process-exit-backend-protocol.json`, small receipts/docs.
Reuse guard/source/protocol/JUnit/ownership helpers; do not modify native physics
or start-method defaults in wave.py, deferred_trial.py, referees or scene schemas.
Generated outputs in runs/artifacts only. No overlapping root writer.

## Ordered implementation and hard parts

1. Implement a top-level picklable worker function. Parent/probe module remains
   stdlib-only; no heavy parent imports before creation of the clean server.
   Set one-thread env before launching any context/server/worker. Use
   `multiprocessing.get_context(method)` and create Pipe/Process from that SAME
   context; never mix fork/spawn locks or mutate the global default.
2. For forkserver, set explicit EMPTY server preload before its first start in
   that fresh probe process. Do not preload Torch, NumPy, MuJoCo, __main__ or Joxing
   modules with heavy imports. Child waits on its control endpoint BEFORE heavy
   imports. Bind server/child/tracker PID/start and full ancestry, verify server
   Threads=1 and no heavy-library mappings before releasing the worker. Unknown
   or threaded/preloaded server means invalid/stop, not an excuse to skip proof.
   Do not reuse another task's server or fork the learner's initialized state.
3. Each cohort runs in its own fresh stdlib root process so no server or library
   state carries between methods/seeds. The supervisor/watchdog is ready before
   any heavy worker import. Record IPC/ownership readiness events, server proof,
   worker import begin/end, digest, close receive/return, parent send/join and exit
   code with monotonic times. Persist planned/actual rows before imports.
4. Payload: explicit seed then tiny fixed CPU tensor/array and RNG-state digest;
   no gradients/model/actor/inference/env construction. Define exact bytes/dtypes/
   shapes and version contract in source/protocol. Register counters/hooks that
   reject mj_step, actor construction/inference and optimizer updates. Do not
   describe a deterministic tensor operation as reinforcement learning.
5. Record managed multiprocessing Finalize and Python atexit markers separately.
   Their absence/presence may differ by method and must be reported. Do not
   promote a backend on speed if its required cleanup receipt is missing. Explicit
   child phase files/pipe closure and verified worker exit are mandatory in both;
   all native/application cleanup would later need to complete BEFORE worker
   return. No change to Python's exit functions or application forced exit.
6. Use bounded poll/recv and the existing serial-close logic/identity checks for
   ONE worker.2s from close send to joined exit, no unbounded send/join. Handle
   parent/worker readiness timeout, missing/reused identity, broken pipe, wrong
   payload/seed, nonzero exit and server/worker/tracker cleanup. On faults stop
   the packet, preserve all partial rows and remaining not_run; cleanup only
   verified owned processes, not shared services or arbitrary matching PIDs.
7. Fake/source tests: exact preregistered order/denominator; clean-server proof
   required before worker release; PID reuse/foreign rejection; mixed contexts;
   import-before-ready and heavy preload denied; corrupted payload/finalizer
   receipt; timeout/broken pipe/nonzero exit; first-fault preservation and no
   duplicate run. No native700muscle tests or actor construction in these controls.

## Preregistration, limits and commands

Before target outcomes, freeze/push/anonymously verify exact source/runtime/
installed Python exit-source hashes, control test IDs, protocol and commands.
Use dedicated fresh `~/joxing` checkout; inspect actual owned processes, capacity,
faults and Git. Require no other owned writer, >=16GiB available/>=10GiB disk,
same7GiBplanned/8GiBhard aggregate observed RSS+retirement reservations,
16owned processes/one payload worker/one thread and100ms gap/probe/hold.
Forkserver/tracker/root/watcher/supervisor count towards process/memory caps.

Proposed90swhole/12sslot/0physics/4MiBfixtures/768KiBreceipts, first limit. Resolve
exact source-test and output bounds before registration. Six fresh probes only;
no unchanged repeat to get a desired median. Record all positive native/terminal
reservations actually observed; if absent label not_observed, never fabricate
zero. This import-only diagnostic cannot qualify a native terminal path; H18's
one native observed path remains the only new native prerequisite here.

Carry eight packets plus toy development charge:16458steps/245.73279180630385s/
32174082fixture/2172741receipt bytes. Original remainder223542steps/
354.26720819369615s/1041567742fixture/2021563receipt bytes. All new controls/probes
consume the remainder. Native successors/four-slot221760step bound must fit
actual remaining credit; no envelope reset or cap relaxation.

Local development once files exist:

```sh
rtk proxy .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_process_exit_probe.py tests/test_live_resource_release.py tests/test_serial_retirement.py
rtk proxy uvx --offline ruff check src/joxing/process_exit_probe.py scripts/qualify_process_exit_probe.py tests/test_process_exit_probe.py
```

Use locked Linux Python for target tests/probes, retain exact versions/seeds/
source/config/normalizer(not_applicable)/actor(not_applicable) identities,
commands/PID-starts/progress and all attempted/valid/invalid/not_run rows.

## Independent audit and next action

Independently reconstruct every paired timing/digest/seed from raw phase files,
not just supervisor verdicts. Verify source/history/vendor/runtime/exit-file
hashes, complete JUnit denominator, clean-server thread/maps/ancestry proof,
context-specific IPC, observed/held RSS arithmetic, resource bounds, all worker/
server/tracker/root identities exited and no writer remains. Check physics and
learning counters stay0. Transfer all raw names/sizes/hashes; Git gets small
text summaries/audits/research notes, no media/checkpoints/large tensors.

If support fails or any process/ownership/cleanup gate fails, retain negative
evidence and define the next diagnostic from the measured phases; do not run
native parity or training. If the tiny controlled hypothesis is supported,
register ONE native H18resource-release constructor with the explicit new context,
same700muscles/200prep/state/cleanup/2s/100ms gates. Only then consider four-worker
constructor and exact fresh H15spawn-reference/forkserver-candidate trajectory/
RNG/critic/optimizer/replay/evaluation parity, followed by original concurrency/
full-source gates. Neither import latency nor start-method parity proves motor
competence or boxing. No default adoption until all appropriate proofs pass.
Update README/docs/v11book/H19/index; push public Git, tag no incomplete v11 and
leave a bounded task or clear worker-ready mission.
