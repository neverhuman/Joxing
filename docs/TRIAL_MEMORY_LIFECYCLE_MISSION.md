Profile concluded capacity-limited; measured evidence in TRIAL_MEMORY_PROFILE_RESULTS.md.
Current next packet: DEFERRED_EVALUATION_ALLOCATION_MISSION.md. No active job.
The original profile proposals below are preserved; do not repeat unchanged.

Current execution packet: experiments/v0.0.11/trial-memory-ownership-repair-protocol.json.
Original first ownership stop is audited in research/H15.md. The named repair
adds ancestry binding for post-inventory newborns; model/config/caps unchanged.

# H15: attribute trial memory, then qualify deferred evaluation allocation

Current mission, Nov1,2026 America/Denver. Constructor-only profile is implemented
and registered in experiments/v0.0.11/trial-memory-profile-protocol.json; target not_run. Do not launch another full suite or
training sweep. H14's standalone process-isolation acceptance failed; all prior
failed roots/limits remain unchanged. Native boxing/balance remain unqualified.

## Objective, assumptions and boundaries

Find enough real allocation savings to preserve the ORIGINAL two simultaneous
learners × four world workers inside8GiB. Do not shrink the test, reduce muscles,
replace RSS with PSS to pass, add skips/xfails, or enlarge the cap. Use PSS/private
bytes only to understand allocations. Model/rules/reward/reference/action defaults
remain unchanged until a separately registered, exact-parity candidate qualifies.

Static source fact: wave.train_run constructs `StanceEnv(training=False)` before
four SubprocVecEnv workers. `defer_initial_evaluation` currently defers rollouts,
not that environment's allocation. Two learner trials therefore retain two extra
evaluation environments during training. Worker Monitor/SubprocVecEnv imports also
bring in SB3/Torch; their contribution is not yet measured. Removing evaluation
models is a hypothesis, not a demonstrated saving.

## Read-first and ownership

Work `/Users/bentaylor/Documents/ChatGPT/JOXING`; RTK on every shell command.
Read AGENTS/RTK, research/README.md, H10/H14/H15, SOURCE_PROCESS_ISOLATION_MISSION,
both isolated-source protocols and failed receipts. Then:

1. src/joxing/wave.py: make_worker, timed_sac_class, train_run, counters, exception
   capture and final vec.close; wave_extension.py: child_trial.
2. tests/test_wave_learner.py, especially the two-trial/four-worker fixture and
   peer-fault last_valid_workers assertion; tests/test_wave*.py.
3. stance_env.py: reset, snapshot, specification; stance_train.py: FaultRecorder;
   BalanceModel preparation/reset accounting; canonical_actor_hash.
4. supervise_isolated_source.py, source_shard_support.py, process-ownership helpers
   and runtime lock. All prior source hashes and historical-object dependencies.

One owner, no agents or other-chat messages. Initial profile owns only new
src/joxing/trial_memory.py, scripts/profile_trial_memory.py,
tests/test_trial_memory.py, experiments/v0.0.11/trial-memory-profile-protocol.json,
small receipts/docs. Do not edit wave.py, original tests, third_party or old roots
for this first profile. Do not change unrelated services/drivers/jobs.

## Packet A: instrument original constructors, never train

1. Implement an opt-in `TrialMemoryProfile` wrapper around the exact existing
   constructors and two-trial/four-worker configuration. Bind source/runtime/
   config/seed/model/action/observation hashes. Preserve original initial actor
   seeds and all700muscles in each world. Initial profile uses unmodified code.
2. Emit causal phase events with monotonic time, PID/start/parent, trial owner and
   role: before/after evaluation-env allocation, before/after four-world startup,
   before/after SAC construction, and intercepted learn-ready. No real model.learn,
   actor.predict or optimizer step may execute. Guard them and test those guards.
   Initialization/reset preparation CAN run; record known prep/physics counters
   separately and mark interrupted totals unknown rather than0.
3. Constructor wrappers must call the originals with unchanged args, in a named
   diagnostic variant. A profile SAC subclass can intercept learn at entry, emit
   readiness, wait at a bounded barrier, then raise a dedicated BaseException.
   Catch it in the profile target; original train_run's finally still closes the
   vector worlds. Do not trigger its ordinary fault-checkpoint path accidentally.
   Preserve actual source files; in-process instrumentation is explicitly logged.
4. Parent samples only bound descendants: RSS, PSS, private clean/dirty bytes,
   parent/role, and presence of libtorch mappings. No environment/credential dump.
   Read /proc/*/smaps_rollup and maps for owned PIDs only. Capture per-role memory
   at the emitted allocation boundaries, not only a process lifetime high-water.
   Retain resource-tracker/manager roles separately from simulation workers.
5. New schema `joxing-trial-memory-profile-v1`: config/source/runtime hashes,
   phase journal, role/PID/start identities, sampled RSS/PSS/private bytes,
   prep-counter exactness/uncertainty, attempted/ready/stopped/not_run trials,
   inference/optimizer guard counters, output manifest, caps and next action.
   Raw memory samples JSONL; compact summary in research/H15 and Git.
6. Bound profile before launch: propose60s active sampling, whole180s after locked
   runtime preflight,8GiB aggregate RSS/16processes/one thread,1GiB fixture scratch/
   4MiB receipts,16GiB free RAM/10GiB disk. Register exact first limits and a lower
   planned sampling stop before8GiB; a jump over the hard cap remains a failure.
   Check projected actor/fault output bytes; never save a large unused replay.
   Initial-actor data, if retained, stay outside Git with hashes.
7. Add meaningful tests: fake owned-process trees/PID reuse, exact phase order,
   missing/foreign roles, guard prevents actual learning/inference, resource-stop
   denominator and cleanup. Include a bounded subprocess fixture. Freeze/push
   protocol/source, anonymously verify, inspect fresh host capacity and run once
   in a NEWdedicated ~/joxing checkout with complete superproject history.

Expected outputs: per-role memory table and phase deltas, all raw hashes, exact
source/config/actor identities where constructed, known reset-preparation counts
or explicit partial bounds, and confirmed owned exits. Stop on drift, missing
ownership, unexpected training/inference, resource/fault or insufficient counters.
A capacity-limited initialization profile may be useful evidence; it is never a
full-source or learned-policy win. Do not repeat it unchanged after a stop.

## Packet B, only if measurements support lifecycle savings

Register a named opt-in deferred-evaluation lifecycle. No default flip during
development. Reuse model/task/learner settings and original four-world count.

- Get specification from an actual world after vector construction. Current
  specification contains balance_config/model/action/observation hashes/slices;
  prove it exactly matches the evaluation specification. Do not construct and
  discard an extra physical environment merely to get metadata, or hide its prep.
- For deferred/confirmation evaluation, allocate the evaluation environment only
  after training. Merely delaying it while keeping all world workers alive may
  move the same peak later. Capture counts and all four last-valid worker
  snapshots, close training workers, then allocate evaluation environment.
- Preserve initial actor BEFORE learning and final actor BEFORE deferred initial
  evaluation, exact evaluation scenarios, private RNG, checkpoint hashes, update
  cadence, reward/mask contracts and unassisted physics. Actor.load/global RNG
  consumption and callback/reset ordering need explicit checkpoint comparisons.
- Before-close faults retain live worker states as before. After-close faults
  retain captured states labeled captured-at-training-end, never stale or invented
  as current. Close must cover partial construction, peer-stop and evaluation fault.

First run original/candidate SINGLE-trial paired parity under8GiB, then the two
simultaneous learners with all eight training worlds. Predeclare equal seeds,
64transition fixture/learning_starts16/12update expectation and both entropy arms,
or copy exact current resolved settings rather than guessing. Require exact
initial/final actors, control/observation/reward/termination histories, update and
RNG checkpoints, four-worker fault retention and source/output provenance.
Timings differ; do not treat elapsed time equality as a numerical invariant.

Only after parity and measured memory gates pass, explicitly choose/integrate the
qualified lifecycle for deferred evaluation under a new recorded variant. Preserve
the original screen path and reproduction via frozen commits. Do not weaken or
omit the original eight-worker concurrency assertions to fit memory.

## Final gate and handoff

On success register a complete fresh-collection/module-isolated Linux battery,
same8GiB/16process/thread/storage limits and exact known strict referee alias
xfail. Every test/JUnit identity must remain covered; no hidden skips. Qualify
the harness report-size preflight and shared total wall budget before launching.
On failure publish allocation evidence and the smallest next worker-ready change.

Every packet retains exact commands, seeds, source/config/model/actor/normalizer
hashes, PID/start/resource samples, all attempted/valid/invalid/not_run rows,
faults and independent exit/output audit. Update H15/index, README, STATUS,
MISSION/IMPLEMENTATION and experiment book; push public Git. No v11tag or boxing
promotion. After source qualification, resume H10's separately registered native
neutral/arm balance, real held continuation and full signed-contact recordings.
