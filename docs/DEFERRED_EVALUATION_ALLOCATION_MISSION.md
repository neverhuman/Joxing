# H15 next: defer evaluation allocation, prove exact training parity

Worker-ready packet after the constructor profile, Nov1 Mountain/Nov2 UTC.
Status: planned, not implemented or run. No full-suite or training sweep now.

## Objective and evidence

Remove the two evaluation environments from the original two-learner/eight-world
training phase without changing training, evaluation or fault evidence. Their
constructor RSS deltas measured615,768,064 and607,580,160bytes. This is allocation
attribution, not proof the candidate saves that amount or fits8GiB. See
TRIAL_MEMORY_PROFILE_RESULTS.md and research/H15.md. All prior failures remain.

Non-goals: native balance/boxing, new rewards, fewer workers/muscles, smaller test
settings, new skips, higher caps, GPU migration, defaults or large training runs.
Preserve all700muscles, original A/B configuration, raw rewards, timing and seeds.

## Read first; single owner

1. AGENTS.md, RTK, research/README.md, H10/H14/H15, this mission and profile results.
2. src/joxing/wave.py: make_worker, timed_sac_class, TrainingContract, train_run,
   counters, callback, actor freeze, deferred initial evaluation and fault capture.
3. tests/test_wave_learner.py and test_wave*.py; wave_extension.py: child_trial.
4. stance_env.py: specification/reset/snapshot; balance.py: prepare; provenance.py,
   stance_train.py: FaultRecorder. Review RNG consumption, not just outputs.
5. The two H15 protocols/receipts, scripts/profile_trial_memory.py,
   joxing_process_ownership.py, source_shard_support.py and supervision helpers.

One integration owner, no agents. Likely edits: wave.py (explicit opt-in dispatch),
new src/joxing/deferred_trial.py, tests/test_deferred_trial.py,
scripts/qualify_deferred_trial.py, scripts/trial_rss_watchdog.py, a NEW protocol in
experiments/v0.0.11, research/H15/docs and small receipts. Do not edit third_party,
original fixture assertions, referees/scenes, old protocols, old run roots or any
source while its owned job executes. Shared schema writes have one owner.

## 1. Fast RSS enforcement before more native startup

The profile's requested0.2s samples actually had median0.473/max1.198s gaps;
largest observed RSS jump was3,157,102,592bytes. Final sample was only53,415,936
bytes below8GiB. Slow PSS scans must not be the only resource monitor.

Implement/test a small independent watchdog that reads only bound ancestry and
PID/start RSS/status, nominal20ms polling. Keep PSS/private sampling separate and
infrequent; never substitute PSS for RSS. Use the same safe newborn ancestry repair.
Record actual gaps, missing/live samples, process count and every stop. The
watchdog itself counts inside16processes/one-thread limits and aggregate RSS.
Use a named owned process group and ownership lease; no unrelated process signal.
Test a bounded rapidly allocating subprocess, PID reuse/foreign children, missing
live RSS and cleanup. Register a maximum acceptable monitor gap and conservative
planned stop below8GiB; stop if observation becomes stale. Do not claim a continuous
RSS bound from periodic samples. Read-only inspect owned cgroup capabilities if
useful; do not change host services, drivers or other cgroups.

## 2. Explicit allocation variant, legacy default preserved

Add a validated keyword such as allocation_lifecycle="legacy" to train_run.
An opt-in "deferred-evaluation-v1" dispatches to a separate implementation;
leave the frozen legacy body and default behavior intact. Reject the new variant
unless defer_initial_evaluation/confirmation is explicitly selected. Record
variant/source/config identity in results; existing callers retain legacy.

The candidate follows this ownership order:

```python
# All construction/fault handling is inside an explicit resource owner.
vec = four_original_workers(same_seed_config_and_reward_scale)
specifications = vec.env_method("specification")
assert all(spec == specifications[0] for spec in specifications)
write_specification(specifications[0])  # prove equal to legacy in paired gate
model = original_sac_constructor(vec, unchanged_arguments)
freeze_initial_actor()
learn_original_schedule_and_callback()
freeze_counts_final_actor_and_four_worker_snapshots()
close_training_workers_and_confirm_exit()
env = StanceEnv(training=False)
assert env.specification() == specifications[0]
evaluate_original_initial_and_final_actors_in_original_order()
```

Do not instantiate an extra world just for metadata. Snapshot/final-counter
queries do not reset or step worlds. Close vector worlds BEFORE evaluation
allocation, including when a peer finishes first. Keep initial/final actor
files frozen before initial evaluation, same private reset RNG, transition
ordering, timeouts/bootstrap masks, replay data, optimizer updates and entropy.

Partial vector construction can fail before train_run's existing try/finally.
Own every process as soon as it is launched; do not silently suppress cleanup
errors. Pre-close faults retain all four live last-valid states. Post-close
faults retain all four captured states with origin="training-end", not invented
current states. Preserve the original failure separately if state capture fails.
Ordinary training faults still save fault checkpoints within the registered cap;
constructor-only ProfileExit remains exclusive to diagnostics.

## 3. Small exact-parity experiment, before concurrency

Implement protocol/source and meaningful controls, then public-freeze/anonymous
verify before formal outcomes in a fresh dedicated ~/joxing checkout. Reuse
locked Python3.12.3/NumPy2.4.2/MuJoCo3.8.0/Torch2.10.0cpu/SB32.7.1 runtime.

Four SEQUENTIAL trial slots: legacyA/candidateA/legacyB/candidateB, seed1097,
four world workers each, original extension configuration with steps64 and
learning_starts16, original network/buffer/entropy, evaluation_ids=[67010],
defer_initial_evaluation=True. Expect12actual optimizer updates each. Resolve
configuration from source and bind its hash; no guessed replacement settings.

Instrument both modes identically and outside production behavior. Retain exact
per-worker observation/control/reward/termination/truncation/next-observation
hashes, reset scenario/order, RNG and actor checkpoints at initial/learn-boundary/
final/evaluation boundaries. Compare both entropy arms, initial and final actor
hashes, SAC updates, counts, replay/timeout masks, evaluation starts and complete
outcomes. Predeclare wall time, memory, run directory and source-variant metadata
as nonnumerical comparison exclusions. Do not hide a mismatch by relaxing numeric
tolerance or restoring RNG only in the comparison harness.

Proposed first limits:600s whole,200,000physics steps INCLUDING reset preparation
and every attempted evaluation,8GiB sampled aggregate RSS, at most4simulation
workers/16owned processes/one thread,1GiB raw/4MiB receipts, min16GiB available
RAM/10GiB disk. All first limits apply to the whole packet; not per trial retries.
Count expected20s evaluation horizon in worst-case estimates, not usual early falls.
Preflight output size and the full worst-case physics ledger. These are proposals;
register exact approved-by-scope settings and watchdog margins before launch.
Do not launch if four slots cannot fit the registered first limit.

Add deterministic fault controls for partial construction, peer-stop, post-close
evaluation failure and all-four-state retention. Run these before native parity.
Every attempted/invalid/unconfirmed/not_run trial remains in the denominator.

## 4. Original concurrency and source gate, conditional

Only after exact single-trial parity, register the original two simultaneous
learners/eight training worlds with both unchanged fixture assertions, followed
by the peer-stop/four-state contract. Retain8GiB/16process limits. Count the
Torch-importing pytest parent: the H15 constructor profiler used a26.5MB stdlib
orchestrator, so its RSS cannot qualify the original test fixture.

On memory failure, retain measured roles and define one new change; do not repeat
the same sweep or shrink the original assertions. On success explicitly decide
whether the new lifecycle is adopted for deferred evaluation, then register a
complete fresh-collection/module-isolated source battery with every test and
the exact known strict referee-alias xfail. No extra skips or omitted cases.

## Completion, stop and handoff

Stop on source/config drift, parity mismatch, missing guards/counters/ownership,
stale monitoring, first resource limit or surviving processes. Audit attempts
independently against frozen source and exact raw hashes, including failed slots.
Write compact H15/index/README/STATUS/MISSION/IMPLEMENTATION/experiment-book
updates, commands, runtime/config/model/action/actor/normalizer identities,
PID/start exits, measured allocations, faults and next action. Push public Git;
no v11tag before its actual completion. Keep raw arrays/actors/media outside Git.

Successful source qualification resumes H10 native neutral/arm balance, real held
continuation and signed-contact recording. No competitive self-play or skill
improvement is established by this allocation work.
