# Next worker: four native allocation-parity trials

Current H15 execution handoff, prospective. Native parity harness/protocol are NOT
implemented or run. First read DEFERRED_EVALUATION_ALLOCATION_MISSION.md. Its
watchdog and candidate implementation prerequisites now exist; consult latest
STATUS/receipts for Linux source-control conclusion. Legacy remains default.

Work `/Users/bentaylor/Documents/ChatGPT/JOXING`; prefix shell commands with RTK.

## Objective, read order and scope

Prove `deferred-evaluation-v1` preserves the original four-world SAC fixture for
both entropy arms while changing resident-environment lifetime. Four sequential
slots: legacyA, candidateA, legacyB, candidateB. Same seed1097, original extension
config resolved through extension_config, steps64, learning_starts16, scenario
67010, deferred initial evaluation. Require64transitions/12actual updates each.
No concurrency/full-source promotion or boxing claim in this first packet.

Read AGENTS/RTK, research index/H15, MISSION/STATUS, the allocation mission/results;
then wave.train_run, deferred_trial.train_deferred/VectorOwner, TrainingContract,
StanceEnv.reset/step/specification/snapshot, BalanceModel.prepare and the frozen
source-control tests. Read trial_rss_watchdog.py and its independent qualification
receipts, then ownership/supervisor helpers and locked runtime. No agents.

One owner. Likely new files: src/joxing/deferred_qualification.py,
src/joxing/physics_credit.py, scripts/qualify_deferred_trial.py,
tests/test_deferred_qualification.py, tests/test_physics_credit.py,
experiments/v0.0.11/deferred-native-parity-protocol.json, small docs/receipts.
Avoid changing qualified watcher/candidate code unless a specific defect is found;
record a new source variant if needed. No third_party, referee/scene, original
fixture assertions, old protocols/roots, or executing-source edits.

## 1. Exact prospective budget preflight

Default episode horizon is20s, not usual early-fall duration. Four trials can use
160,000evaluation steps +2,560training steps. A deliberately pessimistic bound
allows64new training reset scenarios/trial, four initial worker preparations and
evaluation construction/scenario preparation: at most218,560total steps over all
four trials. Recompute this from current source; do not copy a guessed bound.

The previous mission's PROPOSED200,000-step envelope was not a registered native
protocol and does not cover that worst case. Before any outcome, register a
240,000-step TOTAL packet envelope (not per trial),600s whole,8GiB aggregate RSS,
four simulation workers at a time/16owned processes/one thread,1GiB raw/4MiB
receipts,16GiB available RAM/10GiB disk. This uses existing authorization for
larger training budgets and stays below the heartbeat's2M/1800s ceiling. Preserve
the earlier proposal/history; scientific parity/memory gates are unchanged.
No post-outcome budget increase or repeated failed slot.

Implement an atomic file-backed credit ledger with a separate stable flock inode.
Reserve before work, commit exact use and refund unused credit. Interrupted work
keeps its reservation with unknown actual use. Never count a killed constructor
as zero. Include every failed/invalid/not_run slot and evaluation in the ledger.
Instrument BOTH modes identically, in diagnostic processes only:

- Wrap BalanceModel.prepare: reserve its configured preparation step count before
  original call; commit that exact count on success; interrupted amount unknown.
- Wrap StanceEnv.step: reserve one configured control interval before original
  call; commit delta physics_total on success. Record partial/unknown on failure.
- No changed dt, horizon, muscles, reset scenarios, reward, actions or wall cap.

Tests must prove cross-process reservations never exceed cap, missing/abandoned
reservations remain charged, refunds cannot invent credit, and foreign/stale
writers fail closed. Preserve source/config/hash identity with the ledger.

## 2. Fresh trial process and independent guard

The orchestrator and trial module must import only stdlib before readiness.
For EACH slot create a fresh Python process with start_new_session=True; it waits
at a file barrier before importing Torch or constructing native worlds. Record
PID/start/command immediately. Launch the qualified watcher as an independent
process, counting itself and the orchestrator plus the entire trial tree.

Start importing/allocating only after the watcher's first valid sample creates
its `.ready` file. Nominal20ms/max100ms gap/probe; planned RSS stop7GiB, hard8GiB,
16owned processes. Watcher resource/staleness/wall stop invalidates that slot,
leaves later slots not_run, and stops the whole packet. No slow PSS scans in its
loop, no continuous-memory guarantee. Require all recorded exits before next slot.
Bind every launcher PID/start before allowing work, including test helpers.

Reuse original locked Python3.12.3/NumPy2.4.2/MuJoCo3.8.0/Torch2.10.0cpu/SB32.7.1.
Inspect current shared-host jobs/capacity, lease output ownership, verify public
freeze and complete history/vendor pins/source manifest. New dedicated ~/joxing
root only. Setup/tests/credits/cleanup consume the registered whole limit.

## 3. Identical trace instrumentation, no learner perturbation

Install process-local wrappers in BOTH wave and deferred_trial namespaces; those
modules hold imported aliases. Worker factories must be top-level/picklable and
import afresh under spawn. Preserve every original constructor argument and return
value. Do not reuse inherited parent monkeypatches as a worker instrumentation
mechanism. Tests should prove fresh-child instrumentation is actually installed.

Wrap the original Monitor/TrainingContract environment. Record small hash rows at
construction/reset/step/close, never substitute observations/actions:

- Original spec/model/action/observation hashes and700muscles; native state and
  the actual private reset RNG/scenario sequence.
- Exact observation, action, next observation, raw/scaled reward, terminated,
  truncated and timeout/bootstrap masks for all64transitions/four workers.
- Native integration/application snapshot and private reset RNG at training end.
  The legacy vector closes after evaluation; idle worker state must still match.

Hash arrays with dtype/shape and contiguous exact bytes, not rounded decimals.
Explicitly exclude only known wall metadata, e.g. Monitor episode.t; preserve
simulated elapsed time and every semantic info field. Refuse nonfinite data.
Store raw traces/actors outside Git; compact digest rows and manifests in Git.

Instrument SAC by subclassing the original timed_sac_class without changing its
methods' inputs, RNG or update order. Record initial/prelearn/postlearn and each
actual optimizer-update boundary: actor/critic/optimizer/entropy state hashes,
Python/NumPy/TorchCPU RNG hashes and update count. No RNG seeding/restoration by
the observer. For replay compare only occupied slots (pos/full/n_envs respected),
including observations/next observations/actions/rewards/dones/timeouts. Hash
uninitialized unused buffer memory neither for proof nor storage.

Wrap evaluation to bind original initial/final actor IDs, starts, complete rows,
RNG before/after and final states. Preserve initial actor freeze before learning,
final actor freeze before deferred initial evaluation, and original evaluation
order. Keep model/actor/normalizer identity explicit; this task uses no learned
normalizer. Do not consume training RNG while measuring it.

## 4. Registration, tests and actual run

Controls before native launch: canonical array/tensor/state hashing; exclusions
cannot drop physical info; fresh-spawn instrumentation; atomic credit races;
readiness before imports; watcher fault leaves all later slots not_run; malformed
or missing traces cannot compare equal; numerical/actor/RNG divergence fails.
Use bounded fakes/subprocesses, then a fresh Linux control gate.

Freeze source/protocol/actual collected controls publicly before native outcomes.
Run one bounded packet, no retries. Require BOTH arms' exact semantic trace/state/
RNG/replay/evaluation/actor equality,64transitions and12actual updates, legitimate
changed final actor, all attempts valid, all700muscles, no resource/ownership fault.
Declare only timing/resource/variant/source/run-path fields as comparison
exclusions in the protocol before launch. Source/runtime stay identical otherwise.
Report per-slot peak sampled RSS, actual guard gaps, whole time, charged/exact/
unknown physics, all outcomes and all raw hashes. Lower solo peak RSS is not an
eight-worker concurrency proof or a learned skill improvement.

## After conclusion

On mismatch/fault stop and publish the first divergence plus all retained rows;
do not loosen equality, drop a slot or repeat unchanged. On success register a
separate two-learner/eight-world diagnostic with the original fixture assertions
and peer-stop/four-state contract. Keep original pytest-parent overhead in view.
Before adoption, that diagnostic may explicitly wrap child_trial to pass the new
keyword in each spawned trial; never assume parent monkeypatches propagate.

Only after both parity and concurrency qualification, explicitly decide whether
child_trial adopts the qualified lifecycle for deferred evaluation. train_run's
legacy default/screen path stays preserved. Then run the unchanged original
fixture and full fresh-collection source gate; existing assertions are not edited.
This avoids mistaking an opt-in-only diagnostic for integration of the full gate.

Update H15/index, README/STATUS/MISSION/IMPLEMENTATION/experiment book, independent
audit and all attempt receipts; push public Git, no v11tag yet. Leave a bounded
owned task or a clear next mission. Source success routes back to H10 native
balance/held continuation/signed-contact bout recordings, before self-play.
