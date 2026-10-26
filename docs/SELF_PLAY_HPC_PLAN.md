# Fast, meaningful muscle-driven self-play

October 26, 2026. This is the execution route requested for thousands, ultimately hundreds of thousands, of games. Read [v8 results](V0.0.8_RESULTS.md), [GAME](GAME.md), [IMPLEMENTATION](IMPLEMENTATION.md) and [state/history audit](STATE_HISTORY_SELF_PLAY_AUDIT.md) first. Current source still has no production two-player learning game. v8's 40 valid episodes all physically failed around five seconds; replay/correctness passed. Faster versions of those episodes are motor experiments, not boxing games.

## Objective and priorities

Produce increasing held-out match strength per wall hour on the complete 1,400-muscle game. Optimize **useful learning**, not a simulator-only FPS number. Keep all muscles, meaningful contact/referee evidence and causal opponent-aware control. The best near-term hardware path is measured CPU physics scaling plus shared batched policy inference; GPU physics is a separate feature/parity project. This is a hypothesis to benchmark, not a promised speedup.

Two prerequisites progress under separate protocols: reliable goal-conditioned motor skills and a faster correct rollout/learner pipeline. The HPC work need not wait for a boxing champion, but cannot certify boxing ability. Do not launch 100,000 broken rounds or train tactics before motor/contact/rules/replay gates pass. Non-goals: a Rust engine rewrite, simplified or assisted bodies passing native gates, biological damage claims, buying compute, changing services/drivers or hiding faults in episode resets.

### Compute refreshed for this request

Read-only inventory after v8 exited: xbabe1 has 128 logical CPUs, about47.1GiB available RAM and idle RTX4090/24GiB; xbabe2 has128 CPUs, about65.7GiB available and idle3090/24GiB; xbabe3 has128 CPUs, about97.6GiB available and idle3090/24GiB but only46.8GiB free disk. xbabe4 has32 logical CPUs/512GiB physical unified memory, low load about4 and656GiB free disk; physical memory is not a measurement of free memory. These are shared-host snapshots, not reservations. See the [receipt](../artifacts/receipts/hpc-inventory-2026-10-26.json).

Start on xbabe1. Keep node-local physics and inference; benchmark before adding LAN transport. Use xbabe2/3 later for independent seeds and frozen evaluation, with node-local actor replicas so every observation does not traverse SSH/LAN. Keep large raw runs off xbabe3. xbabe4 is valuable for native CPU rollout qualification, analysis and replay/rendering; Docker is not required. Reinspect capacity immediately before every launch.

## Ordered delivery

### 1. v0.0.9: profile and batch the actual controller

Follow the [worker-ready mission](V0.0.9_MISSION.md). First measure the real upstream 3,601-input/700-output actor at batch1–128 on identical retained observations. Compare scalar calls with batched calls, including normalization/conversion. Persist numerical parity, repeated outputs, startup, latency distributions, samples/s and memory. This is isolated inference, not game throughput.

Then profile full CPU stepping: native simulation, per-substep contact/posture reads, reference/observation construction, normalization, prediction, snapshot/copy, IPC, disk and resets. Compare one/two/four/eight/sixteen one-thread workers only while capacity permits. Keep at least8GiB host RAM free and20% GPU memory headroom; lower worker count when measured per-process memory would violate that. All attempted and skipped cases stay in the denominator. Do not assume linear scaling from v8's70.33/130.16 control TPS at one/two workers.

Prioritize the measured bottleneck. If inference/transport is minor, do not build a complex actor service merely because GPUs are available. If Python evidence/copy dominates, first use contiguous buffers and lazy serialization. If native physics dominates, scale CPU workers and investigate the exact GPU backend. A proposed performance win needs >=1.2× median end-to-end throughput at equal physics/evidence and zero new validity failures; a negative benchmark is a valid result.

### 2. Shared inference with explicit ownership

One inference process holds the current actor and a bounded cache of frozen opponent bundles. Physics workers hold MuJoCo state, not full SAC critics/optimizer/checkpoints. Export actor-only state only after proving deterministic output, decoder amplitude and normalization match the loaded checkpoint. Group requests by complete bundle hash and compatible tensor shape. Initially keep float32, disable TF32/reduced precision and avoid quantization.

Proposed files: `src/joxing/inference.py`, `rollout_pool.py`, `tests/test_inference.py`, `tests/test_rollout_pool.py`; standalone measurements in `scripts/benchmark_inference.py` and `benchmark_rollout_pool.py`. Types:

```python
InferenceRequest(world_id, actor_id, episode_id, decision_id,
                 observation_time, bundle_hash, observation, hidden_state)
InferenceReply(world_id, actor_id, episode_id, decision_id,
               bundle_hash, action, next_hidden_state)
```

Preallocate per-worker shared-memory observation/action slots and use small control messages. Validate IDs, shape, finite values and ownership before applying actions. Bounded queues provide backpressure; a delayed or mismatched reply is never silently applied to a new episode. Recurrent state belongs to the actor/episode, not its current batch row. Gather both fighters' observations from one completed world state and obtain both decisions before advancing shared physics.

Start with synchronous batching to make ordering/replay simple. Consider bounded dynamic batches only after measuring straggler cost; retain batch membership/order/backend in the replay contract when needed. Wall-clock waiting is not simulated sensor/actuator jitter. Simulated delay and packet-drop queues require independent causal timestamps/RNG and their own tests.

Add a separate CUDA-enabled PyTorch runtime for inference/learning, preserving the working CPU lock and host driver. Check wheel/device compatibility first. Benchmark actual host-to-device/device-to-host and synchronization cost. `eval()` and inference mode are distinct requirements; inference mode alone does not select evaluation behavior. [PyTorch2.10 inference mode](https://docs.pytorch.org/docs/2.10/generated/torch.autograd.grad_mode.inference_mode.html). CUDA graphs/compile are later candidates for stable shapes after warmup and parity, not initial prerequisites. [CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html).

### 3. Keep correctness without serializing everything on the hot path

Never remove substep contact processing or the referee to obtain a faster FPS claim. Introduce separately hashed `audit` and `train` telemetry modes only after a fixed-action oracle proves identical physics, observations, events and terminations. Keep last-valid integration/application state in contiguous buffers; serialize on faults, registered checkpoints and selected replay boundaries. Aggregate all physical manifold impulse online even when every vector is not written to disk during every training game.

Retain every episode outcome, invalid reason, event ledger, policy/config/seed identity and reproducible command/RNG stream. Keep complete substep evidence for qualification/frozen evaluation, all faults and preregistered diagnostic scenarios. Rendering belongs to separate evaluation, never training's inner loop. Consider C++ collectors/referee reductions only if profiling shows that work dominates; keep MuJoCo as the physics engine and prove parity against Python fixtures.

### 4. Motor learning that can become a boxer

v8 resolves a real failure: tracking ends first, then all40 episodes latch excess tilt about0.81–0.84s later. Merely ignoring the tracking threshold does not provide stable locomotion. Before another broad sweep, inspect reference seams, normalization, root-translation sensitivity, actuator saturation/effort and feasible reset support using retained trajectories. A causal intervention gets its own preregistered variant; do not infer cause from clipping or one favorable clip.

Build a feasible stationary/recovery reset bank and goal-conditioned motor environment following master P3. Keep all700 muscles. Compare teacher-guided initialization/distillation against matched random initialization under equal compute. The released actor needs its native3601-feature contract; it cannot be loaded blindly into the1023-feature stance learner. Freeze decoder schedule/normalizer semantics explicitly. Do not use a walking reference as a boxing animation.

Curriculum: quiet stance → independent left/right reach and return → raised guard → steps/pivots → perturbation/contact recovery. Prefer valid short tasks with achievable goals and held-out evaluations over many immediate falls. Measure foot support, hand errors, drift, command variation and nonnegative effort, with existing stance/recovery gates. Only qualified skills enter a state-conditioned motor decoder. Freeze/version that decoder before tactical PPO; tactical exploration in a smaller latent space is more tractable than independently exploring1400 muscle commands each game.

### 5. Turn the rules into an authoritative two-player environment

Implement master P1/P2 before tactical rewards: fitted shells/self-contact, direct glove target attribution, one-punch/one-score, guard interception, retraction, simultaneous hits, foul/fall/exit review and numerical quarantine. Keep head2/torso1/guard0 and terminal win/loss utility. Do not incentivize injury meters, farming block force or surviving instead of winning.

Create `env.py`, `observations.py`, `referee.py`, `replay.py` and the planned physical/rule/leakage tests. Actors receive own proprioception, visible opponent position/velocity, public game state and time-stamped history. Privileged critic fields cannot leak into actors. Use a declared frame stack/GRU, not merely previous action. Replay includes recurrent state, reference/goal phase, delay queues, normalized action transform, ledger and RNG. Candidate cadence remains50Hz tactics/100Hz motor/1kHz physics and requires new motor compatibility checks; v8's native50Hz controller is not silently sped up.

### 6. Proper historical-opponent learning

Use recurrent tactical PPO with correct bounded-action likelihoods, sequence masks/burn-in and distinct bootstrap/trace masks. A separate learner updates weights from experience; the deployed policy changes muscle actions continuously with its weights fixed for the declared collection/match interval. Begin with fixed-policy collection waves to avoid unaccounted stale-policy PPO data. Record behavior log-probabilities/bundle hashes and enforce the registered lag policy before asynchronous collection.

Opponents include deterministic calibration controllers, immutable historical checkpoints and challenging exploiters, sampled at reset. Freeze actor+decoder+normalizer+timing/action contract per game; EMA may prepare a future bundle but cannot mutate the current opponent. PFSP needs a uniform floor, sparse-data handling and empty/all-zero/all-one fallbacks. A weight such as `w*(1-w)**2` alone discards unbeatable opponents. Keep a payoff matrix and a held-out opponent panel, not only Elo.

Use5/10/20s curriculum bouts only when they are valid games under explicit shorter-clock variants; qualify60s rules separately. Keep falls/fouls/invalids visible. Promotion uses200 paired starts with swapped sides, a positive lower95% improvement bound and no critical regression; use multiple training seeds for method claims. Optimize strength per wall hour, not monotonically increasing training reward/Elo.

### 7. Exact-model GPU physics after qualification

GPU physics could provide the largest arena count, but our existing exact-model Warp probe fails feature/parity gates. Preserve that failure. Audit the actual two-human model: muscles/activation, spatial tendons/wrapping, equalities, collision masks/meshes, integrator, all manifolds and buffer overflow. Use newer-driver hosts only as isolated compatibility candidates, without modifying any driver. Generic humanoid benchmarks do not establish native parity.

The pinned [MuJoCo3.8 Warp documentation](https://mujoco.readthedocs.io/en/3.8.0/mjwarp/index.html) distinguishes throughput from latency and discusses contact/constraint capacities and transfers. Test8/32/128/512 worlds subject to measured memory, not a declared4096 arenas. GPU numerical/backend variants need predeclared local tolerances, invariant/contact/referee fixtures and their own source-host replay; they do not loosen existing exact CPU gates. Only after qualification keep physics, observations, inference/referee reductions and rollouts device-resident.

## What 100,000 games costs

At50 tactical decisions/s,100,000 ten-second games contain50million world transitions;60s games contain300million. The table is arithmetic, excludes setup/evaluation/optimizer overhead and is **not a throughput forecast**:

| Measured complete-workload world transitions/s | 100,000 ×10s games | 100,000 ×60s games |
|---:|---:|---:|
| 1,000 |13.9h|83.3h|
| 10,000 |1.39h|8.33h|

A two-player world transition is one physical transition; it is not automatically two admitted learner samples. Report motor episodes, attempted games, valid games and invalid games separately. Current valid boxing-game count remains0. First milestones are1,000 valid curriculum games, then10,000 with a working frozen-panel learning curve, then100,000 if strength and validity justify the spend.

## Validation, ownership and handoff

One integration owner, no agents. `third_party/**`, completed protocols/evidence and frozen executing checkouts remain read-only. New work goes in a dedicated `~/joxing/v009-*`; raw output in `runs/`, compact results in the version's artifact/book paths. No concurrent referee/scene/schema writers. Across hosts retain the current ceiling of16 owned simulation workers with one computational thread each; larger GPU world batches are separately memory/physics-budgeted. Any later increase beyond the worker ceiling requires measured capacity/scaling evidence and explicit authorization.

Every performance protocol freezes source/config/model/actor/normalizer/backend hashes, case order, seeds, workload, timeout/memory/physics/raw caps and success criteria before outcomes. Run targeted/full regressions and lints; retain all timing samples, faults, not_run rows, warmup/startup and process exits. Validate scalar/batched normalization, action ownership, pending commands, queue restart/stale replies, hidden-state reset and strict fixed-action engine/replay invariants. CUDA/backend differences are explicit variants, not silently accepted drift.

The dashboard must report world transitions/s, physics substeps/s, actual admitted learner samples/s, optimizer updates/s, policy lag, reset rate, queue latency, RAM/VRAM, valid/invalid games and paired strength per wall hour. Publish the measured bottleneck and next mission even when an optimization loses. Stop on physics/referee mismatch, queue identity errors, changed bundles, buffer overflow, unknown contacts, source mutation or the first resource cap. Never trade correct games for a larger game counter.
