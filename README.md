# Joxing

**Public repository: [neverhuman/Joxing](https://github.com/neverhuman/Joxing).** Clone and read the source, tests, plans and published evidence without GitHub credentials.

Muscle-driven boxing research using two [MS-Human-700](https://github.com/LNSGroup/MS-Human-700) humans in one MuJoCo world and the [msgym](https://github.com/LNSGroup/msgym) control ecosystem.

**Latest release: v0.0.7.** Observation-correct glove/timestep transfer and exact Linux replay pass. **All 80 longer native tracking trials completed with zero invalids, but none reached the horizon**; even the unchanged native baseline stopped around 4.1 seconds. No new training or boxing competence is claimed. All 99 Linux tests pass; Mac full 98 plus final 27 targeted checks pass. [Results](docs/V0.0.7_RESULTS.md) · [Experiment book](experiments/v0.0.7/README.md) · [Four-condition replay](artifacts/v0.0.7/transfer/native-transfer.mp4).

**Continuing every 20 minutes:** the supervisor checks actual progress/faults, audits completion and advances the [current mission](docs/MISSION.md). [v0.0.8](docs/V0.0.8_MISSION.md) is ready: substep physical evidence and an explicit continuation diagnostic that separates native tracking failure from physical collapse. Existing stance/game gates remain in force.

**Prior v0.0.6 qualification:** the frozen upstream controller completed 40/40 short native walking tasks with exact Linux replay. Decoder schedule state lost on load was reproduced and repaired; no new optimization occurred. [Results](docs/V0.0.6_RESULTS.md).

**Prior training result, v0.0.5:** 1.44 million transitions and 912 new evaluations, zero invalids. Candidate B improved against its initial policy on two fresh seeds but lost to reference A on one. Replicated superiority and the five-second motor gate failed; every final policy had 0/32 five-second survivors. [Results](docs/V0.0.5_RESULTS.md).

The [state/history code audit](docs/STATE_HISTORY_SELF_PLAY_AUDIT.md) addresses opponent motion, temporal memory, responsive muscles and jitter. Current SAC observes its own velocity/activation and changes muscle commands at 100 Hz **simulated time**. Opponent observations, recurrent acting memory, a connected match referee and self-play league remain missing. The [motor-teacher roadmap](docs/UPSTREAM_MOTOR_TRANSFER_PLAN.md) now has native qualification evidence; glove and stationary/reach transfer remain next work.

**Current stage: executable foundation, not trained autonomous boxing.** The project preserves 700 muscles per actor, defines an auditable points game, tests contact accounting and anti-farming scoring, and includes a small constrained motor-learning experiment. Full-body stance, punching, self-play and GPU physics remain gated work.

**Prior completed proof: v0.0.2.** Muscle-driven strike, one-point referee, measured withdrawal and full mid-contact replay passed a frozen three-seed proof. Trained policies completed hit-and-withdrawal in 192/192 fresh trials versus 0/192 baseline withdrawals; hits remained 192/192 for both. Four CPU workers finished in 4 minutes 17 seconds. This remains a fixed-pelvis, zero-gravity diagnostic.

[v0.0.2 results and receipts](docs/V0.0.2_RESULTS.md) · [Iteration plan](docs/V0.0.2_PLAN.md) · [Before/after video](artifacts/v0.0.2/strike-before-after.mp4)

[Documentation index](docs/README.md) · [Training sequence and test gates](docs/TRAINING.md) · [v0.0.5 execution plan](docs/V0.0.5_PLAN.md) · [Execution status](docs/STATUS.md)

## v0.0.5: what the larger wave established

The original four 40,000-transition conditions tested reward units and entropy temperature. Its best gain was 0.102 seconds, below the registered confirmation threshold. Following the requested budget increase, four independent learners used 16 simulation workers total and 160,000 transitions per condition. Candidate B qualified in the new screen, then faced reference A on fresh seeds:

| Training seed | Initial survival (s) | Final A (s) | Final B (s) | B versus A |
|---|---:|---:|---:|---|
| 761 | 1.023 | 0.803 | 1.294 | +0.491s, positive paired interval |
| 983 | 1.011 | 1.165 | 1.097 | −0.068s, negative paired interval |

These are moving-start fall-recovery episodes, not stationary stance or boxing games. Both waves retain all 912 evaluation attempts and 14,736 completed stochastic motor episodes. Reset intervals do not measure uncertainty across training seeds. The frozen constant evaluated confirmation starts before learner training; learner-policy evaluations were deferred until final actor save. This timing limitation and failed strict cross-platform policy replay are disclosed in the [results](docs/V0.0.5_RESULTS.md).

Real optimizer throughput improved 4.69× on a common-data benchmark; a separate 16-worker rollout benchmark reached 844.67 transitions/s with optimization excluded. The full extension averaged about 360 training transitions/s including setup/evaluation. CPU physics remains the reference; GPU physics is unqualified. Source-host replay passes, all owned jobs exited, and no larger motor proof or competitive training was started.

![Both fresh-seed outcomes; motor and replicated-superiority gates failed](artifacts/v0.0.5/extension/wave-confirmation.png)

## v0.0.4 background: gravity and feedback learning

This iteration implements the body/environment/pilot stages of the earlier v0.0.3 plan. It asks whether feedback through all 700 muscles can improve local balance under ordinary gravity.

| Stage | Work and evidence | Gate to advance |
|---|---|---|
| 1. Qualify the body | Native-root falls, foot contact, muscle/inertia preservation, assistance audit | 32 accepted development resets and twelve finite root probes |
| 2. Test the environment | 1,023 measured observation values, 138 named muscle groups, push impulse, failures and snapshots | Physical/API regression tests; explicit support-channel order across sorted JSON |
| 3. Train and evaluate the pilot | SAC seed 73, up to 100,000 control transitions / 30 minutes; paired initial/final policies on 32 development starts | Publish all 64 attempts, survival change, faults and resource stop |
| 4. Qualify further training | At least 16/32 development starts survive 5 seconds, zero invalids | A failed pilot blocks the larger motor proof |
| 5. Later formal proof | Three training seeds; 768 held-out quiet/push attempts, up to 20 seconds each | Separately gated future run; >=61/64 stance and >=58/64 recovery successes per seed |

Each pilot evaluation lasts up to 20 seconds. Reset-only preparation/projection is logged; there is no fixed pelvis, gravity compensation or state overwrite during the evaluated trajectory. The initial hand position is a posture target, not a raised boxing guard. Report competence separately from measured learning. This historical pilot used at most four CPU simulation workers with one BLAS/OpenMP/Torch thread each; the current v0.0.5 allowance is 16 aggregate workers as described above. GPU simulation remains gated by failed parity tests. The larger program retains a 40-million-substep/four-hour/10-GiB cap; these are limits, not promised training times.

The [training guide](docs/TRAINING.md) distinguishes executable pilot commands from the still-planned formal proof and self-play. Full-body contact coverage, raised guard, reliable stance and sporting fall/foul attribution remain open. Linux snapshot continuation is exact; strict Linux-to-Mac closed-loop full-state parity misses tolerance and remains explicitly failed. The 1/2/4-worker inference/support/reset benchmark measured 65.7/131.8/240.5 control transitions per second, with optimizer work excluded.

![v0.0.4 measured training and paired evaluation; motor gate failed](artifacts/v0.0.4/balance-learning.png)

![Two-human scene inspection; static, not a learned fight](artifacts/scene-inspection.png)

## Start

```bash
git clone --recurse-submodules https://github.com/neverhuman/Joxing.git
cd Joxing
uv sync --extra dev --extra viz
uv run pytest -q
uv run joxing inspect
uv run joxing fixture
uv run joxing render
uv run joxing benchmark --steps 500
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run joxing train-reach --workers 4 --out runs/reach
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run joxing train-strike --workers 4 --out runs/v0.0.2/reproduction
```

The gravity audit needs only the base dependencies. SAC and its reports use the optional motor extra:

```bash
uv run joxing balance-probe --out runs/v0.0.4/my-probe
uv sync --extra dev --extra viz --extra motor
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run joxing train-stance --out runs/v0.0.4/my-pilot --steps 100000 --workers 4 --seed 73 --wall-s 1800 --eval-count 32
```

Runs refuse to overwrite an existing directory. `train-stance` executes a bounded development experiment; it does not start self-play or a formal motor proof. Linux installs CPU Torch wheels; NVIDIA hardware is optional for these commands.

Python 3.12+, MuJoCo 3.8.0. On headless NVIDIA Linux use `MUJOCO_GL=egl` for rendering. Mac can run CPU physics and native rendering without Docker. Run from the checkout: upstream models are pinned Git submodules and are not included in the Python wheel. Generated XML/receipts use local assets; rebuild from the pinned checkout when moving machines.

## Read first

- [Code audit: game state, motion history, jitter and repeated self-play](docs/STATE_HISTORY_SELF_PLAY_AUDIT.md)

- [Game rules and how RL earns reward](docs/GAME.md)
- [Review of all four supplied specs](docs/SPEC_REVIEW.md)
- [Actual compute inventory and HPC strategy](docs/COMPUTE.md)
- [Implementation gates and worker-ready handoff](docs/IMPLEMENTATION.md)
- [Delivered status, commands and evidence](docs/STATUS.md)

The candidate game awards head 2 points and torso 1 point per qualified direct glove encounter. Blocks score zero; sustained pressure cannot score repeatedly. Force and impulse remain visible evidence. Canonical tactical utility is win +1, loss -1, draw 0. No injury model is claimed.

## What executes today

- Namespaced dual scene with 1,400 native muscles, added glove equipment, provisional target/guard shells, ownership maps and mass/parameter checks.
- Staged CPU contact capture and independent momentum-consistency fixtures.
- Pure hit-event referee with direct-target impulse, onset gates, one-score locking, geometric/retraction input contract, idempotent delivery and mid-event restore.
- Candidate match-objective potential shaping with terminal return test.
- CPU diagnostic benchmark and episodic CEM muscle-reach policy search with explicit assistance, held-out reset evaluation, checkpoints and generation logs.
- Native-root gravity rig with reset-only equality projection, signed/persistent support, all-muscle feedback control, bounded SAC, fault receipts and complete control-boundary evaluation snapshots.

The referee is not yet connected to a production two-agent environment. Geometry fitting, complete self-contact, legal/foul adjudication and stable motor skills must pass before self-play. A supplied spec's proposed code or measurements are never treated as delivered capabilities.

## First measured learning result

On xbabe1, four CPU workers completed 384 CEM muscle-reach training episodes in 35 seconds. Over 32 paired held-out resets, mean terminal-window hand-target error fell from 33.9 cm to 4.4 cm. This is a fixed-pelvis, zero-gravity open-loop arm-control diagnostic, not learned boxing or self-play. One training seed and one target neighborhood limit the claim.

![Measured reach learning](artifacts/reach-learning.png)

[Before/after replay](artifacts/reach-before-after.mp4) · [Run receipt](artifacts/receipts/reach-seed7.json) · [Results and limitations](docs/RESULTS.md)

Native teacher reproduction uses the separate locked runtime and commands in the [v0.0.6 results](docs/V0.0.6_RESULTS.md#reproduce-the-bounded-native-qualification); its 3,601 feature interface differs from the stance rig. Mac native compiled-contract portability remains unqualified.

Joxing code is Apache-2.0. Upstream submodules retain their licenses. Supplied engineering documents under `docs/references` retain their original authorship; their inclusion is project reference material, not a separate license grant for third-party datasets. No additional motion-capture corpus is bundled by Joxing; the pinned msgym submodule includes its upstream sample trajectory, whose provenance must be checked before teacher transfer.
