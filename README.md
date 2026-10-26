# Joxing

**Public repository: [neverhuman/Joxing](https://github.com/neverhuman/Joxing).** Clone and read the source, tests, plans and published evidence without GitHub credentials.

Muscle-driven boxing research using two [MS-Human-700](https://github.com/LNSGroup/MS-Human-700) humans in one MuJoCo world and the [msgym](https://github.com/LNSGroup/msgym) control ecosystem.

**Current stage: executable foundation, not trained autonomous boxing.** The project preserves 700 muscles per actor, defines an auditable points game, tests contact accounting and anti-farming scoring, and includes a small constrained motor-learning experiment. Full-body stance, punching, self-play and GPU physics remain gated work.

**Latest iteration: v0.0.2.** Muscle-driven strike, one-point referee, measured withdrawal and full mid-contact replay passed a frozen three-seed proof. Trained policies completed hit-and-withdrawal in 192/192 fresh trials versus 0/192 baseline withdrawals; hits remained 192/192 for both. Four CPU workers finished in 4 minutes 17 seconds. This remains a fixed-pelvis, zero-gravity diagnostic.

[v0.0.2 results and receipts](docs/V0.0.2_RESULTS.md) · [Iteration plan](docs/V0.0.2_PLAN.md) · [Before/after video](artifacts/v0.0.2/strike-before-after.mp4)

**Next planned iteration: [v0.0.3 gravity, balance and recovery tests](docs/V0.0.3_PLAN.md).** Test native-root falls, honest foot support and feasible resets, then run a bounded feedback-learning experiment under gravity. The plan defines held-out stance/push tests, replay, compute limits and separate test-platform and motor-competence gates. Implementation has not started.

[Documentation index](docs/README.md) · [Training sequence and test gates](docs/TRAINING.md) · [Detailed v0.0.3 implementation plan](docs/V0.0.3_PLAN.md) · [Execution status](docs/STATUS.md)

## Next version: v0.0.3

The next proof asks whether one athlete can stay upright and recover under gravity using feedback through its muscles. Training proceeds in this order; a failed prerequisite stops the later stages.

| Stage | Work and evidence | Gate to advance |
|---|---|---|
| 1. Qualify the body | Native-root falls, foot contact, muscle/inertia preservation, assistance audit | Mechanically valid model and reproducible stance resets |
| 2. Test the environment | Real observations, all 700 muscles through 138 named groups, push impulse, failure handling and replay | Deterministic physics/API tests pass |
| 3. Run a small SAC pilot | Seed 73; at most 100,000 control transitions / 30 minutes on development scenarios | At least 16/32 starts survive 5 seconds, zero invalids; diagnose failure before a larger run |
| 4. Freeze and train | Three seeds, at most 500,000 control transitions per seed | Frozen model, resets, reward, budgets and final-checkpoint rule |
| 5. Evaluate and publish | Initial/final policies on 64 quiet and 64 push scenarios per seed: 768 attempts | Per seed: at least 61/64 stance and 58/64 recovery successes, zero invalids; independent replay and all results published |

Each evaluation lasts up to 20 seconds. Report competence separately from learned improvement against the paired initial policies. Start with CPU MuJoCo on xbabe1, at most four simulation workers, one BLAS/OpenMP thread each; use xbabe4 for independent CPU replay. Recheck shared-host availability before launch. Total compute cap: 40 million physics substeps, four hours wall time and 10 GiB artifacts. These are limits, not promised training times. GPU simulation remains gated by failed parity tests.

The [training guide](docs/TRAINING.md) explains the learner, tests, artifacts and next steps after v0.0.3. Its new stance commands are planned interfaces; the commands in **Start** below execute the delivered v0.0.2 code.

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

Python 3.12+, MuJoCo 3.8.0. On headless NVIDIA Linux use `MUJOCO_GL=egl` for rendering. Mac can run CPU physics and native rendering without Docker. Run from the checkout: upstream models are pinned Git submodules and are not included in the Python wheel. Generated XML/receipts use local assets; rebuild from the pinned checkout when moving machines.

## Read first

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

The referee is not yet connected to a production two-agent environment. Geometry fitting, complete self-contact, legal/foul adjudication and stable motor skills must pass before self-play. A supplied spec's proposed code or measurements are never treated as delivered capabilities.

## First measured learning result

On xbabe1, four CPU workers completed 384 CEM muscle-reach training episodes in 35 seconds. Over 32 paired held-out resets, mean terminal-window hand-target error fell from 33.9 cm to 4.4 cm. This is a fixed-pelvis, zero-gravity open-loop arm-control diagnostic, not learned boxing or self-play. One training seed and one target neighborhood limit the claim.

![Measured reach learning](artifacts/reach-learning.png)

[Before/after replay](artifacts/reach-before-after.mp4) · [Run receipt](artifacts/receipts/reach-seed7.json) · [Results and limitations](docs/RESULTS.md)

Joxing code is Apache-2.0. Upstream submodules retain their licenses. Supplied engineering documents under `docs/references` retain their original authorship; their inclusion is project reference material, not a separate license grant for third-party datasets. No motion-capture dataset is bundled.
