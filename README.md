# Joxing

Muscle-driven boxing research using two [MS-Human-700](https://github.com/LNSGroup/MS-Human-700) humans in one MuJoCo world and the [msgym](https://github.com/LNSGroup/msgym) control ecosystem.

**Current stage: executable foundation, not trained autonomous boxing.** The project preserves 700 muscles per actor, defines an auditable points game, tests contact accounting and anti-farming scoring, and includes a small constrained motor-learning experiment. Full-body stance, punching, self-play and GPU physics remain gated work.

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
