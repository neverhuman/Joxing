"""Fresh-seed check of the cycle-0014 gen5/gen6 regressions (cycle 0015).

(A) Round robin among gen2..gen6 on fresh evaluation seeds 100-111, with
    side-swapped pairs and the same seeded action noise (0.05).
(B) Each checkpoint's own first non-foot floor support, played against the
    scripted baseline to the bell (arena), on the same fresh seeds.

    PYTHONPATH=. nice -n 10 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0015-regression-check.py
"""

import itertools
import json
import multiprocessing
import pathlib

import numpy as np

from grokbot import arena, selfplay
from joxing.env import BoxingEnv
from joxing.scene import build_scene

SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")
NAMES = ["gen2", "gen3", "gen4", "gen5", "gen6"]
SEEDS = list(range(100, 112))
NOISE = 0.05
_ARENA = None


def _init_arena():
    global _ARENA
    _ARENA = BoxingEnv(build_scene(), horizon_ticks=2000)


def _stand(task):
    name, params, seed = task
    panel = {
        name: selfplay.LinearPolicy(params, NOISE),
        "baseline": arena.constant(0.0),
    }
    record = arena.play(_ARENA, panel, name, "baseline", "A", seed)
    own = [e["tick"] for e in record["losses"] if e["policy"] == name]
    return name, own[0] if own else None


def main():
    archive = json.loads(SUMMARY.read_text())["archive"]
    archive = {k: np.array(v) for k, v in archive.items()}
    ctx = multiprocessing.get_context("fork")
    with selfplay._pool(4, 3000) as pool:
        tasks, meta = [], []
        for a, b in itertools.combinations(NAMES, 2):
            for (cand, opp, side, s), task in selfplay.evaluation_tasks(
                a, archive, [b], SEEDS, NOISE
            ):
                tasks.append(task)
                meta.append((cand, opp, s))
        outcomes = pool.map(selfplay.play_game, tasks)
    pairs = {}
    for (cand, opp, s), out in zip(meta, outcomes, strict=True):
        pairs.setdefault((cand, opp), {}).setdefault(s, []).append(out["utility"])
    print("(A) fresh-seed head to head, mean paired utility of row vs column")
    for (cand, opp), by_seed in pairs.items():
        samples = [np.mean(v) for v in by_seed.values()]
        print(f"  {cand} vs {opp}: {np.mean(samples):+.2f} over {len(samples)} pairs")
    with ctx.Pool(4, initializer=_init_arena) as pool:
        falls = pool.map(_stand, [(n, archive[n], s) for n in NAMES for s in SEEDS])
    print("(B) own first non-foot support vs baseline (ticks), fresh seeds")
    for name in NAMES:
        ticks = [t for n, t in falls if n == name]
        known = [t for t in ticks if t is not None]
        print(
            f"  {name}: median {np.median(known):.0f}  min {min(known)}  "
            f"max {max(known)}  stood to bell {ticks.count(None)}/{len(ticks)}"
        )


if __name__ == "__main__":
    main()
