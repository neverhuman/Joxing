# Cycle 0044 diagnostic (post hoc, non-gating). Run as:
#   PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
#       python grokbot/redteam/probes/cycle-0044-rank-shares.py OUT.json
"""Measure each opponent's share of the ES update's |rank difference| under
pooled centred ranks (as v0.0.26-v0.0.28 trained) and under per-opponent ranks,
at the v0.0.22 start and at v0.0.28@16. Same task construction and fitness as
motor_gate.train (12 pairs, sigma 0.1, rotation v0.0.19/v0.0.13/base); seeds
from rng 99441 (not a registered seed range); 2 batches per point; lr 0."""

import json
import pathlib
import sys

import numpy as np

from grokbot import motor_gate
from grokbot.motor_primitives import punch_params
from grokbot.selfplay import centred_ranks

r = json.load(open("runs/grokbot/self-play-opponent-gate-0043/report.json"))
v22 = np.array(
    json.load(open("grokbot/redteam/data/cycle-0031-knockdown-gate-summary.json"))[
        "final_params"
    ]
)
v19 = np.array(
    json.load(open("grokbot/redteam/data/cycle-0025-hit-stay-gate-summary.json"))[
        "final_params"
    ]
)
gen4 = np.array(
    json.load(open("grokbot/redteam/data/cycle-0014-league-summary.json"))["archive"][
        "gen4"
    ]
)
c13 = punch_params(gen4, onset=-3.0, duration=0.0, amplitude=2.0)
trainers = [("group_pd", v19, 0.05), ("punch", c13, 0.05), ("group_pd", v22, 0.05)]
points = {"base": v22, "at16": np.array(r["params_at_16"])}


def diffs(paired):
    ranks = centred_ranks(paired.ravel()).reshape(-1, 2)
    return ranks[:, 0] - ranks[:, 1]


def shares(d, groups):
    return [float(np.abs(d[groups == g]).sum() / np.abs(d).sum()) for g in range(3)]


rng = np.random.default_rng(99441)
out = {}
with motor_gate.pool(3, 0.5, 3000) as pool:
    for name, theta in points.items():
        rows = []
        for batch in range(2):
            noise = rng.standard_normal((12, len(theta)))
            seeds = rng.integers(2**31, size=12)
            tasks = [
                (
                    ("group_pd", theta + s * 0.1 * noise[k]),
                    trainers[k % 3],
                    int(seeds[k]),
                )
                for k in range(12)
                for s in (1, -1)
            ]
            res = pool.map(motor_gate.trial, tasks)
            f = [
                motor_gate.fitness(
                    x, 3000, 1.0, valid_knockdown=True, contact_penalty=0.5
                )
                for x in res
            ]
            ok = np.array(
                [f[2 * k] is not None and f[2 * k + 1] is not None for k in range(12)]
            )
            paired = np.array([[f[2 * k], f[2 * k + 1]] for k in range(12) if ok[k]])
            groups = (np.arange(12) % 3)[ok]
            split = np.empty(len(paired))
            for g in range(3):
                split[groups == g] = diffs(paired[groups == g])
            rows.append(
                {
                    "pooled": shares(diffs(paired), groups),
                    "split": shares(split, groups),
                    "mean_fitness_by_opponent": [
                        float(paired[groups == g].mean()) for g in range(3)
                    ],
                    "pooled_total": float(np.abs(diffs(paired)).sum()),
                    "split_total": float(np.abs(split).sum()),
                }
            )
            print(
                name,
                batch,
                {
                    k: [round(v, 3) for v in x] if isinstance(x, list) else round(x, 3)
                    for k, x in rows[-1].items()
                },
                flush=True,
            )
        out[name] = rows
pathlib.Path(sys.argv[1]).write_text(json.dumps(out))
