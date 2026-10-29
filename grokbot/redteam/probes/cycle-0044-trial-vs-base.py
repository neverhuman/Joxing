"""Cycle 0044 diagnostic (post hoc, non-gating): does the training trial agree
with head-to-head against the frozen v0.0.22 base?

motor_gate.trial (the training measurement) of the base, v0.0.28@16 and @48
vs the base on the v0.0.28 monotonicity seeds 42000-42023.

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0044-trial-vs-base.py OUT.json
"""

import json
import pathlib
import sys

import numpy as np

from grokbot import motor_gate

r = json.load(open("runs/grokbot/self-play-opponent-gate-0043/report.json"))
v22 = np.array(
    json.load(open("grokbot/redteam/data/cycle-0031-knockdown-gate-summary.json"))[
        "final_params"
    ]
)
specs = {
    "base": ("group_pd", v22, 0.05),
    "at16": ("group_pd", np.array(r["params_at_16"]), 0.05),
    "at48": ("group_pd", np.array(r["final_params"]), 0.05),
}
seeds = list(range(42000, 42024))
out = {}
with motor_gate.pool(4, 0.5, 3000) as pool:
    for name, spec in specs.items():
        res = pool.map(motor_gate.trial, [(spec, specs["base"], s) for s in seeds])
        fit = [
            motor_gate.fitness(x, 3000, 1.0, valid_knockdown=True, contact_penalty=0.5)
            for x in res
        ]
        opp_first = [
            x["opponent_fall_tick"] is not None
            and x["opponent_fall_tick"] < x["stand_ticks"]
            for x in res
        ]
        me_first = [
            x["stand_ticks"] < 3000
            and (
                x["opponent_fall_tick"] is None
                or x["stand_ticks"] < x["opponent_fall_tick"]
            )
            for x in res
        ]
        out[name] = {
            "mean_fitness": float(np.mean(fit)),
            "mean_stand_s": float(np.mean([x["stand_ticks"] for x in res])) / 1000,
            "valid_knockdown": sum(motor_gate.valid_knocked_down(x) for x in res),
            "opponent_falls_first": sum(opp_first),
            "self_falls_first": sum(me_first),
            "mean_contact": float(np.mean([x["contact_ticks"] for x in res])),
            "opp_fall_median_s": float(
                np.median([x["opponent_fall_tick"] or 3000 for x in res])
            )
            / 1000,
            "per_seed": [
                (
                    x["stand_ticks"],
                    x["opponent_fall_tick"],
                    x["opponent_fall_attribution"],
                )
                for x in res
            ],
        }
        print(
            name,
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in out[name].items()
                if k != "per_seed"
            },
            flush=True,
        )
pathlib.Path(sys.argv[1]).write_text(json.dumps(out))
