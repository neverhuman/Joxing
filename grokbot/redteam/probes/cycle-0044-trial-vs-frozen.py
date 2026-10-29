"""Cycle 0044 diagnostic (post hoc, non-gating): did v0.0.28 trade the base
matchup for the frozen opponents early in training?

motor_gate.trial of the base, @16 and @48 vs frozen v0.0.19 and v0.0.13 on
seeds 42000-42023, plus a post-hoc head-to-head @48 vs the base (not a gate;
it cannot rescue a tag).

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0044-trial-vs-frozen.py OUT.json
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np

from grokbot import motor_gate
from grokbot.motor_primitives import punch_params

spec = importlib.util.spec_from_file_location(
    "g", "grokbot/redteam/probes/cycle-0025-hit-stay-gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
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
pol = {
    "base": ("group_pd", v22, 0.05),
    "at16": ("group_pd", np.array(r["params_at_16"]), 0.05),
    "at48": ("group_pd", np.array(r["final_params"]), 0.05),
}
opp = {"v0019": ("group_pd", v19, 0.05), "v0013": ("punch", c13, 0.05)}
seeds = list(range(42000, 42024))
out = {}
with motor_gate.pool(4, 0.5, 3000) as pool:
    for o, ospec in opp.items():
        for name, spec in pol.items():
            res = pool.map(motor_gate.trial, [(spec, ospec, s) for s in seeds])
            fit = [
                motor_gate.fitness(
                    x, 3000, 1.0, valid_knockdown=True, contact_penalty=0.5
                )
                for x in res
            ]
            out[f"{name}_vs_{o}"] = {
                "mean_fitness": round(float(np.mean(fit)), 3),
                "vkd": sum(motor_gate.valid_knocked_down(x) for x in res),
                "stand_s": round(
                    float(np.mean([x["stand_ticks"] for x in res])) / 1000, 3
                ),
                "contact": round(float(np.mean([x["contact_ticks"] for x in res]))),
            }
            print(f"{name}_vs_{o}", out[f"{name}_vs_{o}"], flush=True)
h = gate.head_to_head(pol["at48"], pol["base"], seeds)
out["posthoc_h2h_48_vs_base"] = gate.brief(h)
print(
    "posthoc_h2h_48_vs_base",
    {
        k: h[k]
        for k in (
            "games",
            "valid_games",
            "wins",
            "losses",
            "draws",
            "no_contest",
            "blocked",
            "win_rate_wilson95",
        )
    },
    flush=True,
)
pathlib.Path(sys.argv[1]).write_text(json.dumps(out))
