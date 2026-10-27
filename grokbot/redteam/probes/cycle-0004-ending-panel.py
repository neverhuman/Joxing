"""Scripted round robin under the reference ending table (cycle 0004).

Plays every SCRIPTED_PANEL pair from both corners for each seed, then rates the
panel with grokbot.league: payoff matrix, Bradley-Terry/Elo, exact Nash mixture
and the exploitability of each pure policy. Compares the points-only bell result
(the env's current rule) with the measured-ending outcome.

    PYTHONPATH=. uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0004-ending-panel.py 2000 0 1
"""

import itertools
import json
import sys
import time

import numpy as np

from grokbot.arena import SCRIPTED_PANEL, paired
from grokbot.league import Game, Ledger, exploitability, solve_zero_sum
from joxing.env import BoxingEnv
from joxing.scene import build_scene


def main(horizon_ticks, seeds):
    env = BoxingEnv(build_scene(), horizon_ticks=horizon_ticks)
    names = list(SCRIPTED_PANEL)
    ledgers = {"bell": Ledger(names), "ending": Ledger(names)}
    began = time.monotonic()
    for first, second in itertools.combinations(names, 2):
        for r in paired(env, SCRIPTED_PANEL, first, second, seeds):
            print(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "first",
                            "second",
                            "first_side",
                            "seed",
                            "outcome",
                            "bell_outcome",
                            "ending",
                            "losses",
                            "invalid_reason",
                        )
                    }
                ),
                flush=True,
            )
            invalid = r["invalid_reason"]
            for key, outcome in (("bell", r["bell_outcome"]), ("ending", r["outcome"])):
                ledgers[key].add(
                    Game(
                        r["first"],
                        r["second"],
                        r["first_side"],
                        None if invalid else outcome,
                        invalid,
                    )
                )
    print(f"wall_s {time.monotonic() - began:.1f}")
    for key, ledger in ledgers.items():
        matrix, _ = ledger.payoff()
        counts = ledger.counts()
        print(
            f"== {key}: attempted {counts['attempted'].sum() // 2}, "
            f"invalid {counts['invalid'].sum() // 2}, "
            f"side_balance {ledger.side_balance():+.3f}"
        )
        print("payoff (row vs column)", names)
        print(np.array2string(matrix, precision=2))
        if np.any(matrix):
            p, _, value = solve_zero_sum(matrix)
            print("elo", dict(zip(names, np.round(ledger.ratings(), 1).tolist())))
            print(
                "nash", dict(zip(names, np.round(p, 3).tolist())), f"value {value:+.3f}"
            )
            print(
                "exploitability of each pure policy",
                {
                    n: exploitability(matrix, np.eye(len(names))[i])
                    for i, n in enumerate(names)
                },
            )
        else:
            print("all-zero payoff: no learning signal")


if __name__ == "__main__":
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(horizon, [int(s) for s in sys.argv[2:]] or [0])
