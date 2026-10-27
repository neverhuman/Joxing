"""Reference ending table at engagement range (cycle 0006).

Runs the scripted round robin, every pair from both corners, in one registered
reset scenario. Prints the per-policy summary (wins, losses, blocked-as-faller,
faults, invalids, falls) and the payoff matrix under the reference ending.

    PYTHONPATH=. uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0006-engagement-range.py \
        close-overlap-collision-probe-v1 2000 0 1
"""

import itertools
import json
import sys
import time

import numpy as np

from grokbot.arena import SCRIPTED_PANEL, paired, policy_summary
from grokbot.league import Game, Ledger
from joxing.env import BoxingEnv
from joxing.scene import build_scene


def main(scenario, horizon_ticks, seeds):
    env = BoxingEnv(build_scene(), horizon_ticks=horizon_ticks, scenario=scenario)
    names = list(SCRIPTED_PANEL)
    records = []
    began = time.monotonic()
    for first, second in itertools.combinations(names, 2):
        records += paired(env, SCRIPTED_PANEL, first, second, seeds)
    print(f"scenario {scenario} horizon {horizon_ticks} seeds {seeds}")
    print(f"games {len(records)} wall_s {time.monotonic() - began:.1f}")
    reasons = {}
    for r in records:
        key = r["invalid_reason"] or (r["ending"] or {}).get("reason", "?")
        key = key.split(":")[0] if r["invalid_reason"] else key
        reasons[key] = reasons.get(key, 0) + 1
    print("endings", json.dumps(reasons, sort_keys=True))
    for name, row in policy_summary(records).items():
        print(name, json.dumps(row))
    ledger = Ledger(names)
    for r in records:
        valid = r["outcome"] is not None
        ledger.add(
            Game(
                r["first"],
                r["second"],
                r["first_side"],
                r["outcome"] if valid else None,
                None if valid else r["invalid_reason"],
            )
        )
    matrix, count = ledger.payoff()
    print("valid games per pair\n", count.astype(int))
    print("payoff (row vs column)", names)
    print(np.array2string(matrix, precision=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [int(s) for s in sys.argv[3:]] or [0])
