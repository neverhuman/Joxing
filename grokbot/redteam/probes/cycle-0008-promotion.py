"""End-to-end promotion decision in the real env (cycle 0008).

arena (side-swapped pairs, common random numbers) -> reference endings ->
promotion gate. Plays candidate and incumbent against each other and against
every panel member, then prints the gate's decision both ways round on the same
records.

    PYTHONPATH=. uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0008-promotion.py 1300 10
"""

import itertools
import json
import sys
import time

from grokbot.arena import SCRIPTED_PANEL, paired, policy_summary
from grokbot.promotion import decide
from joxing.env import BoxingEnv
from joxing.scene import build_scene

CANDIDATE, INCUMBENT = "baseline", "relaxed"
PANEL = ["relaxed", "cocontract", "noise"]


def main(horizon_ticks, seeds):
    env = BoxingEnv(build_scene(), horizon_ticks=horizon_ticks)
    names = sorted({CANDIDATE, INCUMBENT, *PANEL})
    began = time.monotonic()
    records = []
    for a, b in itertools.combinations(names, 2):
        if {a, b} & {CANDIDATE, INCUMBENT}:
            records += paired(env, SCRIPTED_PANEL, a, b, seeds)
    print(
        f"games {len(records)} seeds {len(seeds)} wall_s {time.monotonic() - began:.1f}"
    )
    for name, row in policy_summary(records).items():
        print(name, json.dumps(row))
    for cand, inc in ((CANDIDATE, INCUMBENT), (INCUMBENT, CANDIDATE)):
        d = decide(records, cand, inc, PANEL)
        print(f"== {cand} over {inc}: promote={d.promote}")
        print("   reasons", list(d.reasons))
        print("   head_to_head", json.dumps(d.head_to_head))
        print("   per_member", json.dumps(d.per_member))


if __name__ == "__main__":
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 1300
    main(horizon, list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 10)))
