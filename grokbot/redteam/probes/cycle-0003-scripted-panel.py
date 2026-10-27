"""Scripted-panel round robin in the real diagnostic BoxingEnv (cycle 0003, F1).

Every pair of SCRIPTED_PANEL entries plays both corner assignments for seed 0
from the native keyframe, for HORIZON_TICKS physics ticks. Prints one JSON line
per attempted game; nothing is dropped.

    PYTHONPATH=. uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0003-scripted-panel.py 5000
"""

import itertools
import json
import sys
import time

from grokbot.arena import SCRIPTED_PANEL, paired
from joxing.env import BoxingEnv
from joxing.scene import build_scene

KEYS = ("first", "second", "first_side", "outcome", "points", "events",
        "fall_tick", "ticks", "invalid_reason")


def main(horizon_ticks):
    env = BoxingEnv(build_scene(), horizon_ticks=horizon_ticks)
    began = time.monotonic()
    for first, second in itertools.combinations(SCRIPTED_PANEL, 2):
        for record in paired(env, SCRIPTED_PANEL, first, second, seeds=[0]):
            row = {k: record[k] for k in KEYS}
            row["wall_s"] = round(time.monotonic() - began, 1)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
