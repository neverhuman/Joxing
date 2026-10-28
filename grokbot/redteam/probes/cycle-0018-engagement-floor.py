"""v0.0.14 engagement-floor league as preregistered in protocols/v0.0.14.json.

    PYTHONPATH=. nice -n 10 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0018-engagement-floor.py \
        runs/grokbot/engagement-floor-0018
"""

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

from grokbot import selfplay
from grokbot.motor_primitives import punch_params

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.14.json")
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")


def main(out):
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t, e = protocol["training"], protocol["evaluation"]
    gen4 = np.array(json.loads(SUMMARY.read_text())["archive"]["gen4"])
    began = time.monotonic()
    report = selfplay.run_league(
        generations=t["generations"],
        iterations=t["iterations"],
        pairs=t["pairs"],
        sigma=t["sigma"],
        lr=t["lr"],
        eval_seeds=e["eval_seeds"],
        eval_noise=e["eval_noise"],
        workers=t["workers"],
        horizon_ticks=t["horizon_ticks"],
        seed=t["seed"],
        separation_m=protocol["reset"]["separation_m"],
        anti_stalling=True,
        shaping_weight=t["training_only_terms"]["shaping_weight"],
        shaping_key="extension",
        strike_bonus=t["training_only_terms"]["strike_bonus"],
        min_engaged_share=0.25,
        init_params=punch_params(gen4, onset=-3.0, duration=0.0, amplitude=2.0),
        policy_kind="punch",
        frozen={"v12base": ("params", gen4)},
        valid_bonus=t["training_only_terms"]["valid_bonus"],
        no_contest_penalty=t["training_only_terms"]["no_contest_penalty"],
        log=lambda row: print(json.dumps(row), flush=True),
    )
    report["protocol_sha256"] = hashlib.sha256(raw).hexdigest()
    report["wall_s"] = round(time.monotonic() - began, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("wall_s", report["wall_s"], "protocol", report["protocol_sha256"])


if __name__ == "__main__":
    main(sys.argv[1])
