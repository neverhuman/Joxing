"""v0.0.12 league exactly as preregistered in grokbot/redteam/protocols/v0.0.12.json.

    PYTHONPATH=. nice -n 10 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0016-engagement-league.py \
        runs/grokbot/engagement-0016
"""

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

from grokbot import selfplay

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.12.json")
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")


def main(out):
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t, e = protocol["training"], protocol["evaluation"]
    init = json.loads(SUMMARY.read_text())["archive"]["gen4"]
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
        shaping_weight=t["shaping_weight"],
        min_engaged_share=protocol["gates"]["min_engaged_share"],
        init_params=np.array(init),
        require_valid=False,  # reproduces a recorded degenerate run
        log=lambda row: print(json.dumps(row), flush=True),
    )
    report["protocol_sha256"] = hashlib.sha256(raw).hexdigest()
    report["wall_s"] = round(time.monotonic() - began, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("wall_s", report["wall_s"], "protocol", report["protocol_sha256"])


if __name__ == "__main__":
    main(sys.argv[1])
