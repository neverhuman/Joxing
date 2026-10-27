"""Run a diagnostic self-play league and save its full report (cycle 0014).

    PYTHONPATH=. nice -n 10 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0014-selfplay-league.py \
        runs/grokbot/selfplay-0014

Writes report.json (per-generation stats, archive parameters and every
evaluation game) into the given directory, which must not exist yet or be empty.
"""

import json
import pathlib
import sys
import time

from grokbot import selfplay

CONFIG = dict(
    generations=6,
    iterations=3,
    pairs=8,
    sigma=0.1,
    lr=0.05,
    eval_seeds=[0, 1, 2, 3, 4, 5],
    eval_noise=0.05,
    workers=4,
    horizon_ticks=3000,
    seed=1,
)


def main(out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite")
    began = time.monotonic()
    report = selfplay.run_league(
        **CONFIG, log=lambda row: print(json.dumps(row), flush=True)
    )
    report["config"] = CONFIG
    report["wall_s"] = round(time.monotonic() - began, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("wall_s", report["wall_s"])


if __name__ == "__main__":
    main(sys.argv[1])
