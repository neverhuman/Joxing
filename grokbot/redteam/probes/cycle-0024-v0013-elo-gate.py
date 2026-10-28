"""v0.0.18 addendum: Elo/win-rate gate vs the frozen v0.0.13 champion.

Preregistered in protocols/v0.0.18-addendum-v0013.json (owner 06:28) before any
v0.0.18 result was read. Reads the trained v0.0.18 policy from the run report.

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0024-v0013-elo-gate.py \
        runs/grokbot/group-pd-gate-0023/report.json runs/grokbot/v0013-elo-gate-0024
"""

import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np

from grokbot import selfplay
from grokbot.motor_primitives import punch_params
from grokbot.promotion import _utilities, hoeffding_lower

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.18-addendum-v0013.json")
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")
HORIZON = 3000


def elo(p):
    return 400 * math.log10(p / (1 - p))


def wilson(wins, n, z=1.959964):
    if n == 0:
        return None
    centre = (wins + z * z / 2) / (n + z * z)
    half = z * math.sqrt(wins * (n - wins) / n + z * z / 4) / (n + z * z)
    return [centre - half, centre + half]


def head_to_head(protocol, params):
    """cycle-0023's head-to-head statistics, against the frozen v0.0.13 champion."""
    h, eval_noise = protocol["head_to_head_v0013"], 0.05
    gen4 = np.array(json.loads(SUMMARY.read_text())["archive"]["gen4"])
    champion = punch_params(gen4, onset=-3.0, duration=0.0, amplitude=2.0)
    spec = ("group_pd", params, eval_noise)
    other = ("punch", champion, eval_noise)
    tasks = [(spec, other, side, s) for s in h["seeds"] for side in selfplay.PLAYERS]
    with selfplay._pool(4, HORIZON, separation_m=0.5, anti_stalling=True) as pool:
        games = pool.map(selfplay.play_game, tasks)
    valid = [selfplay.valid_result(x) for x in games]
    records = [
        {
            "first": "candidate",
            "second": "v0.0.13",
            "seed": task[3],
            "outcome": game["utility"] if ok else None,
        }
        for task, game, ok in zip(tasks, games, valid, strict=True)
    ]
    samples = _utilities(records, "candidate", "v0.0.13")
    n = len(samples)
    mean = float(np.mean(samples)) if samples else 0.0
    lower = hoeffding_lower(mean, n, 0.95)
    wins = [x for x, ok in zip(games, valid, strict=True) if ok and x["utility"] > 0]
    hit_wins = sum(x["scoring_strike_ticks"] > 0 for x in wins)
    n_valid = sum(valid)
    p_low = (1 + lower) / 2
    checks = {
        "a_valid_share_at_least_half": n_valid >= 0.5 * len(games),
        "b_lower95_above_zero": lower > 0,
        "c_half_of_wins_with_hits": bool(wins) and hit_wins >= 0.5 * len(wins),
    }
    return {
        "games": len(games),
        "valid_games": n_valid,
        "valid_share": n_valid / len(games),
        "no_contest": sum(bool(x["no_contest"]) for x in games),
        "blocked": sum(bool(x["blocked"]) for x in games),
        "invalid": sum(x["invalid"] is not None for x in games),
        "wins": len(wins),
        "losses": sum(ok and x["utility"] < 0 for x, ok in zip(games, valid)),
        "draws": sum(ok and x["utility"] == 0 for x, ok in zip(games, valid)),
        "wins_with_hits": hit_wins,
        "win_rate": len(wins) / n_valid if n_valid else None,
        "win_rate_wilson95": wilson(len(wins), n_valid),
        "paired_samples": n,
        "mean": mean if samples else None,
        "lower95": lower,
        "elo": elo(min(max((1 + mean) / 2, 0.5 / n), 1 - 0.5 / n)) if n else None,
        "elo_lower95": elo(min(p_low, 1 - 1e-12)) if p_low > 0 else float("-inf"),
        "checks": checks,
        "passed": all(checks.values()),
        "results": [
            {k: v for k, v in x.items() if k != "points"} | {"side": t[2], "seed": t[3]}
            for t, x in zip(tasks, games, strict=True)
        ],
    }


def main(report_path, out):
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    run = json.loads(pathlib.Path(report_path).read_text())
    began = time.monotonic()
    result = head_to_head(protocol, np.array(run["final_params"]))
    result["protocol_sha256"] = hashlib.sha256(raw).hexdigest()
    result["wall_s"] = round(time.monotonic() - began, 1)
    (out / "report.json").write_text(json.dumps(result, indent=1))
    print({k: v for k, v in result.items() if k != "results"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
