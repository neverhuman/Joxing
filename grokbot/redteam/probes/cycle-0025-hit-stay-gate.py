"""v0.0.19 hit-and-stay-up run as preregistered in protocols/v0.0.19.json.

Primary (level-up): the a/b/c gate vs the frozen v0.0.13 champion on new seeds.

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0025-hit-stay-gate.py \
        runs/grokbot/hit-stay-gate-0025
"""

import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np

from grokbot import motor_gate, selfplay
from grokbot.motor_primitives import punch_params
from grokbot.promotion import _utilities, hoeffding_lower

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.19.json")
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")
V16 = pathlib.Path("grokbot/redteam/data/cycle-0020-group-gate-summary.json")
V18 = pathlib.Path("grokbot/redteam/data/cycle-0023-group-pd-gate-summary.json")
HORIZON = 3000
NOISE = 0.05


def sign_test(candidate, reference):
    """One-sided exact sign test that ``candidate`` stands longer (ties dropped)."""
    better = sum(c > r for c, r in zip(candidate, reference, strict=True))
    n = sum(c != r for c, r in zip(candidate, reference, strict=True))
    p = sum(math.comb(n, k) for k in range(better, n + 1)) / 2**n if n else 1.0
    return {"better": better, "pairs": n, "one_sided_p": p}


def elo(p):
    return 400 * math.log10(p / (1 - p))


def wilson(wins, n, z=1.959964):
    if n == 0:
        return None
    centre = (wins + z * z / 2) / (n + z * z)
    half = z * math.sqrt(wins * (n - wins) / n + z * z / 4) / (n + z * z)
    return [centre - half, centre + half]


def head_to_head(candidate, opponent, seeds):
    """The cycle-0023/0024 a/b/c statistics for one candidate vs one opponent."""
    tasks = [(candidate, opponent, side, s) for s in seeds for side in selfplay.PLAYERS]
    with selfplay._pool(4, HORIZON, separation_m=0.5, anti_stalling=True) as pool:
        games = pool.map(selfplay.play_game, tasks)
    valid = [selfplay.valid_result(x) for x in games]
    records = [
        {
            "first": "candidate",
            "second": "opponent",
            "seed": task[3],
            "outcome": game["utility"] if ok else None,
        }
        for task, game, ok in zip(tasks, games, valid, strict=True)
    ]
    samples = _utilities(records, "candidate", "opponent")
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


def brief(result):
    return {k: v for k, v in result.items() if k != "results"}


def main(out):
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t, g = protocol["training"], protocol["gate"]
    gen4 = np.array(json.loads(SUMMARY.read_text())["archive"]["gen4"])
    v18 = json.loads(V18.read_text())
    v18_params = np.array(v18["final_params"])
    opponent = ("params", gen4)
    began = time.monotonic()
    report = {"protocol_sha256": hashlib.sha256(raw).hexdigest()}
    with motor_gate.pool(t["workers"], 0.5, HORIZON) as pool:
        report["v0018_gate"] = motor_gate.gate(
            pool,
            ("group_pd", v18_params, g["eval_noise"]),
            opponent,
            g["seeds"],
            HORIZON,
        )
        report["v0018_reproduced"] = [
            r["stand_ticks"] for r in report["v0018_gate"]["results"]
        ] == v18["candidate_gate"]["stand_ticks"]
        print("v0018_gate", brief(report["v0018_gate"]), report["v0018_reproduced"])
        params, history = motor_gate.train(
            pool,
            v18_params,
            opponent,
            np.random.default_rng(t["seed"]),
            kind="group_pd",
            iterations=t["iterations"],
            pairs=t["pairs"],
            sigma=t["sigma"],
            lr=t["lr"],
            horizon_ticks=HORIZON,
            strike_bonus=1.0,
            stay_up_ticks=500,
            log=lambda row: print(json.dumps(row), flush=True),
        )
        report["history"] = history
        report["final_params"] = params.tolist()
        report["candidate_gate"] = motor_gate.gate(
            pool, ("group_pd", params, g["eval_noise"]), opponent, g["seeds"], HORIZON
        )
    stand = {
        k: [r["stand_ticks"] for r in report[k]["results"]]
        for k in ("v0018_gate", "candidate_gate")
    }
    report["sign_test_vs_v0018"] = sign_test(
        stand["candidate_gate"], stand["v0018_gate"]
    )
    print("candidate_gate", brief(report["candidate_gate"]), flush=True)

    champion = (
        "punch",
        punch_params(gen4, onset=-3.0, duration=0.0, amplitude=2.0),
        NOISE,
    )
    v16 = ("group", np.array(json.loads(V16.read_text())["final_params"]), NOISE)
    v19_spec, v18_spec = ("group_pd", params, NOISE), ("group_pd", v18_params, NOISE)
    primary_seeds = protocol["primary"]["head_to_head"]["seeds"]
    secondary_seeds = list(range(6000, 6024))
    report["primary_vs_v0013"] = head_to_head(v19_spec, champion, primary_seeds)
    print("primary_vs_v0013", brief(report["primary_vs_v0013"]), flush=True)
    report["v0018_vs_v0013_same_seeds"] = head_to_head(
        v18_spec, champion, primary_seeds
    )
    report["v0019_vs_v0016"] = head_to_head(v19_spec, v16, secondary_seeds)
    report["v0018_vs_v0016"] = head_to_head(v18_spec, v16, secondary_seeds)
    a, b = report["v0019_vs_v0016"]["mean"], report["v0018_vs_v0016"]["mean"]
    report["v0016_improvement"] = None if a is None or b is None else a - b
    report["level_up"] = bool(report["primary_vs_v0013"]["passed"])
    report["wall_s"] = round(time.monotonic() - began, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    for k in ("v0018_vs_v0013_same_seeds", "v0019_vs_v0016", "v0018_vs_v0016"):
        print(k, brief(report[k]))
    print("v0016_improvement", report["v0016_improvement"])
    print("sign_test_vs_v0018", report["sign_test_vs_v0018"])
    print("level_up", report["level_up"], "wall_s", report["wall_s"])
    print("protocol", report["protocol_sha256"])


if __name__ == "__main__":
    main(sys.argv[1])
