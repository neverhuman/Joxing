"""v0.0.18 group-PD gate + Elo gate as preregistered in protocols/v0.0.18.json.

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0023-group-pd-gate.py \
        runs/grokbot/group-pd-gate-0023
"""

import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np

from grokbot import motor_gate, selfplay
from grokbot.motor_primitives import group_pd_params
from grokbot.promotion import _utilities, hoeffding_lower

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.18.json")
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")
V16 = pathlib.Path("grokbot/redteam/data/cycle-0020-group-gate-summary.json")
V17 = pathlib.Path("grokbot/redteam/data/cycle-0022-group-feedback-gate-summary.json")
HORIZON = 3000


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


def head_to_head(protocol, params):
    h, g = protocol["head_to_head"], protocol["gate"]
    v16 = np.array(json.loads(V16.read_text())["final_params"])
    spec = ("group_pd", params, g["eval_noise"])
    other = ("group", v16, g["eval_noise"])
    tasks = [(spec, other, side, s) for s in h["seeds"] for side in selfplay.PLAYERS]
    with selfplay._pool(4, HORIZON, separation_m=0.5, anti_stalling=True) as pool:
        games = pool.map(selfplay.play_game, tasks)
    valid = [selfplay.valid_result(x) for x in games]
    records = [
        {
            "first": "candidate",
            "second": "v0.0.16",
            "seed": task[3],
            "outcome": game["utility"] if ok else None,
        }
        for task, game, ok in zip(tasks, games, valid, strict=True)
    ]
    samples = _utilities(records, "candidate", "v0.0.16")
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


def main(out):
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t, g = protocol["training"], protocol["gate"]
    gen4 = np.array(json.loads(SUMMARY.read_text())["archive"]["gen4"])
    v17 = json.loads(V17.read_text())
    v17_params = np.array(v17["final_params"])
    opponent = ("params", gen4)
    began = time.monotonic()
    report = {"protocol_sha256": hashlib.sha256(raw).hexdigest()}
    with motor_gate.pool(t["workers"], 0.5, HORIZON) as pool:
        report["v0017_gate"] = motor_gate.gate(
            pool,
            ("group_feedback", v17_params, g["eval_noise"]),
            opponent,
            g["seeds"],
            HORIZON,
        )
        report["v0017_reproduced"] = [
            r["stand_ticks"] for r in report["v0017_gate"]["results"]
        ] == v17["candidate_gate"]["stand_ticks"]
        summary = {k: v for k, v in report["v0017_gate"].items() if k != "results"}
        print(
            "v0017_gate", summary, "reproduced", report["v0017_reproduced"], flush=True
        )
        params, history = motor_gate.train(
            pool,
            group_pd_params(v17_params),
            opponent,
            np.random.default_rng(t["seed"]),
            kind="group_pd",
            iterations=t["iterations"],
            pairs=t["pairs"],
            sigma=t["sigma"],
            lr=t["lr"],
            horizon_ticks=HORIZON,
            strike_bonus=1.0,
            strike_key="scoring_strike_ticks",
            log=lambda row: print(json.dumps(row), flush=True),
        )
        report["history"] = history
        report["final_params"] = params.tolist()
        report["candidate_gate"] = motor_gate.gate(
            pool, ("group_pd", params, g["eval_noise"]), opponent, g["seeds"], HORIZON
        )
    stand = {
        k: [r["stand_ticks"] for r in report[k]["results"]]
        for k in ("v0017_gate", "candidate_gate")
    }
    report["sign_test_vs_v0017"] = sign_test(
        stand["candidate_gate"], stand["v0017_gate"]
    )
    report["sign_test_vs_v0017"]["passed"] = (
        report["sign_test_vs_v0017"]["one_sided_p"] < 0.05
    )
    c = report["candidate_gate"]
    report["primary_motor_gate_passed"] = bool(
        c["stance_pass_rate"] >= 0.5 and c["reach_pass_rate"] >= 0.5
    )
    print("candidate", {k: v for k, v in c.items() if k != "results"}, flush=True)
    report["head_to_head"] = head_to_head(protocol, params)
    print(
        "head_to_head",
        {k: v for k, v in report["head_to_head"].items() if k != "results"},
    )
    report["level_up"] = bool(
        report["primary_motor_gate_passed"] and report["head_to_head"]["passed"]
    )
    report["wall_s"] = round(time.monotonic() - began, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("sign_test_vs_v0017", report["sign_test_vs_v0017"])
    print("level_up", report["level_up"], "wall_s", report["wall_s"])
    print("protocol", report["protocol_sha256"])


if __name__ == "__main__":
    main(sys.argv[1])
