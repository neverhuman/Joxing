"""v0.0.17 per-group feedback gate as preregistered in protocols/v0.0.17.json.

    PYTHONPATH=. nice -n 10 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0022-group-feedback-gate.py \
        runs/grokbot/group-feedback-gate-0022
"""

import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np

from grokbot import motor_gate, selfplay
from grokbot.motor_primitives import group_feedback_params, punch_params
from grokbot.promotion import _utilities, hoeffding_lower

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.17.json")
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")
V16 = pathlib.Path("grokbot/redteam/data/cycle-0020-group-gate-summary.json")
HORIZON = 3000


def sign_test(candidate, reference):
    """One-sided exact sign test that ``candidate`` stands longer (ties dropped)."""
    better = sum(c > r for c, r in zip(candidate, reference, strict=True))
    n = sum(c != r for c, r in zip(candidate, reference, strict=True))
    p = sum(math.comb(n, k) for k in range(better, n + 1)) / 2**n if n else 1.0
    return {"better": better, "pairs": n, "one_sided_p": p}


def head_to_head(protocol, params):
    h, g = protocol["head_to_head"], protocol["gate"]
    gen4 = np.array(json.loads(SUMMARY.read_text())["archive"]["gen4"])
    champion = punch_params(gen4, onset=-3.0, duration=0.0, amplitude=2.0)
    spec = ("group_feedback", params, g["eval_noise"])
    other = ("punch", champion, g["eval_noise"])
    tasks = [(spec, other, side, s) for s in h["seeds"] for side in selfplay.PLAYERS]
    with selfplay._pool(4, HORIZON, separation_m=0.5, anti_stalling=True) as pool:
        games = pool.map(selfplay.play_game, tasks)
    records = [
        {
            "first": "candidate",
            "second": "v0.0.13",
            "seed": task[3],
            "outcome": game["utility"] if selfplay.valid_result(game) else None,
        }
        for task, game in zip(tasks, games, strict=True)
    ]
    samples = _utilities(records, "candidate", "v0.0.13")
    mean = float(np.mean(samples)) if samples else 0.0
    valid = sum(r["outcome"] is not None for r in records)
    lower = hoeffding_lower(mean, len(samples), 0.95)
    return {
        "games": len(games),
        "valid_games": valid,
        "no_contest": sum(bool(x["no_contest"]) for x in games),
        "blocked": sum(bool(x["blocked"]) for x in games),
        "invalid": sum(x["invalid"] is not None for x in games),
        "paired_samples": len(samples),
        "mean": mean if samples else None,
        "lower95": lower,
        "passed": bool(valid > 0 and lower > 0),
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
    v16 = json.loads(V16.read_text())
    v16_params = np.array(v16["final_params"])
    opponent = ("params", gen4)
    began = time.monotonic()
    report = {"protocol_sha256": hashlib.sha256(raw).hexdigest()}
    with motor_gate.pool(t["workers"], 0.5, HORIZON) as pool:
        for name, spec in (
            ("baseline_gate", ("params", gen4, g["eval_noise"])),
            ("v0016_gate", ("group", v16_params, g["eval_noise"])),
        ):
            report[name] = motor_gate.gate(pool, spec, opponent, g["seeds"], HORIZON)
            print(name, {k: v for k, v in report[name].items() if k != "results"})
        report["v0016_reproduced"] = [
            r["stand_ticks"] for r in report["v0016_gate"]["results"]
        ] == v16["candidate_gate"]["stand_ticks"]
        print("v0.0.16 gate reproduced:", report["v0016_reproduced"], flush=True)
        params, history = motor_gate.train(
            pool,
            group_feedback_params(v16_params),
            opponent,
            np.random.default_rng(t["seed"]),
            kind="group_feedback",
            iterations=t["iterations"],
            pairs=t["pairs"],
            sigma=t["sigma"],
            lr=t["lr"],
            horizon_ticks=HORIZON,
            strike_bonus=1.0,
            log=lambda row: print(json.dumps(row), flush=True),
        )
        report["history"] = history
        report["final_params"] = params.tolist()
        report["candidate_gate"] = motor_gate.gate(
            pool,
            ("group_feedback", params, g["eval_noise"]),
            opponent,
            g["seeds"],
            HORIZON,
        )
    stand = {
        k: [r["stand_ticks"] for r in report[k]["results"]]
        for k in ("baseline_gate", "v0016_gate", "candidate_gate")
    }
    report["sign_test_vs_v0016"] = sign_test(
        stand["candidate_gate"], stand["v0016_gate"]
    )
    report["sign_test_vs_v0016"]["passed"] = (
        report["sign_test_vs_v0016"]["one_sided_p"] < 0.05
    )
    gains = np.clip(
        [(c - b) / 1000 for c, b in zip(stand["candidate_gate"], stand["v0016_gate"])],
        -3,
        3,
    )
    report["stand_gain_vs_v0016_s"] = {
        "mean": float(gains.mean()),
        "lower95": float(gains.mean() - 6 * np.sqrt(np.log(20) / (2 * len(gains)))),
        "pairs": len(gains),
    }
    c = report["candidate_gate"]
    report["primary_motor_gate_passed"] = bool(
        c["stance_pass_rate"] >= 0.5 and c["reach_pass_rate"] >= 0.5
    )
    print("candidate", {k: v for k, v in c.items() if k != "results"}, flush=True)
    if report["primary_motor_gate_passed"]:
        report["head_to_head"] = head_to_head(protocol, params)
        h2h = {k: v for k, v in report["head_to_head"].items() if k != "results"}
        print("head_to_head", h2h)
    else:
        report["head_to_head"] = {
            "run": False,
            "passed": False,
            "reason": "primary motor gate not passed",
        }
    report["level_up"] = bool(
        report["primary_motor_gate_passed"] and report["head_to_head"]["passed"]
    )
    report["wall_s"] = round(time.monotonic() - began, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("sign_test_vs_v0016", report["sign_test_vs_v0016"])
    print("stand_gain_vs_v0016_s", report["stand_gain_vs_v0016_s"])
    print("level_up", report["level_up"], "wall_s", report["wall_s"])
    print("protocol", report["protocol_sha256"])


if __name__ == "__main__":
    main(sys.argv[1])
