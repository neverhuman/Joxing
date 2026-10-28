"""v0.0.16 group-controller motor gate as preregistered in protocols/v0.0.16.json.

    PYTHONPATH=. nice -n 10 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0020-group-gate.py \
        runs/grokbot/group-gate-0020
"""

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

from grokbot import motor_gate
from grokbot.motor_primitives import group_params

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.16.json")
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")
HORIZON = 3000


def main(out):
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t, g = protocol["training"], protocol["gate"]
    gen4 = np.array(json.loads(SUMMARY.read_text())["archive"]["gen4"])
    opponent = ("params", gen4)
    began = time.monotonic()
    report = {"protocol_sha256": hashlib.sha256(raw).hexdigest()}
    with motor_gate.pool(t["workers"], 0.5, HORIZON) as pool:
        report["baseline_gate"] = motor_gate.gate(
            pool, ("params", gen4, g["eval_noise"]), opponent, g["seeds"], HORIZON
        )
        print(
            "baseline",
            {k: v for k, v in report["baseline_gate"].items() if k != "results"},
            flush=True,
        )
        params, history = motor_gate.train(
            pool,
            group_params(gen4, onset=-3.0, duration=0.0, amplitude=2.0),
            opponent,
            np.random.default_rng(t["seed"]),
            kind="group",
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
            pool, ("group", params, g["eval_noise"]), opponent, g["seeds"], HORIZON
        )
    base = report["baseline_gate"]["results"]
    cand = report["candidate_gate"]["results"]
    gains = np.clip(
        [(c["stand_ticks"] - b["stand_ticks"]) / 1000 for c, b in zip(cand, base)],
        -3,
        3,
    )
    n = len(gains)
    report["stand_gain_s"] = {
        "mean": float(gains.mean()),
        "lower95": float(gains.mean() - 6 * np.sqrt(np.log(20) / (2 * n))),
        "pairs": n,
    }
    c = report["candidate_gate"]
    report["primary_motor_gate_passed"] = bool(
        c["stance_pass_rate"] >= 0.5 and c["reach_pass_rate"] >= 0.5
    )
    report["wall_s"] = round(time.monotonic() - began, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("candidate", {k: v for k, v in c.items() if k != "results"})
    print(
        "stand_gain_s",
        report["stand_gain_s"],
        "passed",
        report["primary_motor_gate_passed"],
    )
    print("wall_s", report["wall_s"], "protocol", report["protocol_sha256"])


if __name__ == "__main__":
    main(sys.argv[1])
