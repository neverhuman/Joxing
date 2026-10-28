"""v0.0.20 self-play league as preregistered in protocols/v0.0.20.json.

One lever from v0.0.19: two-player self-play games (match outcome only)
instead of solo motor trials. The a/b/c statistics are the v0.0.19 probe's
``head_to_head``, imported unchanged.

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0027-selfplay-league.py \
        runs/grokbot/selfplay-league-0027
"""

import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

from grokbot import motor_gate, selfplay
from grokbot.motor_primitives import punch_params

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.20.json")
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")
V16 = pathlib.Path("grokbot/redteam/data/cycle-0020-group-gate-summary.json")
V19 = pathlib.Path("grokbot/redteam/data/cycle-0025-hit-stay-gate-summary.json")
GATE = pathlib.Path("grokbot/redteam/probes/cycle-0025-hit-stay-gate.py")
HORIZON = 3000
NOISE = 0.05


def _v0019_probe():
    spec = importlib.util.spec_from_file_location("v0019_probe", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(out, report, complete):
    """Checkpoint after every stage so an exception never discards results."""
    (out / "report.json").write_text(
        json.dumps(report | {"complete": complete}, indent=1)
    )


def main(out):
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    lg = protocol["league"]
    gate = _v0019_probe()
    gen4 = np.array(json.loads(SUMMARY.read_text())["archive"]["gen4"])
    v19 = np.array(json.loads(V19.read_text())["final_params"])
    v16 = np.array(json.loads(V16.read_text())["final_params"])
    champion13 = punch_params(gen4, onset=-3.0, duration=0.0, amplitude=2.0)
    frozen = {
        "v0019": ("group_pd", v19),
        "v0016": ("group", v16),
        "v0013": ("punch", champion13),
    }
    began = time.monotonic()
    report = {"protocol_sha256": hashlib.sha256(raw).hexdigest()}
    report["league"] = selfplay.run_league(
        generations=lg["generations"],
        iterations=lg["iterations"],
        pairs=lg["pairs"],
        sigma=lg["sigma"],
        lr=lg["lr"],
        eval_seeds=lg["eval_seeds"],
        eval_noise=lg["eval_noise"],
        workers=lg["workers"],
        horizon_ticks=HORIZON,
        seed=lg["seed"],
        panel_eval=lg["panel_eval"],
        separation_m=0.5,
        anti_stalling=True,
        min_engaged_share=0.25,
        init_params=v19,
        policy_kind="group_pd",
        frozen=frozen,
        require_valid=True,
        log=lambda row: print(json.dumps(row), flush=True),
    )
    save(out, report, complete=False)
    league = report["league"]
    name = league["champion"]
    params = np.array(league["archive"][name])
    last = f"gen{lg['generations']}"
    print("champion", name, flush=True)

    candidate = ("group_pd", params, NOISE)
    primary_seeds = protocol["primary"]["head_to_head"]["seeds"]
    secondary_seeds = list(range(8000, 8024))
    stages = [
        ("primary_vs_v0019", candidate, ("group_pd", v19, NOISE), primary_seeds),
        ("vs_v0016", candidate, ("group", v16, NOISE), secondary_seeds),
        ("vs_v0013", candidate, ("punch", champion13, NOISE), secondary_seeds),
    ]
    if name != last:
        final = ("group_pd", np.array(league["archive"][last]), NOISE)
        stages.append(
            (
                "final_generation_vs_v0019",
                final,
                ("group_pd", v19, NOISE),
                primary_seeds,
            )
        )
    for key, spec, other, seeds in stages:
        report[key] = gate.head_to_head(spec, other, seeds)
        print(key, gate.brief(report[key]), flush=True)
        save(out, report, complete=False)
    with motor_gate.pool(lg["workers"], 0.5, HORIZON) as pool:
        report["champion_motor_gate"] = motor_gate.gate(
            pool, candidate, ("params", gen4), list(range(1000, 1024)), HORIZON
        )
    print("champion_motor_gate", gate.brief(report["champion_motor_gate"]))
    report["champion"] = name
    report["level_up"] = bool(report["primary_vs_v0019"]["passed"])
    report["wall_s"] = round(time.monotonic() - began, 1)
    save(out, report, complete=True)
    print("level_up", report["level_up"], "wall_s", report["wall_s"])
    print("protocol", report["protocol_sha256"])


if __name__ == "__main__":
    main(sys.argv[1])
