"""v0.0.21 training-opponent run as preregistered in protocols/v0.0.21.json.

One lever from v0.0.19: the solo hit-and-stay-up fitness is trained against
the frozen v0.0.19 instead of the gen4 posture. The head-to-head statistics are
the v0.0.19 probe's ``head_to_head``, imported unchanged. Validity is checked
first (owner 12:05): fewer than ``min_valid_games`` valid primary games is a
loud VALIDITY_FAILED (non-zero exit), never a null or a pass.

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0029-mirror-opponent-gate.py \
        runs/grokbot/mirror-opponent-gate-0029
"""

import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

from grokbot import motor_gate
from grokbot.motor_primitives import punch_params

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.21.json")
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
    t, g = protocol["training"], protocol["gate"]
    gate = _v0019_probe()
    gen4 = np.array(json.loads(SUMMARY.read_text())["archive"]["gen4"])
    v19_summary = json.loads(V19.read_text())
    v19 = np.array(v19_summary["final_params"])
    v16 = np.array(json.loads(V16.read_text())["final_params"])
    champion13 = punch_params(gen4, onset=-3.0, duration=0.0, amplitude=2.0)
    posture = ("params", gen4)
    mirror = ("group_pd", v19, NOISE)  # the one change: the training opponent
    began = time.monotonic()
    report = {"protocol_sha256": hashlib.sha256(raw).hexdigest()}
    with motor_gate.pool(t["workers"], 0.5, HORIZON) as pool:
        report["v0019_gate"] = motor_gate.gate(
            pool, ("group_pd", v19, g["eval_noise"]), posture, g["seeds"], HORIZON
        )
        report["v0019_reproduced"] = [
            r["stand_ticks"] for r in report["v0019_gate"]["results"]
        ] == v19_summary["candidate_gate"]["stand_ticks"]
        print(
            "v0019_gate", gate.brief(report["v0019_gate"]), report["v0019_reproduced"]
        )
        params, history = motor_gate.train(
            pool,
            v19,
            mirror,
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
            pool, ("group_pd", params, g["eval_noise"]), posture, g["seeds"], HORIZON
        )
    print("candidate_gate", gate.brief(report["candidate_gate"]), flush=True)
    save(out, report, complete=False)

    candidate = ("group_pd", params, NOISE)
    primary_seeds = protocol["primary"]["head_to_head"]["seeds"]
    secondary_seeds = protocol["secondary_reported_not_gating"]["seeds"]
    stages = [
        ("primary_vs_v0019", candidate, mirror, primary_seeds),
        ("v0019_mirror_same_seeds", mirror, mirror, primary_seeds),
        ("vs_v0016", candidate, ("group", v16, NOISE), secondary_seeds),
        ("vs_v0013", candidate, ("punch", champion13, NOISE), secondary_seeds),
    ]
    for key, spec, other, seeds in stages:
        report[key] = gate.head_to_head(spec, other, seeds)
        print(key, gate.brief(report[key]), flush=True)
        save(out, report, complete=False)
    primary = report["primary_vs_v0019"]
    need = protocol["primary"]["min_valid_games"]
    report["validity_precondition"] = {
        "required_valid_games": need,
        "valid_games": primary["valid_games"],
        "passed": primary["valid_games"] >= need,
    }
    checks = primary["checks"]
    if not report["validity_precondition"]["passed"]:
        report["verdict"] = "VALIDITY_FAILED"
    elif checks["b_lower95_above_zero"] and checks["c_half_of_wins_with_hits"]:
        report["verdict"] = "LEVEL_UP"
    else:
        report["verdict"] = "CRITERION_NOT_MET"
    report["level_up"] = report["verdict"] == "LEVEL_UP"
    report["wall_s"] = round(time.monotonic() - began, 1)
    save(out, report, complete=True)
    print("verdict", report["verdict"], "level_up", report["level_up"])
    print("wall_s", report["wall_s"], "protocol", report["protocol_sha256"])
    if report["verdict"] == "VALIDITY_FAILED":
        sys.exit(
            f"VALIDITY_FAILED: {primary['valid_games']}/{primary['games']} valid "
            f"primary games < {need}; the win-rate criteria were not evaluated"
        )


if __name__ == "__main__":
    main(sys.argv[1])
