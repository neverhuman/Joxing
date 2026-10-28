"""v0.0.21 compute-monotonicity check (owner 13:10, agent adequacy).

Registered in protocols/v0.0.21-monotonicity-addendum.json before any v0.0.21
result was read. More training must not lose to less: v0.0.21@48 vs @16 and
@16 vs @0 (= v0.0.19), each with >= 40/48 valid games and a Wilson 95% lower
bound on the more-trained side's win rate above 0.55. The @16 checkpoint is
re-trained from the same seed and must reproduce the main run's history.

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0030-compute-monotonicity.py \
        runs/grokbot/mirror-opponent-gate-0029/report.json \
        runs/grokbot/compute-monotonicity-0030
"""

import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

from grokbot import motor_gate

PROTOCOL_PATH = pathlib.Path(
    "grokbot/redteam/protocols/v0.0.21-monotonicity-addendum.json"
)
V21_PROTOCOL = pathlib.Path("grokbot/redteam/protocols/v0.0.21.json")
V19 = pathlib.Path("grokbot/redteam/data/cycle-0025-hit-stay-gate-summary.json")
GATE = pathlib.Path("grokbot/redteam/probes/cycle-0025-hit-stay-gate.py")
HORIZON = 3000
NOISE = 0.05
N_ITERATIONS = 16
SEEDS = list(range(12000, 12024))
MIN_VALID = 40


def _v0019_probe():
    spec = importlib.util.spec_from_file_location("v0019_probe", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verdict(result):
    if result["valid_games"] < MIN_VALID:
        return "ADEQUACY_UNMEASURABLE"
    low = (result["win_rate_wilson95"] or [0.0])[0]
    return "PASS" if low > 0.55 else "FAIL"


def main(run_report, out):
    raw = PROTOCOL_PATH.read_bytes()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    gate = _v0019_probe()
    t = json.loads(V21_PROTOCOL.read_text())["training"]
    run = json.loads(pathlib.Path(run_report).read_text())
    v19 = np.array(json.loads(V19.read_text())["final_params"])
    mirror = ("group_pd", v19, NOISE)
    began = time.monotonic()
    report = {"protocol_sha256": hashlib.sha256(raw).hexdigest()}
    with motor_gate.pool(t["workers"], 0.5, HORIZON) as pool:
        params16, history16 = motor_gate.train(
            pool,
            v19,
            mirror,
            np.random.default_rng(t["seed"]),
            kind="group_pd",
            iterations=N_ITERATIONS,
            pairs=t["pairs"],
            sigma=t["sigma"],
            lr=t["lr"],
            horizon_ticks=HORIZON,
            strike_bonus=1.0,
            stay_up_ticks=500,
            log=lambda row: print(json.dumps(row), flush=True),
        )
    report["reproduced"] = history16 == run["history"][:N_ITERATIONS]
    report["params16"] = params16.tolist()
    print("reproduced", report["reproduced"], flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    if not report["reproduced"]:
        report["decision"] = "REPRODUCTION_FAILED"
    else:
        at48 = ("group_pd", np.array(run["final_params"]), NOISE)
        at16 = ("group_pd", params16, NOISE)
        for key, more, less in (("3N_vs_N", at48, at16), ("N_vs_0", at16, mirror)):
            report[key] = gate.head_to_head(more, less, SEEDS)
            report[key]["verdict"] = verdict(report[key])
            print(key, report[key]["verdict"], gate.brief(report[key]), flush=True)
            (out / "report.json").write_text(json.dumps(report, indent=1))
        both = all(report[k]["verdict"] == "PASS" for k in ("3N_vs_N", "N_vs_0"))
        report["decision"] = "MONOTONE" if both else "NOT_SHOWN_MONOTONE"
    report["wall_s"] = round(time.monotonic() - began, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("decision", report["decision"], "wall_s", report["wall_s"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
