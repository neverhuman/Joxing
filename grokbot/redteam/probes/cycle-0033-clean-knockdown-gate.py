"""v0.0.23 clean-knockdown objective as preregistered in protocols/v0.0.23.json.

One lever from the v0.0.22 recipe: the bonus needs a CLEAN knockdown (a head/torso
hit, then an opponent fall the ending attributes, not one left unresolved by body
contact). Checked in order, failing loud (non-zero exit) on validity or
decisiveness: the primary vs frozen v0.0.19, the v0.0.13 hard gate, and
more-training-beats-less (@48 vs @16, @16 vs v0.0.19).

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0033-clean-knockdown-gate.py \
        runs/grokbot/clean-knockdown-gate-0033
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

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.23.json")
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")
V16 = pathlib.Path("grokbot/redteam/data/cycle-0020-group-gate-summary.json")
V19 = pathlib.Path("grokbot/redteam/data/cycle-0025-hit-stay-gate-summary.json")
GATE = pathlib.Path("grokbot/redteam/probes/cycle-0025-hit-stay-gate.py")
MONO = pathlib.Path("grokbot/redteam/probes/cycle-0030-compute-monotonicity.py")
HORIZON = 3000
NOISE = 0.05


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(out, report, complete):
    """Checkpoint after every stage so an exception never discards results."""
    (out / "report.json").write_text(
        json.dumps(report | {"complete": complete}, indent=1)
    )


def primary_verdict(result, need_valid, need_decisive):
    if result["valid_games"] < need_valid:
        return "VALIDITY_FAILED"
    if result["wins"] + result["losses"] < need_decisive:
        return "DECISIVE_FAILED"
    checks = result["checks"]
    if checks["b_lower95_above_zero"] and checks["c_half_of_wins_with_hits"]:
        return "PRIMARY_PASSED"
    return "CRITERION_NOT_MET"


def main(out):
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t, g = protocol["training"], protocol["gate"]
    gate, mono = _load(GATE, "v0019_probe"), _load(MONO, "v0021_mono")
    gen4 = np.array(json.loads(SUMMARY.read_text())["archive"]["gen4"])
    v19_summary = json.loads(V19.read_text())
    v19 = np.array(v19_summary["final_params"])
    v16 = np.array(json.loads(V16.read_text())["final_params"])
    champion13 = punch_params(gen4, onset=-3.0, duration=0.0, amplitude=2.0)
    posture = ("params", gen4)
    mirror = ("group_pd", v19, NOISE)
    began = time.monotonic()
    report = {"protocol_sha256": hashlib.sha256(raw).hexdigest()}
    rng = np.random.default_rng(t["seed"])
    common = dict(
        kind="group_pd",
        pairs=t["pairs"],
        sigma=t["sigma"],
        lr=t["lr"],
        horizon_ticks=HORIZON,
        strike_bonus=1.0,
        clean_knockdown=True,
        log=lambda row: print(json.dumps(row), flush=True),
    )
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
        # 16 + 32 iterations on one generator == one 48-iteration call.
        at16, first = motor_gate.train(
            pool, v19, mirror, rng, iterations=t["checkpoint_at"], **common
        )
        params, rest = motor_gate.train(
            pool,
            at16,
            mirror,
            rng,
            iterations=t["iterations"] - t["checkpoint_at"],
            **common,
        )
        for k, row in enumerate(rest):
            row["iteration"] = t["checkpoint_at"] + k
        report["history"] = first + rest
        report["params_at_16"] = at16.tolist()
        report["final_params"] = params.tolist()
        report["candidate_gate"] = motor_gate.gate(
            pool, ("group_pd", params, g["eval_noise"]), posture, g["seeds"], HORIZON
        )
    print("candidate_gate", gate.brief(report["candidate_gate"]), flush=True)
    save(out, report, complete=False)

    def play(key, more, less, seeds):
        report[key] = gate.head_to_head(more, less, seeds)
        report[key]["by_seat"] = mono.by_seat(report[key])
        print(key, gate.brief(report[key]), flush=True)
        save(out, report, complete=False)
        return report[key]

    candidate = ("group_pd", params, NOISE)
    p = protocol["primary"]
    primary = play("primary_vs_v0019", candidate, mirror, p["head_to_head"]["seeds"])
    report["primary_verdict"] = primary_verdict(
        primary, p["min_valid_games"], p["min_decisive_games"]
    )
    hg = protocol["hard_gate_v0013"]
    v13 = play(
        "hard_gate_vs_v0013", candidate, ("punch", champion13, NOISE), hg["seeds"]
    )
    report["hard_gate_passed"] = v13["valid_games"] >= 40
    m = protocol["monotonicity"]
    at16_spec = ("group_pd", at16, NOISE)
    for key, more, less in (
        ("mono_48_vs_16", candidate, at16_spec),
        ("mono_16_vs_0", at16_spec, mirror),
    ):
        result = play(key, more, less, m["seeds"])
        result["verdict"] = mono.verdict(result)
        save(out, report, complete=False)
    report["monotone"] = all(
        report[k]["verdict"] == "PASS" for k in ("mono_48_vs_16", "mono_16_vs_0")
    )
    play("v0019_mirror_same_seeds", mirror, mirror, p["head_to_head"]["seeds"])
    play("vs_v0016", candidate, ("group", v16, NOISE), hg["seeds"])
    report["level_up"] = bool(
        report["primary_verdict"] == "PRIMARY_PASSED"
        and report["hard_gate_passed"]
        and report["monotone"]
    )
    report["wall_s"] = round(time.monotonic() - began, 1)
    save(out, report, complete=True)
    print(
        "primary",
        report["primary_verdict"],
        "hard_gate",
        report["hard_gate_passed"],
        "monotone",
        report["monotone"],
        "level_up",
        report["level_up"],
    )
    print("wall_s", report["wall_s"], "protocol", report["protocol_sha256"])
    if report["primary_verdict"] in ("VALIDITY_FAILED", "DECISIVE_FAILED"):
        sys.exit(
            f"{report['primary_verdict']}: {primary['valid_games']}/{primary['games']} "
            f"valid, {primary['wins'] + primary['losses']}/{primary['games']} "
            "decisive primary games"
        )


if __name__ == "__main__":
    main(sys.argv[1])
