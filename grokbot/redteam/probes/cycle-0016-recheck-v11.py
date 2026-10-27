"""Re-run the champion gate on the v0.0.11 checkpoints under v0.0.12 rules.

Every v0.0.11 checkpoint (gen0..gen6) plays every other and the scripted panel
at the 0.80 m range reset with anti-stalling, 6 noisy side-swapped seed pairs
each. Prints engagement and no-contest shares and the v0.0.11 lineage's gate
decisions with the preregistered engagement gate.

    PYTHONPATH=. nice -n 10 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0016-recheck-v11.py
"""

import json
import pathlib

import numpy as np

from grokbot import selfplay
from grokbot.arena import SCRIPTED_PANEL
from grokbot.promotion import GateConfig, decide

PROTOCOL = json.loads(
    pathlib.Path("grokbot/redteam/protocols/v0.0.12.json").read_text()
)
SUMMARY = pathlib.Path("grokbot/redteam/data/cycle-0014-league-summary.json")


def main():
    archive = {
        k: np.array(v) for k, v in json.loads(SUMMARY.read_text())["archive"].items()
    }
    names = sorted(archive)
    seeds = PROTOCOL["evaluation"]["eval_seeds"]
    noise = PROTOCOL["evaluation"]["eval_noise"]
    tasks, meta = [], []
    for i, cand in enumerate(names):
        for (c, o, side, s), task in selfplay.evaluation_tasks(
            cand, archive, names[:i] + list(SCRIPTED_PANEL), seeds, noise
        ):
            tasks.append(task)
            meta.append((c, o, side, s))
    with selfplay._pool(
        4,
        PROTOCOL["training"]["horizon_ticks"],
        PROTOCOL["reset"]["separation_m"],
        anti_stalling=True,
    ) as pool:
        outcomes = pool.map(selfplay.play_game, tasks)
    records = []
    for (c, o, side, s), out in zip(meta, outcomes, strict=True):
        records.append(
            {
                "first": c,
                "second": o,
                "first_side": side,
                "seed": s,
                "outcome": None
                if out["blocked"] or out["no_contest"]
                else int(out["utility"]),
                "invalid_reason": "no contest: no engagement before the ending"
                if out["no_contest"]
                else "blocked: unresolved loss attribution"
                if out["blocked"]
                else None,
                "fault_player": None,
                "losses": [],
                "ending": None,
                "engaged": {
                    c: out["strike_ticks"] > 0,
                    o: out["opponent_strike_ticks"] > 0,
                },
                "contact_ticks": out["contact_ticks"],
            }
        )
    contacts = sum(r["contact_ticks"] for r in records)
    print(f"games {len(records)}  contact ticks {contacts}")
    for name in names:
        own = [r for r in records if name in (r["first"], r["second"])]
        engaged = np.mean([r["engaged"][name] for r in own])
        stalled = np.mean(
            [(r["invalid_reason"] or "").startswith("no contest") for r in own]
        )
        print(f"{name}: engaged {engaged:.3f}  no-contest {stalled:.3f}")
    config = GateConfig(
        min_valid=len(seeds), min_engaged_share=PROTOCOL["gates"]["min_engaged_share"]
    )
    for cand, champ in [("gen1", "gen0"), ("gen2", "gen1"), ("gen4", "gen2")]:
        d = decide(records, cand, champ, [champ, *SCRIPTED_PANEL], config)
        print(
            f"gate {cand} over {champ}: promote={d.promote} reasons={list(d.reasons)}"
        )


if __name__ == "__main__":
    main()
