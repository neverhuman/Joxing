"""Round-robin ladder of every frozen version, v0.0.11 to v0.0.20.

Reporting addendum (owner 10:05/10:30), registered in
protocols/v0.0.20-ladder-addendum.json before any v0.0.20 output was read:
Elo on one scale, intransitive triads (non-transitive cycling) and the
degeneracy of the games (stall rate, game length).

    PYTHONPATH=. nice -n 19 uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0028-ladder.py \
        runs/grokbot/selfplay-league-0027/report.json runs/grokbot/ladder-0028
"""

import hashlib
import itertools
import json
import pathlib
import sys
import time

import numpy as np

from grokbot import selfplay
from grokbot.league import Game, Ledger

PROTOCOL_PATH = pathlib.Path("grokbot/redteam/protocols/v0.0.20-ladder-addendum.json")
DATA = pathlib.Path("grokbot/redteam/data")
SEEDS = range(9000, 9006)
HORIZON = 3000
NOISE = 0.05


def players(league_report):
    def archive(name):
        d = json.loads((DATA / name).read_text())
        return np.array(d["archive"][d["champion"]])

    def final(name):
        return np.array(json.loads((DATA / name).read_text())["final_params"])

    league = json.loads(pathlib.Path(league_report).read_text())["league"]
    return {
        "v0.0.11": ("params", archive("cycle-0014-league-summary.json")),
        "v0.0.12": ("params", archive("cycle-0016-engagement-summary.json")),
        "v0.0.13": ("punch", archive("cycle-0017-punch-summary.json")),
        "v0.0.14": ("punch", archive("cycle-0018-engagement-floor-summary.json")),
        "v0.0.15": ("punch", final("cycle-0019-motor-gate-summary.json")),
        "v0.0.16": ("group", final("cycle-0020-group-gate-summary.json")),
        "v0.0.17": (
            "group_feedback",
            final("cycle-0022-group-feedback-gate-summary.json"),
        ),
        "v0.0.18": ("group_pd", final("cycle-0023-group-pd-gate-summary.json")),
        "v0.0.19": ("group_pd", final("cycle-0025-hit-stay-gate-summary.json")),
        "v0.0.20": ("group_pd", np.array(league["archive"][league["champion"]])),
    }


def dedupe(roster):
    """Identical (kind, params) play once, under the earliest version name."""
    distinct, aliases = {}, {}
    for name, (kind, params) in roster.items():
        same = next(
            (
                d
                for d, (k, p) in distinct.items()
                if k == kind and len(p) == len(params) and np.array_equal(p, params)
            ),
            None,
        )
        if same is None:
            distinct[name] = (kind, params)
        else:
            aliases[name] = same
    return distinct, aliases


def intransitive_triads(names, score):
    """Triples where a beats b, b beats c and c beats a (each score nonzero)."""

    def beats(a, b):
        s = score.get((a, b))
        return s is not None and s > 0

    out = []
    for a, b, c in itertools.combinations(names, 3):
        if beats(a, b) and beats(b, c) and beats(c, a):
            out.append([a, b, c])
        elif beats(b, a) and beats(c, b) and beats(a, c):
            out.append([a, c, b])
    return out


def shares(rows):
    n = len(rows)
    return {
        "games": n,
        "valid": sum(selfplay.valid_result(r) for r in rows) / n,
        "no_contest": sum(bool(r["no_contest"]) for r in rows) / n,
        "blocked": sum(bool(r["blocked"]) for r in rows) / n,
        "engine_fault": sum(r["invalid"] is not None for r in rows) / n,
        "median_end_tick": float(np.median([r["tick"] for r in rows])),
    }


def main(league_report, out):
    raw = PROTOCOL_PATH.read_bytes()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    distinct, aliases = dedupe(players(league_report))
    names = list(distinct)
    pairs = list(itertools.combinations(names, 2))
    tasks = [(pair, side, seed) for pair in pairs for seed in SEEDS for side in "AB"]
    jobs = [
        ((*distinct[a], NOISE), (*distinct[b], NOISE), side, seed)
        for (a, b), side, seed in tasks
    ]
    with selfplay._pool(4, HORIZON, separation_m=0.5, anti_stalling=True) as pool:
        results = pool.map(selfplay.play_game, jobs)
    ledger = Ledger(names)
    by_pair = {}
    for ((a, b), side, seed), r in zip(tasks, results, strict=True):
        ok = selfplay.valid_result(r)
        ledger.add(
            Game(a, b, side, int(r["utility"]) if ok else None, None if ok else "x")
        )
        by_pair.setdefault((a, b), []).append(r)
    score = {}
    for (a, b), rows in by_pair.items():
        valid = [r["utility"] for r in rows if selfplay.valid_result(r)]
        if valid:
            score[(a, b)] = float(np.mean(valid))
            score[(b, a)] = -score[(a, b)]
    try:
        ratings = ledger.ratings(anchor="v0.0.11")
        low, high, draws = ledger.bootstrap_ratings(
            samples=200, seed=0, anchor="v0.0.11"
        )
    except ValueError as error:  # a disconnected comparison graph has no scale
        ratings, low, high, draws = [np.nan] * len(names), None, None, 0
        print("ratings unavailable:", error)
    fin = selfplay.finite_or_none
    report = {
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "players": names,
        "aliases": aliases,
        "ratings_elo": {n: fin(x) for n, x in zip(names, ratings)},
        "bootstrap_95": {n: [fin(lo), fin(hi)] for n, lo, hi in zip(names, low, high)}
        if low is not None
        else None,
        "bootstrap_samples": draws,
        "pair_scores": {f"{a} vs {b}": s for (a, b), s in score.items() if a < b},
        "intransitive_triads": intransitive_triads(names, score),
        "degeneracy_overall": shares(results),
        "degeneracy_by_pair": {
            f"{a} vs {b}": shares(rows) for (a, b), rows in by_pair.items()
        },
        "wall_s": round(time.monotonic() - began, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "players",
                    "aliases",
                    "ratings_elo",
                    "intransitive_triads",
                    "degeneracy_overall",
                    "wall_s",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
