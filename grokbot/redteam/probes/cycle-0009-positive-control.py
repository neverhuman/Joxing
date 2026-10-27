"""Exploitability curves of the three meta-solvers on known games (cycle 0009).

    PYTHONPATH=. uv run --project experiments/v0.0.11/runtime --locked \
        python grokbot/redteam/probes/cycle-0009-positive-control.py
"""

from grokbot.positive_control import RPS, random_symmetric_game, run

CHECKPOINTS = (0, 2, 5, 10, 20, 50, 199)


def row(label, history):
    values = [history[i]["exploitability"] for i in CHECKPOINTS if i < len(history)]
    print(f"{label:<34}" + " ".join(f"{v:6.3f}" for v in values))


def main():
    print(
        f"{'exploitability at iteration':<34}"
        + " ".join(f"{i:>6}" for i in CHECKPOINTS)
    )
    games = [("rps", RPS)] + [
        (f"random20 seed{s}", random_symmetric_game(20, s)) for s in range(6)
    ]
    for name, game in games:
        for meta in ("latest", "uniform", "nash"):
            row(f"{name} {meta} exact", run(game, meta, 200))
        row(
            f"{name} nash sampled200",
            run(game, "nash", 200, games_per_pair=200, seed=1),
        )


if __name__ == "__main__":
    main()
