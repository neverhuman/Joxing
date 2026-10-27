"""Known-payoff league positive control (grokbot; P4 of the execution plan).

Before a boxing league runs, its machinery must pass on games with known
answers. This runs a PSRO-style population loop on a finite symmetric zero-sum
game, with an exact best-response oracle over pure strategies. The only things
under test are therefore the meta-solver, the empirical payoff estimate and the
exploitability measurement:

* ``latest``: plain self-play. Best-respond to the newest member only.
* ``uniform``: fictitious play. Best-respond to the population's empirical
  frequency, where every iteration adds one copy of its best response.
* ``nash``: PSRO-Nash. Best-respond to the Nash mixture of the population's
  empirical payoff matrix (``grokbot.league.solve_zero_sum``).

Payoffs between members are exact or estimated from sampled side-swapped games
through ``grokbot.league.Ledger``, so evaluation noise is part of the control.
Exploitability of the meta-strategy is measured against the TRUE game. A real
league never has the true game, and that is what makes this a positive control.
"""

from __future__ import annotations

import numpy as np

from grokbot.league import Game, Ledger, exploitability, solve_zero_sum

META_SOLVERS = ("latest", "uniform", "nash")
RPS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


def random_symmetric_game(n, seed):
    """Antisymmetric utilities in [-1, 1] for n pure strategies."""
    upper = np.triu(np.random.default_rng(seed).uniform(-1, 1, (n, n)), 1)
    return upper - upper.T


def best_response(game, mixture):
    """Lowest-index pure strategy maximizing utility against ``mixture``."""
    return int(np.argmax(game @ mixture))


def run(game, meta, iterations, *, games_per_pair=None, seed=0, start=0):
    """Population loop; returns one row per iteration.

    ``games_per_pair=None`` uses exact payoffs. Otherwise each new pair of
    distinct members plays that many side-swapped game pairs, with outcomes
    sampled as +1 with probability (1 + u) / 2 and -1 otherwise.
    """
    game = np.asarray(game, float)
    n = game.shape[0]
    if meta not in META_SOLVERS:
        raise ValueError(f"meta must be one of {META_SOLVERS}")
    if game.shape != (n, n) or not np.allclose(game, -game.T):
        raise ValueError("Symmetric zero-sum game required")
    rng = np.random.default_rng(seed)
    ledger = Ledger([f"s{i}" for i in range(n)])
    counts = np.zeros(n)
    counts[start] = 1
    latest = start
    played = set()
    history = []
    for iteration in range(iterations):
        members = np.flatnonzero(counts)
        if games_per_pair is not None:
            for i in members:
                for j in members:
                    if i < j and (i, j) not in played:
                        played.add((i, j))
                        for _ in range(games_per_pair):
                            win = rng.random() < (1 + game[i, j]) / 2
                            for side in ("A", "B"):
                                ledger.add(
                                    Game(f"s{i}", f"s{j}", side, 1 if win else -1)
                                )
        mixture = np.zeros(n)
        if meta == "latest":
            mixture[latest] = 1.0
        elif meta == "uniform":
            mixture = counts / counts.sum()
        elif len(members) == 1:
            mixture[members[0]] = 1.0
        else:
            if games_per_pair is None:
                restricted = game[np.ix_(members, members)]
            else:
                restricted = ledger.payoff()[0][np.ix_(members, members)]
            mixture[members] = solve_zero_sum(restricted)[0]
        added = best_response(game, mixture)
        history.append(
            {
                "iteration": iteration,
                "members": members.tolist(),
                "meta": mixture.tolist(),
                "exploitability": exploitability(game, mixture),
                "added": added,
            }
        )
        counts[added] += 1
        latest = added
    return history
