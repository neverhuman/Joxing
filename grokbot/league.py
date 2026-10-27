"""Known-payoff evaluation oracle for two-player self-play (grokbot, cycle 0002).

Pure numpy. Turns a list of finished games between frozen policies into:

* a payoff ledger that keeps every attempted game in its denominator
  (valid wins/draws/losses per ordered pair and side, plus invalid counts),
* an antisymmetric empirical payoff matrix ``M[i, j]`` = mean utility of i vs j,
* Bradley-Terry ratings on the Elo scale with seeded bootstrap intervals,
* the exact maximin (Nash) mixture of the zero-sum matrix game and the
  exploitability of any population mixture, with a duality-gap certificate.

Utility is the canonical tactical result: win +1, draw 0, loss -1. Invalid games
never become results; they are counted and reported separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ELO = 400.0 / math.log(10.0)
SIDES = ("A", "B")


@dataclass(frozen=True)
class Game:
    """One attempted game. ``outcome`` is from ``first``'s point of view."""

    first: str
    second: str
    first_side: str
    outcome: int | None
    invalid_reason: str | None = None

    def __post_init__(self):
        if not self.first or not self.second or self.first == self.second:
            raise ValueError("Two distinct named policies required")
        if self.first_side not in SIDES:
            raise ValueError("first_side must be 'A' or 'B'")
        if self.invalid_reason is None:
            if self.outcome not in (-1, 0, 1):
                raise ValueError("Valid game outcome must be -1, 0 or +1")
        elif self.outcome is not None or not self.invalid_reason:
            raise ValueError("Invalid game carries a reason and no outcome")

    @property
    def valid(self):
        return self.invalid_reason is None


class Ledger:
    def __init__(self, players):
        players = list(players)
        if len(set(players)) != len(players) or len(players) < 2:
            raise ValueError("At least two distinct players required")
        self.players = players
        self.index = {p: i for i, p in enumerate(players)}
        self.games: list[Game] = []

    def add(self, game: Game):
        if game.first not in self.index or game.second not in self.index:
            raise ValueError("Unregistered player")
        self.games.append(game)

    def _arrays(self, games=None):
        games = self.games if games is None else games
        valid = [g for g in games if g.valid]
        i = np.array([self.index[g.first] for g in valid], int)
        j = np.array([self.index[g.second] for g in valid], int)
        u = np.array([g.outcome for g in valid], float)
        return i, j, u

    def counts(self):
        """Per-ordered-pair totals; [i, j] counts games of i against j from i's view."""
        n = len(self.players)
        out = {k: np.zeros((n, n), int) for k in ("win", "draw", "loss", "invalid")}
        for g in self.games:
            a, b = self.index[g.first], self.index[g.second]
            if not g.valid:
                out["invalid"][a, b] += 1
                out["invalid"][b, a] += 1
                continue
            key = {1: "win", 0: "draw", -1: "loss"}[g.outcome]
            mirror = {"win": "loss", "draw": "draw", "loss": "win"}[key]
            out[key][a, b] += 1
            out[mirror][b, a] += 1
        out["attempted"] = out["win"] + out["draw"] + out["loss"] + out["invalid"]
        return out

    def side_balance(self):
        """Mean utility of the corner-A player over valid games; 0 if fair."""
        values = [
            g.outcome if g.first_side == "A" else -g.outcome
            for g in self.games
            if g.valid
        ]
        return float(np.mean(values)) if values else float("nan")

    def payoff(self, games=None):
        """Antisymmetric mean-utility matrix; NaN where a pair has no valid game."""
        n = len(self.players)
        total, count = np.zeros((n, n)), np.zeros((n, n))
        i, j, u = self._arrays(games)
        np.add.at(total, (i, j), u)
        np.add.at(total, (j, i), -u)
        np.add.at(count, (i, j), 1)
        np.add.at(count, (j, i), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = total / count
        matrix[count == 0] = np.nan
        np.fill_diagonal(matrix, 0.0)
        return matrix, count

    def ratings(self, *, anchor=None, games=None, prior_games=0.5):
        """Bradley-Terry strengths on the Elo scale (draw = half win).

        ``prior_games`` adds that many virtual draws between every pair that met,
        keeping undefeated players finite. Players without a valid game are NaN.
        Ratings are centred on mean 0, or on ``anchor`` = 0 when given. A
        disconnected comparison graph has no common scale and is rejected.
        """
        n = len(self.players)
        wins = np.zeros((n, n))
        i, j, u = self._arrays(games)
        np.add.at(wins, (i, j), (u + 1) / 2)
        np.add.at(wins, (j, i), (1 - u) / 2)
        met = (wins + wins.T) > 0
        rated = np.flatnonzero(met.any(1))
        if len(rated) < 2:
            raise ValueError("Ratings need at least one valid game")
        reached, frontier = {rated[0]}, [rated[0]]
        while frontier:
            new = set(np.flatnonzero(met[frontier].any(0))) - reached
            reached |= new
            frontier = list(new)
        if len(reached) != len(rated):
            raise ValueError("Disconnected comparison graph: no common rating scale")
        sub = np.ix_(rated, rated)
        met, wins = met[sub], wins[sub] + prior_games / 2 * met[sub]
        games_ij = wins + wins.T
        theta = np.zeros(len(rated))
        for _ in range(10_000):
            # Hunter (2004) minorise-maximise update for Bradley-Terry.
            p = np.exp(theta)
            denom = np.where(met, games_ij / (p[:, None] + p[None, :]), 0.0).sum(1)
            new = np.log(np.maximum(wins.sum(1), 1e-300) / denom)
            new -= new.mean()
            if np.max(np.abs(new - theta)) < 1e-12:
                theta = new
                break
            theta = new
        rating = np.full(n, np.nan)
        rating[rated] = theta * ELO
        if anchor is not None:
            rating = rating - rating[self.index[anchor]]
        return rating

    def bootstrap_ratings(self, *, samples=200, seed=0, anchor=None, level=0.95):
        """Percentile intervals from resampling valid games with replacement.

        Resamples whose comparison graph disconnects are skipped; the number of
        usable resamples is returned with the (low, high) bounds.
        """
        rng = np.random.default_rng(seed)
        valid = [g for g in self.games if g.valid]
        draws = []
        for _ in range(samples):
            pick = rng.integers(0, len(valid), len(valid))
            try:
                draws.append(
                    self.ratings(anchor=anchor, games=[valid[k] for k in pick])
                )
            except ValueError:
                continue
        if not draws:
            raise ValueError("No resample has a connected comparison graph")
        tail = (1 - level) / 2
        draws = np.array(draws)
        low, high = np.full((2, len(self.players)), np.nan)
        seen = np.isfinite(draws).any(0)
        low[seen], high[seen] = np.nanquantile(draws[:, seen], [tail, 1 - tail], axis=0)
        return low, high, len(draws)


def solve_zero_sum(matrix, *, tol=1e-9):
    """Exact maximin row mixture and game value for a finite zero-sum matrix.

    Simplex with Bland's rule on: max 1'y s.t. M' y <= 1, y >= 0, where
    M' = M - min(M) + 1 > 0. The row mixture comes from the slack duals.
    Returns (row_mixture, column_mixture, value).
    """
    m = np.asarray(matrix, float)
    if m.ndim != 2 or not np.isfinite(m).all():
        raise ValueError("Finite 2-D payoff matrix required")
    rows, cols = m.shape
    shift = 1.0 - m.min()
    shifted = m + shift
    # Tableau: [shifted | I | 1] with objective row [-1 ... | 0 | 0].
    tableau = np.zeros((rows + 1, cols + rows + 1))
    tableau[:rows, :cols] = shifted
    tableau[:rows, cols : cols + rows] = np.eye(rows)
    tableau[:rows, -1] = 1.0
    tableau[-1, :cols] = -1.0
    basis = list(range(cols, cols + rows))
    for _ in range(50 * (rows + cols) + 100):
        entering = next((k for k in range(cols + rows) if tableau[-1, k] < -tol), None)
        if entering is None:
            break
        column = tableau[:rows, entering]
        ratios = np.full(rows, np.inf)
        positive = column > tol
        ratios[positive] = tableau[:rows, -1][positive] / column[positive]
        best = ratios.min()
        if not np.isfinite(best):
            raise RuntimeError("Unbounded payoff program")
        leaving = min(
            (r for r in range(rows) if positive[r] and ratios[r] <= best + tol),
            key=lambda r: basis[r],
        )
        pivot = tableau[leaving] / tableau[leaving, entering]
        tableau -= np.outer(tableau[:, entering], pivot)
        tableau[leaving] = pivot
        basis[leaving] = entering
    else:
        raise RuntimeError("Simplex iteration limit")
    y = np.zeros(cols)
    for r, b in enumerate(basis):
        if b < cols:
            y[b] = tableau[r, -1]
    x = tableau[-1, cols : cols + rows].copy()
    total = tableau[-1, -1]
    value = 1.0 / total - shift
    row = np.clip(x / total, 0.0, None)
    col = np.clip(y / total, 0.0, None)
    row, col = row / row.sum(), col / col.sum()
    gap = (row @ m).min(), (m @ col).max()
    if gap[1] - gap[0] > 1e-7 * max(1.0, np.abs(m).max()):
        raise RuntimeError("Nash certificate failed: duality gap " + str(gap))
    return row, col, float(value)


def exploitability(matrix, mixture):
    """Best pure-response utility against a population mixture.

    ``(M @ p)[j]`` is what policy j earns against an opponent drawn from p. For an
    antisymmetric (symmetric zero-sum) matrix the game value is 0, so this is >= 0
    and equals 0 exactly at a Nash mixture.
    """
    m = np.asarray(matrix, float)
    p = np.asarray(mixture, float)
    if (
        m.ndim != 2
        or m.shape[0] != m.shape[1]
        or not np.isfinite(m).all()
        or not np.allclose(m, -m.T)
    ):
        raise ValueError("Finite antisymmetric square payoff matrix required")
    if p.shape != (m.shape[0],) or np.any(p < -1e-12) or abs(p.sum() - 1) > 1e-9:
        raise ValueError("Mixture must be a probability vector over the policies")
    return float(np.max(m @ p))
