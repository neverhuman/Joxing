"""Reference ending-utility table for one bout (grokbot candidate, not the game).

GAME.md keeps competition disabled until falls, exits and fouls are adjudicated,
and P2.5 of the execution plan states the intended table. This module is an
executable reference for that table, so evaluation and self-play can get a
nonzero terminal utility from measured evidence rather than from a pelvis-height
shortcut. Its constants are a named candidate (``VARIANT``), not calibrated
sporting thresholds, and the env's own rules are unchanged.

Table (utility for corner A, B):

1. A numerical/engine invalid is never a result: no utility.
2. The first valid loss event ends the bout. A self-caused fall or complete exit
   loses even when leading; simultaneous self-caused losses draw.
3. A loss attributed to an opponent's foul is not the victim's loss and cannot
   reward the offender; play continues to the next event.
4. Unresolved attribution blocks the bout: no utility, reported as such.
5. Otherwise the bell decides: more points wins, equal points draw. An external
   truncation is not a sporting end: no terminal utility.

Attribution here is deliberately conservative. A fall is ``self`` only when no
inter-actor contact force occurred in the lookback window; anything else is
``unresolved`` until a foul referee exists. No ``opponent`` verdict is ever
inferred from contact alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VARIANT = "grokbot-ending-reference-v0"
PLAYERS = ("A", "B")
FOOT_BODIES = ("_calcn_", "_toes_")
SUPPORT_FORCE_N = 60.0  # ~10 % of one athlete's 596 N weight
SUPPORT_PERSIST_TICKS = 100  # 100 ms of continuous support; a brush is not a fall
ATTRIBUTION_LOOKBACK_TICKS = 500  # any opponent contact in the last 0.5 s


@dataclass(frozen=True)
class LossEvent:
    player: str
    tick: int
    kind: str  # "fall" | "exit"
    attribution: str  # "self" | "opponent" | "unresolved"

    def __post_init__(self):
        if self.player not in PLAYERS or type(self.tick) is not int or self.tick < 0:
            raise ValueError("Loss event needs a corner and a nonnegative tick")
        if self.kind not in ("fall", "exit"):
            raise ValueError("Unknown loss kind")
        if self.attribution not in ("self", "opponent", "unresolved"):
            raise ValueError("Unknown attribution")


@dataclass(frozen=True)
class Ending:
    utility: tuple[int, int] | None  # (A, B); None means no sporting result
    reason: str
    tick: int | None


def decide(points_ab, events, *, bell, invalid_reason=None, bell_tick=None):
    """Apply the table to one bout's measured evidence."""
    if invalid_reason:
        return Ending(None, "invalid: " + invalid_reason, None)
    if len(points_ab) != 2 or any(type(p) is not int or p < 0 for p in points_ab):
        raise ValueError("Points must be two nonnegative integers")
    for tick in sorted({e.tick for e in events}):
        now = [e for e in events if e.tick == tick and e.attribution != "opponent"]
        if not now:
            continue  # foul-caused losses never end the bout for the offender
        if any(e.attribution == "unresolved" for e in now):
            return Ending(None, "blocked: unresolved loss attribution", tick)
        losers = {e.player for e in now}
        if losers == set(PLAYERS):
            return Ending((0, 0), "draw: simultaneous self-caused losses", tick)
        (loser,) = losers
        kind = "/".join(sorted({e.kind for e in now}))
        return Ending(
            (-1, 1) if loser == "A" else (1, -1), f"{loser} self-caused {kind}", tick
        )
    if not bell:
        return Ending(None, "no sporting end: external truncation", None)
    a, b = points_ab
    return Ending((int(np.sign(a - b)), int(np.sign(b - a))), "bell: points", bell_tick)


def contact_ticks(evidence, registry):
    """Ticks with any force-producing contact between the two athletes."""
    ticks = []
    for row in evidence:
        for c in row["contacts"]:
            actors = {registry[c["geom1"]]["actor"], registry[c["geom2"]]["actor"]}
            if actors == set(PLAYERS) and c["normal_force_N"] > 0:
                ticks.append(row["tick"])
                break
    return ticks


def attribute(touch_ticks, start, end, lookback_ticks=ATTRIBUTION_LOOKBACK_TICKS):
    """self unless the athletes touched from lookback before start to end."""
    i = np.searchsorted(touch_ticks, start - lookback_ticks)
    return "unresolved" if i < len(touch_ticks) and touch_ticks[i] <= end else "self"


def support_losses(
    evidence,
    registry,
    *,
    force_N=SUPPORT_FORCE_N,
    persist_ticks=SUPPORT_PERSIST_TICKS,
    lookback_ticks=ATTRIBUTION_LOOKBACK_TICKS,
):
    """First persistent non-foot floor support per corner, as fall events.

    evidence is BoxingEnv.evidence: one row per physics tick with its
    contacts. registry maps geom id to the scene's semantic geom record.
    The fall starts at the first tick of the qualifying streak.
    """
    floor = {g for g, r in registry.items() if r["role"] == "floor"}
    if len(floor) != 1:
        raise ValueError("Exactly one semantic floor geom required")
    support = {
        g: r["actor"]
        for g, r in registry.items()
        if r["actor"] in PLAYERS and not any(f in r["body"] for f in FOOT_BODIES)
    }
    touches = contact_ticks(evidence, registry)
    streak = {p: 0 for p in PLAYERS}
    found = {}
    for row in evidence:
        load = {p: 0.0 for p in PLAYERS}
        for c in row["contacts"]:
            g1, g2 = c["geom1"], c["geom2"]
            other = g2 if g1 in floor else g1 if g2 in floor else None
            if other in support:
                load[support[other]] += c["normal_force_N"]
        for p in PLAYERS:
            streak[p] = streak[p] + 1 if load[p] >= force_N else 0
            if streak[p] == persist_ticks and p not in found:
                start = row["tick"] - persist_ticks + 1
                verdict = attribute(touches, start, row["tick"], lookback_ticks)
                found[p] = LossEvent(p, start, "fall", verdict)
    return [found[p] for p in PLAYERS if p in found]
