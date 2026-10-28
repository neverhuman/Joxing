"""Reference ending-utility table for one bout (grokbot candidate, not the game).

GAME.md keeps competition disabled until falls, exits and fouls are adjudicated,
and P2.5 of the execution plan states the intended table. This module is an
executable reference for that table, so evaluation and self-play can get a
nonzero terminal utility from measured evidence rather than from a pelvis-height
shortcut. Its constants are a named candidate (``VARIANT``), not calibrated
sporting thresholds, and the env's own rules are unchanged.

Table (utility for corner A, B):

1. A numerical/engine invalid is never a result: no utility.
2. The first valid loss event ends the bout. A self-caused or knockdown fall, or
   a complete exit, loses even when leading; simultaneous losses draw.
3. A loss attributed to an opponent's foul is not the victim's loss and cannot
   reward the offender; play continues to the next event.
4. Unresolved attribution blocks the bout: no utility, reported as such.
5. Otherwise the bell decides: more points wins, equal points draw. An external
   truncation is not a sporting end: no terminal utility.

Attribution (v1) classifies every force-producing contact between the athletes
in the lookback window, using the referee's own constants:

* a glove's contact run with the opponent's head, torso, guard or glove is a
  strike when, on its first tick, the contact approaches at >= ``approach_min_mps``
  AND the referee's touch shows the glove moving at >= ``active_speed_min_mps``
  relative to its own trunk (so walking into a still glove is no strike). Its first
  ``impact_window_s`` counts as boxing; a fall after boxing contact only is a
  ``knockdown`` and loses, so knocking the opponent down is worth something;
* everything else blocks: body on body (clinch, lean, shove), a glove on a
  non-scoring surface (possible foul), a slow glove push, and holding past the
  impact window. A fall after any blocking contact is ``unresolved``;
* no contact at all makes the fall ``self``.

No ``opponent`` (foul) verdict is ever inferred from contact alone. Residual
exploit: a falling athlete can still lean or slowly push to force a block, so
``arena.policy_summary`` reports blocked-as-faller rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from joxing.rules import Rules

VARIANT = "grokbot-ending-reference-v2"
PLAYERS = ("A", "B")
FOOT_BODIES = ("_calcn_", "_toes_")
SUPPORT_FORCE_N = 60.0  # ~10 % of one athlete's 596 N weight
SUPPORT_PERSIST_TICKS = 100  # 100 ms of continuous support; a brush is not a fall
ATTRIBUTION_LOOKBACK_TICKS = 500  # any opponent contact in the last 0.5 s
STRIKE_TARGETS = ("head", "torso", "guard", "glove")
SCORING_TARGETS = ("head", "torso")  # a strike here is a hit, not a blocked punch
ATTRIBUTIONS = ("self", "knockdown", "opponent", "unresolved")


@dataclass(frozen=True)
class LossEvent:
    player: str
    tick: int
    kind: str  # "fall" | "exit"
    attribution: str  # one of ATTRIBUTIONS

    def __post_init__(self):
        if self.player not in PLAYERS or type(self.tick) is not int or self.tick < 0:
            raise ValueError("Loss event needs a corner and a nonnegative tick")
        if self.kind not in ("fall", "exit"):
            raise ValueError("Unknown loss kind")
        if self.attribution not in ATTRIBUTIONS:
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
            return Ending((0, 0), "draw: simultaneous losses", tick)
        (loser,) = losers
        kind = "/".join(sorted({e.kind for e in now}))
        how = (
            "knocked down"
            if any(e.attribution == "knockdown" for e in now)
            else "self-caused"
        )
        return Ending(
            (-1, 1) if loser == "A" else (1, -1), f"{loser} {how} {kind}", tick
        )
    if not bell:
        return Ending(None, "no sporting end: external truncation", None)
    a, b = points_ab
    return Ending((int(np.sign(a - b)), int(np.sign(b - a))), "bell: points", bell_tick)


class ContactClassifier:
    """Incremental strike/blocking classification of evidence rows (see below)."""

    def __init__(self, registry, rules=None):
        self.registry = registry
        self.rules = rules or Rules()
        self.window = round(self.rules.impact_window_s / self.rules.dt_s)
        self.gap = round(self.rules.gap_s / self.rules.dt_s)
        self.runs = {}  # glove geom -> [first_tick, last_tick, is_strike]
        self.strikes, self.blocking = [], []
        self.strikes_by = {p: [] for p in PLAYERS}  # ticks each corner was striking
        self.scoring_by = {p: [] for p in PLAYERS}  # ...on a head or torso target

    def feed(self, row):
        rules, registry, t = self.rules, self.registry, row["tick"]
        boxing = blocks = False
        strikers_tick, scorers_tick = set(), set()
        active = {
            (glove, touch["sample_id"]): touch["active_speed_mps"]
            for glove, interval in row.get("gloves", {}).items()
            for touch in interval["touches"]
        }
        for index, c in enumerate(row["contacts"]):
            ends = ((c["geom1"], c["geom2"]), (c["geom2"], c["geom1"]))
            roles = {g: registry[g] for g in (c["geom1"], c["geom2"])}
            if c["normal_force_N"] <= 0 or {r["actor"] for r in roles.values()} != set(
                PLAYERS
            ):
                continue
            gloves = [
                (g, o)
                for g, o in ends
                if roles[g]["role"] == "glove" and roles[o]["role"] in STRIKE_TARGETS
            ]
            if not gloves:
                blocks = True
            sample = f"{index}:{c['geom1']}:{c['geom2']}"
            striking = False
            strikers, scorers = set(), set()
            for g, o in gloves:
                speed = active.get((registry[g]["name"], sample), -np.inf)
                fast = (
                    c["approach_mps"] >= rules.approach_min_mps
                    and speed >= rules.active_speed_min_mps
                )
                run = self.runs.get(g)
                if run is None or t - run[1] > self.gap:
                    run = self.runs[g] = [t, t, fast]
                run[1] = t
                if run[0] == t:
                    run[2] = run[2] or fast
                if run[2] and t - run[0] < self.window:
                    striking = True
                    strikers.add(registry[g]["actor"])
                    if roles[o]["role"] in SCORING_TARGETS:
                        scorers.add(registry[g]["actor"])
            # A punch into a still guarding glove is boxing: one striking glove
            # in the contact is enough.
            if gloves and striking:
                boxing = True
                strikers_tick |= strikers
                scorers_tick |= scorers
            elif gloves:
                blocks = True
        if boxing:
            self.strikes.append(t)
        if blocks:
            self.blocking.append(t)
        for p in strikers_tick:
            self.strikes_by[p].append(t)
        for p in scorers_tick:
            self.scoring_by[p].append(t)


def classify_contacts(evidence, registry, rules=None):
    """(strike_ticks, blocking_ticks) of force-producing inter-athlete contact.

    A glove's contact run (gaps of at most ``gap_s``) with a strike target is a
    strike if a contact on its first tick approaches at >= ``approach_min_mps``
    and the referee's touch for that glove (``row["gloves"]``, matched by sample
    id ``index:geom1:geom2``) moves at >= ``active_speed_min_mps`` against its own
    trunk. Only the run's first ``impact_window_s`` is boxing. A glove contact
    without a referee touch, and every other contact tick, blocks attribution.
    """
    classifier = ContactClassifier(registry, rules)
    for row in evidence:
        classifier.feed(row)
    return classifier.strikes, classifier.blocking


def attribute(contacts, start, end, lookback_ticks=ATTRIBUTION_LOOKBACK_TICKS):
    """Verdict for a loss over [start - lookback, end] from classified contacts."""

    def touched(ticks):
        i = np.searchsorted(ticks, start - lookback_ticks)
        return bool(i < len(ticks) and ticks[i] <= end)

    strikes, blocking = contacts
    if touched(blocking):
        return "unresolved"
    return "knockdown" if touched(strikes) else "self"


class BoutMonitor:
    """Incremental fall detection and attribution over evidence rows.

    Feed ``BoxingEnv.evidence`` rows in tick order with ``update``; each costs
    O(new rows), so a learner can check every control step of a 60 s bout.
    ``losses`` holds each corner's first persistent non-foot floor support,
    attributed with the contacts seen up to the tick it qualified. It equals
    ``support_losses`` on the same rows.
    """

    def __init__(
        self,
        registry,
        *,
        force_N=SUPPORT_FORCE_N,
        persist_ticks=SUPPORT_PERSIST_TICKS,
        lookback_ticks=ATTRIBUTION_LOOKBACK_TICKS,
        rules=None,
    ):
        floor = {g for g, r in registry.items() if r["role"] == "floor"}
        if len(floor) != 1:
            raise ValueError("Exactly one semantic floor geom required")
        self.floor = floor
        self.support = {
            g: r["actor"]
            for g, r in registry.items()
            if r["actor"] in PLAYERS and not any(f in r["body"] for f in FOOT_BODIES)
        }
        self.force_N, self.persist_ticks = force_N, persist_ticks
        self.lookback_ticks = lookback_ticks
        self.contacts = ContactClassifier(registry, rules)
        self.streak = {p: 0 for p in PLAYERS}
        self.found = {}
        self.last_tick = None

    def update(self, rows):
        for row in rows:
            t = row["tick"]
            if self.last_tick is not None and t != self.last_tick + 1:
                raise ValueError("Evidence rows must be contiguous and in tick order")
            self.last_tick = t
            self.contacts.feed(row)
            load = {p: 0.0 for p in PLAYERS}
            for c in row["contacts"]:
                g1, g2 = c["geom1"], c["geom2"]
                other = g2 if g1 in self.floor else g1 if g2 in self.floor else None
                if other in self.support:
                    load[self.support[other]] += c["normal_force_N"]
            for p in PLAYERS:
                self.streak[p] = self.streak[p] + 1 if load[p] >= self.force_N else 0
                if self.streak[p] == self.persist_ticks and p not in self.found:
                    start = t - self.persist_ticks + 1
                    verdict = self.attribute(start, t)
                    self.found[p] = LossEvent(p, start, "fall", verdict)

    def attribute(self, start, end):
        return attribute(
            (self.contacts.strikes, self.contacts.blocking),
            start,
            end,
            self.lookback_ticks,
        )

    @property
    def losses(self):
        return [self.found[p] for p in PLAYERS if p in self.found]


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
    monitor = BoutMonitor(
        registry,
        force_N=force_N,
        persist_ticks=persist_ticks,
        lookback_ticks=lookback_ticks,
    )
    monitor.update(evidence)
    return monitor.losses


def anti_stalling(ending, strikes_by, points_ab):
    """Reference rule v3: a result reached without engagement is a no-contest.

    If neither corner landed a strike (``ContactClassifier.strikes_by``) before
    the ending tick and no points were scored, the bout gives no utility: a
    "win" by outlasting a passive opponent is not boxing and must not pay.
    Blocked and invalid endings pass through unchanged.
    """
    if ending.utility is None or any(points_ab):
        return ending
    limit = ending.tick if ending.tick is not None else np.inf
    if any(t <= limit for ticks in strikes_by.values() for t in ticks):
        return ending
    return Ending(None, "no contest: no engagement before the ending", ending.tick)
