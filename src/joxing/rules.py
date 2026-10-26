"""Pure candidate hit referee. Inputs must come from a validated physical adapter.

This module is fixture-ready, not production-qualified. Recovery distances and active
strike evidence are measured inputs; callers cannot replace them with policy flags.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .provenance import digest


@dataclass(frozen=True)
class Rules:
    schema: str = "joxing-points-v0-candidate"
    dt_s: float = 0.001
    impact_window_s: float = 0.050
    gap_s: float = 0.008
    clear_s: float = 0.030
    clearance_m: float = 0.020
    retraction_m: float = 0.030
    impulse_min_Ns: float = 0.25
    approach_min_mps: float = 0.5
    active_speed_min_mps: float = 0.5
    face_cos_min: float = 0.5
    head_points: int = 2
    torso_points: int = 1

    def __post_init__(self):
        for name in (
            "dt_s",
            "impact_window_s",
            "gap_s",
            "clear_s",
            "clearance_m",
            "retraction_m",
            "impulse_min_Ns",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("impact_window_s", "gap_s", "clear_s"):
            ratio = getattr(self, name) / self.dt_s
            if abs(ratio - round(ratio)) > 1e-8:
                raise ValueError(f"{name} must contain integer physics ticks")

    @property
    def hash(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class Touch:
    """One distinct manifold point for one glove and opponent at one tick."""

    sample_id: str
    zone: str  # head, torso, guard, illegal/non_scoring
    normal_force_N: float
    approach_mps: float
    active_speed_mps: float
    face_cos: float
    striking_face: bool = True

    def __post_init__(self):
        values = (
            self.normal_force_N,
            self.approach_mps,
            self.active_speed_mps,
            self.face_cos,
        )
        if not all(math.isfinite(v) for v in values) or self.normal_force_N < 0:
            raise ValueError("Nonfinite or negative contact force: quarantine episode")


@dataclass
class Encounter:
    sequence: int
    onset_tick: int
    peak_extension_m: float
    last_contact_tick: int
    last_direct_tick: int | None = None
    direct_started: bool = False
    onset_eligible: dict[str, bool] = field(default_factory=dict)
    impulses: dict[str, float] = field(default_factory=dict)
    window_impulse_Ns: float = 0.0
    guard_window_impulse_Ns: float = 0.0
    peak_force_N: float = 0.0
    finalized: bool = False


class HitTracker:
    """One instance per glove. Serialize application state alongside engine state."""

    def __init__(self, episode_id: str, glove_id: str, rules: Rules | None = None):
        self.rules = rules or Rules()
        self.episode_id, self.glove_id = episode_id, glove_id
        self.last_tick = -1
        self.sequence = 0
        self.current: Encounter | None = None
        self.clear_since: int | None = None
        self.ledger: dict[str, dict] = {}

    def _ticks(self, seconds):
        return round(seconds / self.rules.dt_s)

    def update(
        self,
        tick: int,
        touches: list[Touch],
        *,
        clearance_m: float,
        extension_m: float,
        bell: bool = False,
    ) -> list[dict]:
        if tick == self.last_tick:
            return []  # idempotent delivery retry
        if tick < self.last_tick or (
            self.last_tick >= 0 and tick != self.last_tick + 1
        ):
            raise ValueError("Missing or reversed physics tick: quarantine episode")
        if not all(math.isfinite(v) for v in (clearance_m, extension_m)):
            raise ValueError("Invalid motion evidence")
        self.last_tick = tick
        unique = {}
        for touch in touches:
            if touch.sample_id in unique and unique[touch.sample_id] != touch:
                raise ValueError("Conflicting contact sample identity")
            unique[touch.sample_id] = touch
        contacts = [t for t in unique.values() if t.normal_force_N > 0]
        if contacts and self.current is None:
            self.sequence += 1
            self.current = Encounter(self.sequence, tick, extension_m, tick)
        event = self.current
        if event is None:
            return []
        event.peak_extension_m = max(event.peak_extension_m, extension_m)
        if contacts:
            event.last_contact_tick = tick
            self.clear_since = None
        elif clearance_m >= self.rules.clearance_m:
            if self.clear_since is None:
                self.clear_since = tick
        else:
            self.clear_since = None
        recovered = (
            self.clear_since is not None
            and tick - self.clear_since + 1 >= self._ticks(self.rules.clear_s)
            and event.peak_extension_m - extension_m >= self.rules.retraction_m
        )
        emitted = []
        if not event.finalized:
            # Tick denotes start of the solved interval. bell finalizes prior evidence only.
            in_window = (
                tick - event.onset_tick < self._ticks(self.rules.impact_window_s)
                and not bell
            )
            if in_window:
                event.peak_force_N = max(
                    event.peak_force_N, sum(t.normal_force_N for t in contacts)
                )
                for touch in contacts:
                    impulse = touch.normal_force_N * self.rules.dt_s
                    event.window_impulse_Ns += impulse
                    if touch.zone == "guard":
                        event.guard_window_impulse_Ns += impulse
                    if touch.zone not in ("head", "torso"):
                        continue
                    event.direct_started = True
                    event.last_direct_tick = tick
                    eligible = (
                        touch.approach_mps >= self.rules.approach_min_mps
                        and touch.active_speed_mps >= self.rules.active_speed_min_mps
                        and touch.face_cos >= self.rules.face_cos_min
                        and touch.striking_face
                    )
                    # Latch kinematic onset per zone; current face membership still gates impulse.
                    event.onset_eligible.setdefault(touch.zone, eligible)
                    if (
                        event.onset_eligible[touch.zone]
                        and touch.striking_face
                        and touch.face_cos >= self.rules.face_cos_min
                    ):
                        event.impulses[touch.zone] = (
                            event.impulses.get(touch.zone, 0.0) + impulse
                        )
            direct_released = (
                event.last_direct_tick is not None
                and tick - event.last_direct_tick >= self._ticks(self.rules.gap_s)
            )
            deadline = tick - event.onset_tick + 1 >= self._ticks(
                self.rules.impact_window_s
            )
            if bell or deadline or direct_released or recovered:
                points = {
                    "head": self.rules.head_points,
                    "torso": self.rules.torso_points,
                }
                zone = (
                    min(
                        event.impulses, key=lambda z: (-event.impulses[z], points[z], z)
                    )
                    if event.impulses
                    else None
                )
                award = (
                    points[zone]
                    if zone and event.impulses[zone] >= self.rules.impulse_min_Ns
                    else 0
                )
                event_id = f"{self.episode_id}/{self.glove_id}/{event.sequence}"
                record = {
                    "event_id": event_id,
                    "rules_hash": self.rules.hash,
                    "onset_tick": event.onset_tick,
                    "finalized_tick": tick,
                    "reason": "bell" if bell else "window" if deadline else "release",
                    "zone": zone,
                    "points": award,
                    "direct_impulses_Ns": event.impulses.copy(),
                    "guard_window_impulse_Ns": event.guard_window_impulse_Ns,
                    "window_impulse_Ns": event.window_impulse_Ns,
                    "peak_force_N": event.peak_force_N,
                    "onset_eligible": event.onset_eligible.copy(),
                }
                self.ledger[event_id] = record
                emitted.append(record)
                event.finalized = True
        if event.finalized and recovered:
            self.current = None
            self.clear_since = None
        return emitted

    def snapshot(self) -> dict:
        return {
            "rules": asdict(self.rules),
            "episode_id": self.episode_id,
            "glove_id": self.glove_id,
            "last_tick": self.last_tick,
            "sequence": self.sequence,
            "clear_since": self.clear_since,
            "current": asdict(self.current) if self.current else None,
            "ledger": self.ledger.copy(),
        }

    @classmethod
    def restore(cls, state: dict) -> HitTracker:
        result = cls(state["episode_id"], state["glove_id"], Rules(**state["rules"]))
        for key in ("last_tick", "sequence", "clear_since", "ledger"):
            setattr(result, key, state[key])
        result.current = Encounter(**state["current"]) if state["current"] else None
        return result


def potential(score_difference: int, terminal: bool = False) -> float:
    return 0.0 if terminal else 0.25 * math.tanh(score_difference / 4)


def match_reward(
    old_diff: int, new_diff: int, *, terminal: bool, outcome: int = 0
) -> tuple[float, float]:
    if outcome not in (-1, 0, 1) or (not terminal and outcome != 0):
        raise ValueError("Outcome must be terminal win/loss/draw")
    reward = outcome + potential(new_diff, terminal) - potential(old_diff)
    return reward, -reward
