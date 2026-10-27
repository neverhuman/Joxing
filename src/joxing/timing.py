"""Bounded private muscle-command queues on the shared integer physics clock."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .bodies.base import owned_readonly
from .history import tick
from .provenance import digest


@dataclass(frozen=True)
class CommandPacket:
    game_id: str
    player: str
    decision_tick: int
    apply_tick: int
    command: np.ndarray
    bundle_hash: str

    def __post_init__(self):
        if not self.game_id or self.player not in {"A", "B"} or not self.bundle_hash:
            raise ValueError("Missing command ownership/bundle")
        if tick(self.decision_tick) > tick(self.apply_tick):
            raise ValueError("Command cannot apply before decision")
        if (
            self.command.shape != (700,)
            or self.command.dtype != np.float32
            or not np.isfinite(self.command).all()
            or np.any(np.abs(self.command) > 1)
        ):
            raise ValueError("Command must be finite bounded float32 normalized700")
        object.__setattr__(self, "command", owned_readonly(self.command))

    def snapshot(self):
        return {
            "game_id": self.game_id,
            "player": self.player,
            "decision_tick": self.decision_tick,
            "apply_tick": self.apply_tick,
            "command": self.command.tolist(),
            "bundle_hash": self.bundle_hash,
        }

    @property
    def hash(self):
        return digest(self.snapshot())

    @classmethod
    def restore(cls, state):
        raw = np.asarray(state["command"], np.float64)
        if not np.isfinite(raw).all() or np.any(np.abs(raw) > 1):
            raise ValueError("Corrupt saved command")
        return cls(
            state["game_id"],
            state["player"],
            state["decision_tick"],
            state["apply_tick"],
            raw.astype(np.float32),
            state["bundle_hash"],
        )


class ActionQueue:
    def __init__(
        self, game_id, player, bundle_hash, *, capacity=64, max_delay_ticks=40
    ):
        if not game_id or player not in {"A", "B"} or not bundle_hash:
            raise ValueError("Missing queue ownership/bundle")
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("Invalid queue capacity")
        tick(max_delay_ticks)
        self.game_id, self.player, self.bundle_hash = game_id, player, bundle_hash
        self.capacity, self.max_delay_ticks = capacity, max_delay_ticks
        self.pending = {}
        self.seen = OrderedDict()
        self.last_polled_tick = -1
        self.last_applied = None
        self.last_delivery = None

    def enqueue(self, packet, *, now_tick):
        tick(now_tick)
        if (packet.game_id, packet.player, packet.bundle_hash) != (
            self.game_id,
            self.player,
            self.bundle_hash,
        ):
            raise ValueError("Foreign command owner/bundle")
        fingerprint = self.seen.get(packet.decision_tick)
        if fingerprint is not None:
            if fingerprint != packet.hash:
                raise ValueError("Conflicting command decision identity")
            return False
        if (
            packet.decision_tick > now_tick
            or packet.apply_tick < now_tick
            or now_tick != self.last_polled_tick + 1
            or packet.apply_tick <= self.last_polled_tick
            or packet.apply_tick - packet.decision_tick > self.max_delay_ticks
            or (
                self.last_applied is not None
                and packet.decision_tick <= self.last_applied.decision_tick
            )
        ):
            raise ValueError("Future, stale or excessive-delay command")
        if len(self.pending) >= self.capacity:
            raise TimeoutError("Private action queue capacity")
        self.pending[packet.decision_tick] = packet
        self.seen[packet.decision_tick] = packet.hash
        while len(self.seen) > 2 * self.capacity:
            self.seen.popitem(last=False)
        return True

    def pop_due(self, current_tick):
        tick(current_tick)
        if current_tick == self.last_polled_tick:
            return None  # Idempotent retry; no second application.
        if current_tick != self.last_polled_tick + 1:
            raise ValueError("Missing or reversed action-queue physics tick")
        due = [p for p in self.pending.values() if p.apply_tick <= current_tick]
        for p in due:
            del self.pending[p.decision_tick]
        eligible = [
            p
            for p in due
            if self.last_applied is None
            or p.decision_tick > self.last_applied.decision_tick
        ]
        chosen = max(eligible, key=lambda p: p.decision_tick) if eligible else None
        self.last_polled_tick = current_tick
        self.last_delivery = {
            "tick": current_tick,
            "applied_decision": None if chosen is None else chosen.decision_tick,
            "discarded_decisions": sorted(
                p.decision_tick for p in due if p is not chosen
            ),
        }
        if chosen is not None:
            self.last_applied = chosen
        return chosen

    def snapshot(self):
        return {
            "schema": "joxing-action-queue-v1",
            "game_id": self.game_id,
            "player": self.player,
            "bundle_hash": self.bundle_hash,
            "capacity": self.capacity,
            "max_delay_ticks": self.max_delay_ticks,
            "last_polled_tick": self.last_polled_tick,
            "last_applied": None
            if self.last_applied is None
            else self.last_applied.snapshot(),
            "pending": [p.snapshot() for _, p in sorted(self.pending.items())],
            "seen": [[k, v] for k, v in self.seen.items()],
            "last_delivery": self.last_delivery,
        }

    @classmethod
    def restore(cls, state, *, game_id, player, bundle_hash):
        if state["schema"] != "joxing-action-queue-v1" or (
            state["game_id"],
            state["player"],
            state["bundle_hash"],
        ) != (game_id, player, bundle_hash):
            raise ValueError("Foreign action queue snapshot")
        q = cls(
            game_id,
            player,
            bundle_hash,
            capacity=state["capacity"],
            max_delay_ticks=state["max_delay_ticks"],
        )
        polled = state["last_polled_tick"]
        if type(polled) is not int or polled < -1:
            raise ValueError("Malformed queue clock")
        if len(state["seen"]) > 2 * q.capacity or len(state["pending"]) > q.capacity:
            raise ValueError("Unbounded saved queue")
        for decision, fingerprint in state["seen"]:
            tick(decision)
            if (
                decision in q.seen
                or not isinstance(fingerprint, str)
                or len(fingerprint) != 64
            ):
                raise ValueError("Malformed seen-decision receipt")
            q.seen[decision] = fingerprint
        for entry in state["pending"]:
            packet = CommandPacket.restore(entry)
            if (
                (packet.game_id, packet.player, packet.bundle_hash)
                != (game_id, player, bundle_hash)
                or packet.apply_tick <= polled
                or packet.decision_tick in q.pending
            ):
                raise ValueError("Malformed pending command")
            if (
                packet.apply_tick - packet.decision_tick > q.max_delay_ticks
                or q.seen.get(packet.decision_tick) != packet.hash
            ):
                raise ValueError("Pending command identity/delay mismatch")
            q.pending[packet.decision_tick] = packet
        if state["last_applied"] is not None:
            packet = CommandPacket.restore(state["last_applied"])
            if (
                (packet.game_id, packet.player, packet.bundle_hash)
                != (
                    game_id,
                    player,
                    bundle_hash,
                )
                or packet.apply_tick > polled
                or q.seen.get(packet.decision_tick) != packet.hash
            ):
                raise ValueError("Malformed applied command")
            q.last_applied = packet
        q.last_polled_tick = polled
        q.last_delivery = state["last_delivery"]
        if (polled == -1) != (q.last_delivery is None):
            raise ValueError("Queue delivery receipt missing or premature")
        if q.last_delivery is not None and (
            q.last_delivery["tick"] != polled
            or len(q.last_delivery["discarded_decisions"]) > q.capacity
            or (
                q.last_delivery["applied_decision"] is not None
                and (
                    q.last_applied is None
                    or q.last_delivery["applied_decision"]
                    != q.last_applied.decision_tick
                )
            )
        ):
            raise ValueError("Queue delivery clock mismatch")
        return q
