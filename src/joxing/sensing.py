"""Bounded causal sensor transport; pending packets and match IDs stay private."""

import copy
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .bodies.base import owned_readonly
from .history import HistoryView, TimedHistory, TimedSample, tick
from .observations import DIMENSION, SCHEMA_HASH, ActorObservation
from .provenance import digest


@dataclass(frozen=True)
class SensorPacket:
    game_id: str
    player: str
    schema_hash: str
    sampled_tick: int
    delivery_tick: int
    values: np.ndarray

    def __post_init__(self):
        if (
            not self.game_id
            or self.player not in {"A", "B"}
            or self.schema_hash != SCHEMA_HASH
        ):
            raise ValueError("Foreign sensor owner/schema")
        if tick(self.delivery_tick) < tick(self.sampled_tick):
            raise ValueError("Sensor delivery precedes sampling")
        if (
            self.values.dtype != np.float32
            or self.values.shape != (DIMENSION,)
            or not np.isfinite(self.values).all()
        ):
            raise ValueError("Sensor features require finite float32 declared vector")
        object.__setattr__(self, "values", owned_readonly(self.values))

    def snapshot(self):
        return {
            "game_id": self.game_id,
            "player": self.player,
            "schema_hash": self.schema_hash,
            "sampled_tick": self.sampled_tick,
            "delivery_tick": self.delivery_tick,
            "values": self.values.tolist(),
        }

    @property
    def hash(self):
        return digest(self.snapshot())

    @classmethod
    def restore(cls, state):
        packet = cls(
            state["game_id"],
            state["player"],
            state["schema_hash"],
            state["sampled_tick"],
            state["delivery_tick"],
            np.array(state["values"], np.float32),
        )
        if packet.snapshot() != state:
            raise ValueError("Noncanonical sensor payload")
        return packet


@dataclass(frozen=True)
class ActorInput:
    decision_tick: int
    latest_observation: ActorObservation | None
    history: HistoryView
    available: bool
    sample_age_ticks: int
    receipt_age_ticks: int


class SensorQueue:
    def __init__(self, game_id, player, *, capacity=64, max_delay_ticks=40):
        if not game_id or player not in {"A", "B"}:
            raise ValueError("Missing sensor ownership")
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("Invalid sensor capacity")
        if type(max_delay_ticks) is not int or not 0 <= max_delay_ticks <= 40:
            raise ValueError("Invalid sensor delay bound")
        self.game_id, self.player = game_id, player
        self.capacity, self.max_delay_ticks = capacity, max_delay_ticks
        self.pending = {}
        self.records = OrderedDict()
        self.last_polled_tick = -1
        self.history = TimedHistory(game_id, player, SCHEMA_HASH, DIMENSION)

    def enqueue(self, packet, *, now_tick, drop=False):
        tick(now_tick)
        if type(drop) is not bool:
            raise ValueError("Explicit boolean drop disposition required")
        if (packet.game_id, packet.player, packet.schema_hash) != (
            self.game_id,
            self.player,
            SCHEMA_HASH,
        ):
            raise ValueError("Foreign sensor packet")
        previous = self.records.get(packet.sampled_tick)
        if previous is not None:
            if (
                previous["sha256"] != packet.hash
                or previous["dropped_at_source"] != drop
            ):
                raise ValueError("Conflicting sampled sensor identity")
            return False
        if (
            packet.sampled_tick != now_tick
            or now_tick not in {self.last_polled_tick, self.last_polled_tick + 1}
            or packet.delivery_tick - packet.sampled_tick > self.max_delay_ticks
        ):
            raise ValueError("Future/stale/missing-clock/excess-delay sensor packet")
        if not drop and len(self.pending) >= self.capacity:
            raise TimeoutError("Sensor queue capacity")
        self.records[packet.sampled_tick] = {
            "sampled_tick": packet.sampled_tick,
            "delivery_tick": packet.delivery_tick,
            "sha256": packet.hash,
            "received_tick": None,
            "disposition": "dropped" if drop else "pending",
            "dropped_at_source": drop,
        }
        if not drop:
            self.pending[packet.sampled_tick] = packet
        while len(self.records) > 128:
            protected = self.needed_ticks()
            old = next(t for t in self.records if t not in protected)
            del self.records[old]
        return True

    def pop_due(self, current_tick):
        tick(current_tick)
        if current_tick not in {self.last_polled_tick, self.last_polled_tick + 1}:
            raise ValueError("Missing or reversed sensor physics tick")
        result = []
        for p in sorted(
            self.pending.values(), key=lambda p: (p.delivery_tick, p.sampled_tick)
        ):
            if p.delivery_tick > current_tick:
                continue
            latest = max(self.history._samples, default=-1)
            self.history.receive(
                TimedSample(
                    self.game_id,
                    self.player,
                    SCHEMA_HASH,
                    p.sampled_tick,
                    current_tick,
                    p.values,
                ),
                now_tick=current_tick,
            )
            self.records[p.sampled_tick].update(
                received_tick=current_tick,
                disposition="delivered" if p.sampled_tick > latest else "late_history",
            )
            result.append(copy.deepcopy(self.records[p.sampled_tick]))
            del self.pending[p.sampled_tick]
        self.last_polled_tick = current_tick
        return result

    def actor_input(self, decision_tick):
        tick(decision_tick)
        if decision_tick != self.last_polled_tick:
            raise ValueError("Actor input requires the current received clock")
        h = self.history.at(decision_tick)
        available = bool(h.valid.any())
        observation = (
            None
            if not available
            else ActorObservation(
                "actor-visible",
                self.player,
                int(h.sampled_ticks[-1]),
                SCHEMA_HASH,
                h.values[-1],
            )
        )
        # Actual private game_id may encode an experiment seed. It never crosses
        # this boundary, nor do fingerprints, queued packets or bundle IDs.
        return ActorInput(
            decision_tick,
            observation,
            h,
            available,
            int(h.sample_age_ticks[-1]),
            int(h.receipt_age_ticks[-1]),
        )

    def needed_ticks(self):
        return set(self.pending) | set(self.history._samples)

    def snapshot(self):
        return {
            "schema": "joxing-sensor-queue-v1",
            "game_id": self.game_id,
            "player": self.player,
            "capacity": self.capacity,
            "max_delay_ticks": self.max_delay_ticks,
            "last_polled_tick": self.last_polled_tick,
            "pending": [p.snapshot() for _, p in sorted(self.pending.items())],
            "records": list(copy.deepcopy(self.records).values()),
            "history": self.history.snapshot(),
        }

    @classmethod
    def restore(cls, state, *, game_id, player):
        if state["schema"] != "joxing-sensor-queue-v1" or (
            state["game_id"],
            state["player"],
        ) != (game_id, player):
            raise ValueError("Foreign sensor snapshot")
        q = cls(
            game_id,
            player,
            capacity=state["capacity"],
            max_delay_ticks=state["max_delay_ticks"],
        )
        n = state["last_polled_tick"]
        if (
            type(n) is not int
            or n < -1
            or len(state["records"]) > 128
            or len(state["pending"]) > q.capacity
        ):
            raise ValueError("Invalid sensor clock/receipt bound")
        q.last_polled_tick = n
        for r in state["records"]:
            t = tick(r["sampled_tick"])
            if (
                t in q.records
                or t > n + 1
                or not t <= tick(r["delivery_tick"]) <= t + q.max_delay_ticks
                or not isinstance(r["sha256"], str)
                or len(r["sha256"]) != 64
                or type(r["dropped_at_source"]) is not bool
            ):
                raise ValueError("Malformed sampled sensor receipt")
            if r["disposition"] in {"delivered", "late_history"}:
                if (
                    r["dropped_at_source"]
                    or not r["delivery_tick"] <= tick(r["received_tick"]) <= n
                ):
                    raise ValueError("Future/foreign receipt time")
            elif r["disposition"] in {"pending", "dropped"}:
                if (
                    r["received_tick"] is not None
                    or (r["disposition"] == "dropped") != r["dropped_at_source"]
                ):
                    raise ValueError("Invalid sensor disposition")
            else:
                raise ValueError("Unknown sensor disposition")
            q.records[t] = copy.deepcopy(r)
        for raw in state["pending"]:
            packet = SensorPacket.restore(raw)
            r = q.records.get(packet.sampled_tick)
            if (
                (packet.game_id, packet.player) != (game_id, player)
                or packet.sampled_tick in q.pending
                or packet.delivery_tick < n
                or r is None
                or r["sha256"] != packet.hash
                or r["disposition"] != "pending"
            ):
                raise ValueError("Pending sensor receipt mismatch")
            q.pending[packet.sampled_tick] = packet
        if set(q.pending) != {
            t for t, r in q.records.items() if r["disposition"] == "pending"
        }:
            raise ValueError("Missing pending sensor packet")
        q.history = TimedHistory.restore(
            state["history"], game_id=game_id, player=player, schema_hash=SCHEMA_HASH
        )
        for t, sample in q.history._samples.items():
            r = q.records.get(t)
            if (
                r is None
                or r["disposition"] not in {"delivered", "late_history"}
                or r["received_tick"] != sample.received_tick
                or sample.received_tick > n
                or SensorPacket(
                    game_id, player, SCHEMA_HASH, t, r["delivery_tick"], sample.values
                ).hash
                != r["sha256"]
            ):
                raise ValueError("History differs from actual delivered sensor")
        if q.snapshot() != state:
            raise ValueError("Noncanonical sensor snapshot")
        return q
