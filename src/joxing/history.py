"""Causal, per-game/per-corner bounded sensor history for the match harness.

This stores already filtered actor features. It does not turn private body
measurements into a safe actor observation, or change the frozen motor policy.
"""

from dataclasses import dataclass

import numpy as np

from .bodies.base import owned_readonly


def tick(value):
    if type(value) is not int or value < 0:
        raise ValueError("Ticks must be nonnegative integers")
    return value


@dataclass(frozen=True)
class TimedSample:
    game_id: str
    player: str
    schema_hash: str
    sampled_tick: int
    received_tick: int
    values: np.ndarray

    def __post_init__(self):
        if not self.game_id or self.player not in {"A", "B"} or not self.schema_hash:
            raise ValueError("Missing sample ownership/schema")
        if tick(self.sampled_tick) > tick(self.received_tick):
            raise ValueError("Sample cannot arrive before it was measured")
        if self.values.dtype != np.float32 or self.values.ndim != 1:
            raise ValueError("Actor features require an explicit float32 vector")
        object.__setattr__(self, "values", owned_readonly(self.values))


@dataclass(frozen=True)
class HistoryView:
    values: np.ndarray
    valid: np.ndarray
    sampled_ticks: np.ndarray
    sample_age_ticks: np.ndarray
    receipt_age_ticks: np.ndarray


class TimedHistory:
    def __init__(self, game_id, player, schema_hash, dimension, capacity=8):
        if not game_id or player not in {"A", "B"} or not schema_hash:
            raise ValueError("Missing history ownership/schema")
        if type(dimension) is not int or not 0 < dimension <= 16384:
            raise ValueError("Invalid history feature dimension")
        if type(capacity) is not int or not 0 < capacity <= 64:
            raise ValueError("Invalid history capacity")
        self.game_id, self.player, self.schema_hash = game_id, player, schema_hash
        self.dimension, self.capacity = dimension, capacity
        self._samples = {}

    def receive(self, sample, *, now_tick):
        tick(now_tick)
        if (
            (sample.game_id, sample.player, sample.schema_hash)
            != (self.game_id, self.player, self.schema_hash)
            or sample.values.shape != (self.dimension,)
            or sample.received_tick != now_tick
        ):
            raise ValueError("Foreign, malformed or incorrectly timed sensor packet")
        previous = self._samples.get(sample.sampled_tick)
        if previous is not None:
            if not np.array_equal(previous.values, sample.values):
                raise ValueError("Conflicting sample identity")
            return  # Retry preserves the first actual arrival time.
        self._samples[sample.sampled_tick] = sample
        for old in sorted(self._samples)[: -self.capacity]:
            del self._samples[old]

    def at(self, decision_tick):
        tick(decision_tick)
        available = [
            self._samples[t]
            for t in sorted(self._samples)
            if self._samples[t].received_tick <= decision_tick
        ]
        values = np.zeros((self.capacity, self.dimension), np.float32)
        valid = np.zeros(self.capacity, bool)
        sampled = np.full(self.capacity, -1, np.int64)
        age, receipt_age = sampled.copy(), sampled.copy()
        for index, sample in enumerate(available, self.capacity - len(available)):
            values[index] = sample.values
            valid[index] = True
            sampled[index] = sample.sampled_tick
            age[index] = decision_tick - sample.sampled_tick
            receipt_age[index] = decision_tick - sample.received_tick
        return HistoryView(
            *(owned_readonly(a) for a in (values, valid, sampled, age, receipt_age))
        )

    def snapshot(self):
        return {
            "schema": "joxing-timed-history-v1",
            "game_id": self.game_id,
            "player": self.player,
            "schema_hash": self.schema_hash,
            "dimension": self.dimension,
            "capacity": self.capacity,
            "samples": [
                {
                    "sampled_tick": s.sampled_tick,
                    "received_tick": s.received_tick,
                    "values": s.values.tolist(),
                }
                for _, s in sorted(self._samples.items())
            ],
        }

    @classmethod
    def restore(cls, state, *, game_id, player, schema_hash):
        if (
            state["schema"] != "joxing-timed-history-v1"
            or (state["game_id"], state["player"], state["schema_hash"])
            != (game_id, player, schema_hash)
            or len(state["samples"]) > state["capacity"]
        ):
            raise ValueError("Foreign history snapshot")
        result = cls(
            game_id, player, schema_hash, state["dimension"], state["capacity"]
        )
        for row in state["samples"]:
            if row["sampled_tick"] in result._samples:
                raise ValueError("Duplicate saved sample identity")
            result.receive(
                TimedSample(
                    game_id,
                    player,
                    schema_hash,
                    row["sampled_tick"],
                    row["received_tick"],
                    np.array(row["values"], np.float32),
                ),
                now_tick=row["received_tick"],
            )
        return result
