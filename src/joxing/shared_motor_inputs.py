"""Immutable native teacher samples and scalar paired inference.

No engine stepping, forward recomputation, reset or replay-state surgery. Full
reference phase and source-engine validation remain a separate integration gate.
The input cache must be captured at the native motor boundary BEFORE any later
mj_forward recomputes acceleration/forces. It is not a1970game observation.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from .provenance import digest
from .shared_motor import native_teacher_observation

SCHEMA = "joxing-native-motor-input-v1"
FIELDS = {
    "schema",
    "player",
    "game_id",
    "sampled_tick",
    "dtype",
    "values",
    "values_sha256",
    "reference_sha256",
}


def values_sha(values):
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class MotorPolicyInput:
    player: str
    game_id: str
    sampled_tick: int
    values: np.ndarray
    reference_sha256: str

    def __post_init__(self):
        if (
            self.player not in ("A", "B")
            or type(self.game_id) is not str
            or not 1 <= len(self.game_id) <= 128
        ):
            raise ValueError("Motor input ownership")
        if (
            type(self.sampled_tick) is not int
            or self.sampled_tick < 0
            or self.sampled_tick % 20
        ):
            raise ValueError("Native20ms motor sample tick")
        if (
            type(self.reference_sha256) is not str
            or len(self.reference_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.reference_sha256)
        ):
            raise ValueError("Motor reference identity")
        values = np.asarray(self.values)
        if (
            values.dtype != np.float32
            or values.shape != (3601,)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Native cached3601float32 input")
        # Bytes-backed arrays cannot have their writable flag re-enabled.
        frozen = np.frombuffer(values.tobytes(order="C"), dtype=np.float32)
        object.__setattr__(self, "values", frozen)

    def snapshot(self):
        return {
            "schema": SCHEMA,
            "player": self.player,
            "game_id": self.game_id,
            "sampled_tick": self.sampled_tick,
            "dtype": self.values.dtype.str,
            "values": self.values.tolist(),
            "values_sha256": values_sha(self.values),
            "reference_sha256": self.reference_sha256,
        }

    @classmethod
    def restore(cls, value):
        if (
            type(value) is not dict
            or set(value) != FIELDS
            or value["schema"] != SCHEMA
            or value["dtype"] != np.dtype(np.float32).str
        ):
            raise ValueError("Motor sample schema/dtype")
        # JSON numbers must roundtrip losslessly to float32; reject hidden extra
        # precision instead of silently downcasting an invented observation.
        raw = np.asarray(value["values"])
        values = np.asarray(value["values"], dtype=np.float32)
        if raw.shape != (3601,) or not np.array_equal(raw, values.astype(np.float64)):
            raise ValueError("Motor sample precision/shape")
        result = cls(
            value["player"],
            value["game_id"],
            value["sampled_tick"],
            values,
            value["reference_sha256"],
        )
        if values_sha(result.values) != value["values_sha256"]:
            raise ValueError("Motor sample byte digest")
        return result


def reference_sha(reference):
    return digest(
        {
            "qpos": reference.qpos.tolist(),
            "future_qpos": reference.future_qpos.tolist(),
            "keypoints_relative_pelvis": reference.keypoints_relative_pelvis.tolist(),
        }
    )


def capture_pair(data, maps, references, *, game_id, tick):
    if set(maps) != {"A", "B"} or set(references) != {"A", "B"}:
        raise ValueError("Both motor corners required")
    # Capture BOTH actual native raw inputs before any inference or callback.
    return {
        a: MotorPolicyInput(
            a,
            game_id,
            tick,
            native_teacher_observation(data, maps[a], references[a])[0],
            reference_sha(references[a]),
        )
        for a in ("A", "B")
    }


def validate_pair(inputs, game_id, tick):
    if set(inputs) != {"A", "B"}:
        raise ValueError("Both motor inputs required")
    for a, sample in inputs.items():
        if (
            not isinstance(sample, MotorPolicyInput)
            or sample.player != a
            or sample.game_id != game_id
            or sample.sampled_tick != tick
        ):
            raise ValueError("Motor input mixed owner/game/time")


class FrozenPairTeacher:
    """Two scalar calls using one verified frozen actor and immutable samples.

    Production construction uses from_inputs, which reuses hash-checked v6/v7
    loading. Direct construction supports explicit unit fixtures; it does not
    certify arbitrary model objects or claim a physical motor result.
    """

    def __init__(self, model, normalizer):
        self.model, self.normalizer = model, normalizer

    @classmethod
    def from_inputs(cls, directory):
        from pathlib import Path

        from .native_transfer import load_controller

        return cls(*load_controller(Path(directory)))

    def infer_pair(self, inputs, *, game_id, tick):
        validate_pair(inputs, game_id, tick)
        self.normalizer.check()
        commands = {}
        for a in ("A", "B"):
            normalized = self.normalizer.apply(inputs[a].values)
            action, hidden = self.model.predict(normalized, deterministic=True)
            value = np.asarray(action)
            if (
                hidden is not None
                or value.dtype != np.float32
                or value.shape != (700,)
                or not np.isfinite(value).all()
                or np.max(abs(value)) > 1
            ):
                raise ValueError("Frozen native teacher action contract")
            # Keep normalized700 output; the environment applies sigmoid once.
            commands[a] = np.frombuffer(value.tobytes(), dtype=np.float32)
        self.normalizer.check()
        return commands
