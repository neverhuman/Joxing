"""Opt-in arm-only normalized command composition, not a balance guarantee.

All700 outputs remain present. The frozen teacher owns every protected channel;
musculoskeletal coupling can still affect balance. No task, clock, normalizer,
actor, reference phase, excitation transform or existing decoder is modified.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .motor_residual import AMPLITUDE, expand_groups, validate_groups

VARIANT = "native700-arm-file-residual-v1"
SCHEMA = "joxing-held-upper-residual-v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def checked(value, length):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != (length,)
        or not np.isfinite(value).all()
        or np.any(abs(value) > 1)
    ):
        raise ValueError("Expected bounded float32 normalized command/residual")
    return value


@dataclass(frozen=True)
class UpperResidualSpec:
    actuator_names: tuple[str, ...]
    groups: tuple[tuple[int, ...], ...]
    selected_ids: tuple[int, ...]
    source_hashes: tuple[tuple[str, str], ...]
    active_groups: tuple[int, ...] = field(init=False)
    protected_ids: tuple[int, ...] = field(init=False)
    mixed_groups: tuple[int, ...] = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self):
        names = tuple(self.actuator_names)
        groups = validate_groups(self.groups)
        selected = tuple(self.selected_ids)
        sources = tuple(tuple(item) for item in self.source_hashes)
        if (
            len(names) != 700
            or len(set(names)) != 700
            or any(type(n) is not str or not n for n in names)
        ):
            raise ValueError("Upper residual native actuator name order")
        if (
            not selected
            or len(selected) >= 700
            or any(type(i) is not int or not 0 <= i < 700 for i in selected)
            or tuple(sorted(set(selected))) != selected
        ):
            raise ValueError("Upper residual unique sorted muscle selection")
        if (
            not sources
            or any(
                len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
                or len(item[1]) != 64
                or any(c not in "0123456789abcdef" for c in item[1])
                for item in sources
            )
            or len({item[0] for item in sources}) != len(sources)
        ):
            raise ValueError("Upper residual source binding")
        eligible = set(selected)
        active = tuple(i for i, g in enumerate(groups) if eligible.intersection(g))
        protected = tuple(i for i in range(700) if i not in eligible)
        mixed = tuple(i for i in active if not set(groups[i]) <= eligible)
        for name, value in (
            ("actuator_names", names),
            ("groups", groups),
            ("selected_ids", selected),
            ("source_hashes", sources),
            ("active_groups", active),
            ("protected_ids", protected),
            ("mixed_groups", mixed),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "sha256", digest(self.descriptor()))

    @classmethod
    def from_native_model(cls, model, groups, source_root):
        import mujoco

        root = Path(source_root)
        if (model.nq, model.nv, model.nu, model.na) != (85, 85, 700, 700):
            raise ValueError("Upper residual requires the intact native700 model")
        files = ["MS-Human-700.xml"]
        ordered, arms, tendon_names = [], [], []
        for inc in ET.parse(root / files[0]).getroot().findall("include"):
            name = inc.get("file", "")
            if not name.startswith("Muscle/"):
                continue
            if name not in {
                "Muscle/Muscle_Leg_r.xml",
                "Muscle/Muscle_Leg_l.xml",
                "Muscle/Muscle_Arm_r.xml",
                "Muscle/Muscle_Arm_l.xml",
                "Muscle/Muscle_Torso.xml",
            }:
                raise ValueError("Unregistered native muscle include")
            files.append(name)
            actuator = ET.parse(root / name).getroot().find("actuator")
            if actuator is None:
                raise ValueError("Native muscle include lacks actuators")
            for node in actuator:
                if node.tag != "general" or node.get("class") != "muscle":
                    raise ValueError("Native muscle actuator kind")
                ordered.append(node.attrib["name"])
                tendon_names.append(node.attrib["tendon"])
                if name in ("Muscle/Muscle_Arm_r.xml", "Muscle/Muscle_Arm_l.xml"):
                    arms.append(node.attrib["name"])
        actual = tuple(model.actuator(i).name for i in range(700))
        if actual != tuple(ordered) or len(arms) != 122 or len(files) != 6:
            raise ValueError("Native compiled/source actuator order or arm membership")
        for i, tendon in enumerate(tendon_names):
            if (
                model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON
                or model.tendon(int(model.actuator_trnid[i, 0])).name != tendon
                or model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_MUSCLE
            ):
                raise ValueError("Native actuator tendon/muscle binding")
        sources = tuple(
            (n, hashlib.sha256((root / n).read_bytes()).hexdigest()) for n in files
        )
        return cls(
            actual,
            groups,
            tuple(i for i, name in enumerate(actual) if name in set(arms)),
            sources,
        )

    def descriptor(self):
        return {
            "variant": VARIANT,
            "actuator_names": list(self.actuator_names),
            "groups": [list(g) for g in self.groups],
            "selected_ids": list(self.selected_ids),
            "source_hashes": dict(self.source_hashes),
            "active_groups": list(self.active_groups),
            "protected_ids": list(self.protected_ids),
            "mixed_groups": list(self.mixed_groups),
            "amplitude": AMPLITUDE,
            "action_dimension": len(self.active_groups),
            "command_domain": "normalized700float32; caller applies existing excitation exactly once",
            "semantics": "Source arm-file membership; protected commands unchanged, physical coupling not prevented",
        }

    def check(self):
        if digest(self.descriptor()) != self.sha256:
            raise ValueError("Frozen upper residual specification drift")


def compose(teacher, residual, spec):
    if not isinstance(spec, UpperResidualSpec):
        raise TypeError("Upper residual specification required")
    spec.check()
    teacher = checked(teacher, 700)
    residual = checked(residual, len(spec.active_groups))
    latent = np.zeros(138, np.float32)
    latent[list(spec.active_groups)] = residual
    correction = np.float32(AMPLITUDE) * expand_groups(latent, spec.groups)
    # Never add zero to protected channels: preserve signed zero and every bit.
    active = np.zeros(700, bool)
    active[list(spec.selected_ids)] = True
    active &= correction != 0
    output = teacher.copy()
    output[active] = np.clip(teacher[active] + correction[active], -1, 1)
    return output


@dataclass(frozen=True)
class HeldUpperCommand:
    spec: UpperResidualSpec
    actor: str
    game_id: str
    sampled_tick: int
    teacher_bytes: bytes
    residual_bytes: bytes
    command_bytes: bytes

    def __post_init__(self):
        if (
            self.actor not in ("A", "B")
            or type(self.game_id) is not str
            or not 1 <= len(self.game_id) <= 128
            or type(self.sampled_tick) is not int
            or self.sampled_tick < 0
            or self.sampled_tick % 20
        ):
            raise ValueError("Held upper command owner/game/sample clock")
        if any(
            type(v) is not bytes
            for v in (self.teacher_bytes, self.residual_bytes, self.command_bytes)
        ):
            raise ValueError("Held upper command immutable bytes required")
        teacher = np.frombuffer(self.teacher_bytes, np.float32)
        residual = np.frombuffer(self.residual_bytes, np.float32)
        expected = compose(teacher, residual, self.spec)
        if self.command_bytes != expected.tobytes():
            raise ValueError("Held upper command decoder differs")

    @classmethod
    def create(cls, teacher, residual, spec, *, actor, game_id, sampled_tick):
        command = compose(teacher, residual, spec)
        return cls(
            spec,
            actor,
            game_id,
            sampled_tick,
            teacher.tobytes(),
            residual.tobytes(),
            command.tobytes(),
        )

    def command_at(self, *, actor, game_id, engine_tick):
        self.spec.check()
        if (
            actor != self.actor
            or game_id != self.game_id
            or type(engine_tick) is not int
            or not self.sampled_tick <= engine_tick < self.sampled_tick + 20
        ):
            raise ValueError("Held upper command ownership/interval")
        return np.frombuffer(self.command_bytes, np.float32)

    def snapshot(self):
        self.spec.check()
        result = {
            "schema": SCHEMA,
            "spec_sha256": self.spec.sha256,
            "actor": self.actor,
            "game_id": self.game_id,
            "sampled_tick": self.sampled_tick,
            "physics_dt_s": 0.001,
            "hold_ticks": 20,
        }
        for name, value in (
            ("teacher", self.teacher_bytes),
            ("residual", self.residual_bytes),
            ("command", self.command_bytes),
        ):
            result[name] = np.frombuffer(value, np.float32).tolist()
            result[name + "_sha256"] = hashlib.sha256(value).hexdigest()
        return result | {"sha256": digest(result)}

    @classmethod
    def restore(cls, saved, spec, *, actor, game_id, sampled_tick):
        fields = {
            "schema",
            "spec_sha256",
            "actor",
            "game_id",
            "sampled_tick",
            "teacher",
            "residual",
            "command",
            "teacher_sha256",
            "residual_sha256",
            "command_sha256",
            "sha256",
            "physics_dt_s",
            "hold_ticks",
        }
        if (
            type(saved) is not dict
            or set(saved) != fields
            or saved["schema"] != SCHEMA
            or saved["spec_sha256"] != spec.sha256
            or saved["physics_dt_s"] != 0.001
            or saved["hold_ticks"] != 20
            or digest({k: v for k, v in saved.items() if k != "sha256"})
            != saved["sha256"]
            or (saved["actor"], saved["game_id"], saved["sampled_tick"])
            != (actor, game_id, sampled_tick)
        ):
            raise ValueError("Held upper snapshot identity")
        payloads = {}
        for name, length in (
            ("teacher", 700),
            ("residual", len(spec.active_groups)),
            ("command", 700),
        ):
            raw = np.asarray(saved[name], np.float64)
            value = np.asarray(saved[name], np.float32)
            checked(value, length)
            if (
                raw.shape != (length,)
                or not np.array_equal(raw, value.astype(np.float64))
                or array_sha(value) != saved[name + "_sha256"]
            ):
                raise ValueError("Held upper snapshot precision/payload")
            payloads[name] = value.tobytes()
        return cls(
            spec,
            actor,
            game_id,
            sampled_tick,
            payloads["teacher"],
            payloads["residual"],
            payloads["command"],
        )


def compose_pair(teachers, residuals, spec, *, game_id, sampled_tick):
    if set(teachers) != {"A", "B"} or set(residuals) != {"A", "B"}:
        raise ValueError("Both upper residual command owners required")
    return {
        a: HeldUpperCommand.create(
            teachers[a],
            residuals[a],
            spec,
            actor=a,
            game_id=game_id,
            sampled_tick=sampled_tick,
        )
        for a in "AB"
    }
