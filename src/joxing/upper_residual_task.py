"""Opt-in shared motor composition and replay; diagnostic arm policy only.

The original teacher, references, scene, excitation and physical task are reused.
This module supplies neither a boxing reward nor an optimizer or balance claim.
"""

import json
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .env import excitation
from .provenance import digest
from .shared_motor_inputs import MotorPolicyInput, validate_pair
from .shared_motor_task import SharedMotorTask
from .upper_residual import HeldUpperCommand, checked, compose_pair

PLAYERS = ("A", "B")
SCHEMA = "joxing-shared-upper-residual-replay-v1"


def sealed(value):
    return value | {"sha256": digest(value)}


def unseal(value, fields):
    if (
        type(value) is not dict
        or set(value) != set(fields) | {"sha256"}
        or digest({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]
    ):
        raise ValueError("Upper task snapshot fields/digest")
    return value


def generators(states):
    if type(states) is not dict or set(states) != set(PLAYERS):
        raise ValueError("Arm policy private RNG owners")
    result = {}
    for a in PLAYERS:
        g = np.random.Generator(np.random.PCG64())
        try:
            g.bit_generator.state = states[a]
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("Arm policy PCG64 state") from error
        if digest(g.bit_generator.state) != digest(states[a]):
            raise ValueError("Arm policy RNG state precision/fields")
        result[a] = g
    return result


@dataclass(frozen=True)
class DiagnosticArmPolicy:
    """Private A/B streams, immutable state and pure proposals; not trained RL."""

    encoded: bytes

    def __post_init__(self):
        if type(self.encoded) is not bytes:
            raise ValueError("Arm policy immutable state")
        value = self.snapshot()
        c = value["contract"]
        if (
            type(c) is not dict
            or set(c) != {"variant", "seed", "dimension", "amplitude"}
            or c["variant"] != "diagnostic-private-uniform-arm-v1"
            or type(c["seed"]) is not int
            or not 0 <= c["seed"] < 2**32
            or type(c["dimension"]) is not int
            or not 1 <= c["dimension"] <= 138
            or type(c["amplitude"]) is not float
            or not np.isfinite(c["amplitude"])
            or not 0 <= c["amplitude"] <= 1
            or type(value["decisions"]) is not int
            or value["decisions"] < 0
        ):
            raise ValueError("Arm diagnostic policy contract")
        if value["decisions"] == 0:
            if value["last_tick"] is not None or value["game_id"] is not None:
                raise ValueError("Arm policy initial clock")
        elif (
            type(value["last_tick"]) is not int
            or value["last_tick"] != (value["decisions"] - 1) * 20
            or type(value["game_id"]) is not str
            or not 1 <= len(value["game_id"]) <= 128
        ):
            raise ValueError("Arm policy decision clock")
        generators(value["rng"])
        # This named diagnostic draws exactly dimension float64 uniforms per
        # owner/decision. A re-signed RNG state must still match seed/progress.
        for a, seed in zip(
            PLAYERS, np.random.SeedSequence(c["seed"]).spawn(2), strict=True
        ):
            bitgen = np.random.PCG64(seed)
            bitgen.advance(c["dimension"] * value["decisions"])
            if digest(bitgen.state) != digest(value["rng"][a]):
                raise ValueError("Arm policy seed/decision RNG progress")

    @classmethod
    def start(cls, *, seed, dimension, amplitude=0.0):
        states = {
            a: np.random.PCG64(s).state
            for a, s in zip(PLAYERS, np.random.SeedSequence(seed).spawn(2), strict=True)
        }
        return cls.from_snapshot(
            sealed(
                {
                    "contract": {
                        "variant": "diagnostic-private-uniform-arm-v1",
                        "seed": seed,
                        "dimension": dimension,
                        "amplitude": amplitude,
                    },
                    "game_id": None,
                    "last_tick": None,
                    "decisions": 0,
                    "rng": states,
                }
            )
        )

    @classmethod
    def from_snapshot(cls, value, expected_contract=None):
        if expected_contract is not None and value.get("contract") != expected_contract:
            raise ValueError("Arm policy behavior bundle")
        return cls(json.dumps(value, sort_keys=True, allow_nan=False).encode())

    def snapshot(self):
        return unseal(
            json.loads(self.encoded),
            {"contract", "game_id", "last_tick", "decisions", "rng"},
        )

    @property
    def contract(self):
        return self.snapshot()["contract"]

    def last_actions(self):
        value = self.snapshot()
        if not value["decisions"]:
            return None
        c = value["contract"]
        result = {}
        for a, seed in zip(
            PLAYERS, np.random.SeedSequence(c["seed"]).spawn(2), strict=True
        ):
            bitgen = np.random.PCG64(seed)
            bitgen.advance(c["dimension"] * (value["decisions"] - 1))
            result[a] = (
                np.random.Generator(bitgen).uniform(-1, 1, c["dimension"])
                * c["amplitude"]
            ).astype(np.float32)
        return result

    def propose(self, inputs, *, game_id, tick):
        validate_pair(inputs, game_id, tick)
        saved = self.snapshot()
        if tick != saved["decisions"] * 20 or (
            saved["game_id"] is not None and saved["game_id"] != game_id
        ):
            raise ValueError("Arm policy causal decision clock/owner")
        gs = generators(saved["rng"])
        c = saved["contract"]
        actions = {
            a: (gs[a].uniform(-1, 1, c["dimension"]) * c["amplitude"]).astype(
                np.float32
            )
            for a in PLAYERS
        }
        next_policy = type(self).from_snapshot(
            sealed(
                {
                    "contract": c,
                    "game_id": game_id,
                    "last_tick": tick,
                    "decisions": saved["decisions"] + 1,
                    "rng": {a: gs[a].bit_generator.state for a in PLAYERS},
                }
            ),
            expected_contract=c,
        )
        return actions, next_policy


class UpperPairTeacher:
    """Armed only for one paired decision; never called during a held interval."""

    def __init__(self, teacher, spec):
        self.teacher, self.spec = teacher, spec
        self.pending = self.prepared = None

    def stage(self, actions, *, game_id, tick):
        if (
            self.pending is not None
            or self.prepared is not None
            or set(actions) != set(PLAYERS)
        ):
            raise ValueError("Upper teacher paired staging")
        frozen = {
            a: checked(actions[a], len(self.spec.active_groups)).tobytes()
            for a in PLAYERS
        }
        self.pending = (game_id, tick, frozen)

    def infer_pair(self, inputs, *, game_id, tick):
        validate_pair(inputs, game_id, tick)
        if self.pending is None or self.pending[:2] != (game_id, tick):
            raise ValueError("Upper teacher unarmed or foreign decision")
        actions = {a: np.frombuffer(v, np.float32) for a, v in self.pending[2].items()}
        self.pending = None  # One shot, including a teacher exception; never retry it.
        teachers = self.teacher.infer_pair(inputs, game_id=game_id, tick=tick)
        records = compose_pair(
            teachers, actions, self.spec, game_id=game_id, sampled_tick=tick
        )
        self.prepared = records
        return {
            a: r.command_at(actor=a, game_id=game_id, engine_tick=tick)
            for a, r in records.items()
        }

    def clear(self):
        self.pending = self.prepared = None


class UpperResidualTask:
    """Composition owns policy/records; unchanged SharedMotorTask owns dynamics."""

    def __init__(
        self, scene, drivers, teacher, policy, spec, *, horizon_ticks, teacher_bundle
    ):
        if not teacher_bundle or any(
            type(v) is not str
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for v in teacher_bundle.values()
        ):
            raise ValueError("Explicit teacher bundle hashes required")
        self.spec, self.teacher = spec, teacher
        self._policy = DiagnosticArmPolicy.from_snapshot(policy.snapshot())
        if self._policy.contract["dimension"] != len(spec.active_groups):
            raise ValueError("Arm policy/mask action dimension")
        self.adapter = UpperPairTeacher(teacher, spec)
        self._contract = json.dumps(
            {
                "schema": "joxing-shared-upper-residual-task-v1",
                "spec_sha256": spec.sha256,
                "policy": self._policy.contract,
                "teacher_bundle": teacher_bundle,
                "competition": False,
            },
            sort_keys=True,
        ).encode()
        self.base = SharedMotorTask(
            scene,
            drivers,
            self.adapter,
            horizon_ticks=horizon_ticks,
            bundle_hash=digest(json.loads(self._contract)),
        )
        self.contract_hash = digest(
            {"upper": json.loads(self._contract), "base": self.base.contract_hash}
        )
        self._records = self._record_inputs = None
        self.invalid = False

    @property
    def policy(self):
        return self._policy

    @property
    def records(self):
        return self._records

    def _check(self):
        self.spec.check()
        c = json.loads(self._contract)
        if (
            self.invalid
            or self.base.teacher is not self.adapter
            or self.adapter.teacher is not self.teacher
            or self.adapter.spec is not self.spec
            or digest(self.base.contract) != self.base.contract_hash
            or self._policy.contract != c["policy"]
            or c["spec_sha256"] != self.spec.sha256
            or digest({"upper": c, "base": self.base.contract_hash})
            != self.contract_hash
        ):
            raise ValueError("Upper task bundle/policy/adapter drift or invalid")

    def begin_control(self):
        self._check()
        if self.base.remaining or self.base.done or self.base.invalid:
            raise RuntimeError("Upper task decision during hold or ending")
        self.base._check()
        actions, proposed = self._policy.propose(
            self.base.inputs, game_id=self.base.game_id, tick=self.base.tick
        )
        self.adapter.stage(actions, game_id=self.base.game_id, tick=self.base.tick)
        try:
            commands = self.base.begin_control()
            records = self.adapter.prepared
        except Exception:
            self.invalid = True  # No retry after an incomplete teacher call.
            raise
        finally:
            self.adapter.clear()
        # Pure proposal/record validation preceded the base's paired ctrl commit.
        self._policy = proposed
        self._records = MappingProxyType(records)
        self._record_inputs = MappingProxyType(dict(self.base.inputs))
        return commands

    def advance_substep(self, guard):
        self._check()
        if not self.base.remaining or self._records is None:
            raise RuntimeError("Upper task has no held command")
        for a, record in self._records.items():
            command = record.command_at(
                actor=a, game_id=self.base.game_id, engine_tick=self.base.tick
            )
            if command.tobytes() != self.base.commands[a].tobytes():
                raise ValueError("Upper task held record drift")
        return self.base.advance_substep(guard)

    def _decode_records(self, base, saved, policy):
        if saved is None:
            if (
                base["tick"] != 0
                or base["remaining"] != 0
                or base["commands"] is not None
                or policy.snapshot()["decisions"] != 0
            ):
                raise ValueError("Upper task initial record ownership")
            return None, None
        if set(saved) != set(PLAYERS):
            raise ValueError("Upper task held record owners")
        tick, remaining = base["tick"], base["remaining"]
        sample = tick + remaining - 20 if remaining else tick - 20
        state = policy.snapshot()
        if state["last_tick"] != sample or state["game_id"] != self.base.game_id:
            raise ValueError("Upper task policy/hold clock")
        records, inputs = {}, {}
        expected_actions = policy.last_actions()
        for a in PLAYERS:
            if set(saved[a]) != {"held", "input"}:
                raise ValueError("Upper task held record fields")
            records[a] = HeldUpperCommand.restore(
                saved[a]["held"],
                self.spec,
                actor=a,
                game_id=self.base.game_id,
                sampled_tick=sample,
            )
            if records[a].residual_bytes != expected_actions[a].tobytes():
                raise ValueError("Upper task policy/held residual mismatch")
            inputs[a] = MotorPolicyInput.restore(saved[a]["input"])
            raw = np.asarray(base["commands"][a], np.float64)
            command = np.frombuffer(records[a].command_bytes, np.float32)
            if (
                raw.shape != (700,)
                or not np.array_equal(raw, command.astype(np.float64))
                or raw.astype(np.float32).tobytes() != records[a].command_bytes
            ):
                raise ValueError("Upper task base/record command mismatch")
            if remaining and saved[a]["input"] != base["inputs"][a]:
                raise ValueError("Upper task held causal input mismatch")
        validate_pair(inputs, self.base.game_id, sample)
        return MappingProxyType(records), MappingProxyType(inputs)

    def snapshot(self):
        self._check()
        base = self.base.snapshot()
        saved = (
            None
            if self._records is None
            else {
                a: {
                    "held": self._records[a].snapshot(),
                    "input": self._record_inputs[a].snapshot(),
                }
                for a in PLAYERS
            }
        )
        self._decode_records(base, saved, self._policy)
        if self._records is not None:
            for a, r in self._records.items():
                command = np.frombuffer(r.command_bytes, np.float32)
                ctrl = self.base.data.ctrl[self.base.maps[a].actuator_ids]
                if ctrl.tobytes() != excitation(command).astype(ctrl.dtype).tobytes():
                    raise ValueError("Upper task held engine control mismatch")
        return sealed(
            {
                "schema": SCHEMA,
                "contract_hash": self.contract_hash,
                "base": base,
                "records": saved,
                "policy": self._policy.snapshot(),
            }
        )

    def restore(self, snapshot):
        self._check()
        saved = unseal(
            snapshot, {"schema", "contract_hash", "base", "records", "policy"}
        )
        if saved["schema"] != SCHEMA or saved["contract_hash"] != self.contract_hash:
            raise ValueError("Upper task foreign replay bundle")
        proposed = DiagnosticArmPolicy.from_snapshot(
            saved["policy"], expected_contract=json.loads(self._contract)["policy"]
        )
        records, inputs = self._decode_records(
            saved["base"], saved["records"], proposed
        )
        self.base.restore(saved["base"])  # Existing atomic engine/reference validation.
        self._policy, self._records, self._record_inputs = proposed, records, inputs
