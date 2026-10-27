"""Explicit parallel-walking motor variant; not competition or learned boxing.

Both intact athletes share one MjData. Reset is the ONLY place poses/velocities
are assigned. Every evaluated substep is muscle driven. Native references are
known self-goals, not kinematic writes or future opponent observations.
"""

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from .env import INTEGRATION, excitation
from .physics import step_contacts
from .provenance import digest
from .scene import build_scene
from .shared_motor import KEYPOINTS, MotorReferenceState, NativeMotorMap
from .shared_motor_inputs import (
    MotorPolicyInput,
    capture_pair,
    reference_sha,
    validate_pair,
)

VARIANT = "pretrained-parallel-walk-1ms-v1"
PLAYERS = ("A", "B")


def build_motor_scene():
    return build_scene(spawn_variant="parallel-walk-v1")


@dataclass
class NativeReferenceDriver:
    """A native reset donor/reference timeline; its engine is never stepped."""

    task: object
    seed: int

    def __post_init__(self):
        self.task.reset(self.seed)
        if self.task.physics_total != 0:
            raise ValueError("Reset donor must not execute physics")
        env = self.task.env
        self.index = int(env.current_traj_index)
        self.init_time = float(env.init_time)
        self.terminate_time = float(env.terminate_time)
        self.cycles = int(env.cycles)
        self.initial = {
            "qpos": self.task.data.qpos.copy(),
            "qvel": self.task.data.qvel.copy(),
            "activation": self.task.data.act.copy(),
        }
        self.tick = 0
        self.sample_time = 0.0
        self.reference = self._state(env.qpos_ref, env.xpos_ref, env.qpos_ref_future)
        self.initial_reference = self.reference

    def _state(self, qpos, xpos, future):
        model = self.task.model
        key = [model.body(n).id for n in KEYPOINTS]
        pelvis = model.body("pelvis").id
        return MotorReferenceState(
            qpos, future, np.asarray(xpos)[key] - np.asarray(xpos)[pelvis]
        )

    def sample(self, tick, engine_time):
        if (
            type(tick) is not int
            or tick < self.tick
            or tick % 20
            or abs(engine_time - tick * 0.001) > 1e-9
        ):
            raise ValueError("Reference sample clock")
        if tick == 0:
            self.reference = self.initial_reference
        else:
            # Preserve native arithmetic order and its next-control target lead.
            time = engine_time + self.init_time + 0.02
            qb, xb, _ = self.task.env.trajectory.query_batch(
                time + np.arange(6, dtype=np.float64) * 0.02, self.index
            )
            self.reference = self._state(qb[0], xb[0], qb[1:6])
        self.tick = tick
        self.sample_time = float(engine_time)
        return self.reference

    def snapshot(self):
        return {
            "seed": self.seed,
            "index": self.index,
            "init_time": self.init_time,
            "terminate_time": self.terminate_time,
            "cycles": self.cycles,
            "tick": self.tick,
            "sample_time": self.sample_time,
            "qpos": self.reference.qpos.tolist(),
            "future_qpos": self.reference.future_qpos.tolist(),
            "keypoints_relative_pelvis": self.reference.keypoints_relative_pelvis.tolist(),
        }


class SharedMotorTask:
    """Motor-only shared physical loop; first proof must supply audit/replay gates.

    No scoring/competition API. The connected game environment remains separate;
    this explicit variant isolates native motor transfer and recording first.
    """

    def __init__(self, scene, drivers, teacher, *, horizon_ticks, bundle_hash):
        if (
            scene.report["profile"] != VARIANT
            or scene.model.opt.timestep != 0.001
            or scene.model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        ):
            raise ValueError("Named shared motor scene required")
        if (
            set(drivers) != set(PLAYERS)
            or type(horizon_ticks) is not int
            or not 20 <= horizon_ticks <= 20000
            or horizon_ticks % 20
            or not bundle_hash
        ):
            raise ValueError("Shared motor configuration")
        self.scene, self.model, self.drivers, self.teacher = (
            scene,
            scene.model,
            drivers,
            teacher,
        )
        self.horizon_ticks = horizon_ticks
        self.maps = {
            p: NativeMotorMap(self.model, drivers[p].task.model, p, spawn_yaw_rad=0.0)
            for p in PLAYERS
        }
        self.bundle_hash = bundle_hash
        self.data = mujoco.MjData(self.model)
        self.tick = self.remaining = self.physics_total = 0
        self.done = self.invalid = False
        self.commands = None
        self.game_id = (
            "motor-" + digest({p: drivers[p].snapshot() for p in PLAYERS})[:32]
        )
        mujoco.mj_resetData(self.model, self.data)
        for p in PLAYERS:
            m, initial = self.maps[p], drivers[p].initial
            self.data.qpos[m.qpos_ids] = initial["qpos"]
            self.data.qvel[m.qvel_ids] = initial["qvel"]
            self.data.act[m.activation_ids] = initial["activation"]
        # Native reset starts with zero excitation/activation; no normalized held
        # command exists yet. First inference at tick0 precedes the first step.
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._check()
        self.inputs = self._capture()
        self.contract = {
            "schema": "joxing-shared-motor-task-v1",
            "variant": VARIANT,
            "scene_hash": scene.report["scene_hash"],
            "horizon_ticks": horizon_ticks,
            "bundle_hash": bundle_hash,
            "initial_references": {p: drivers[p].snapshot() for p in PLAYERS},
            "transform": "normalized700->sigmoid once float32",
            "competition": False,
        }
        self.contract_hash = digest(self.contract)

    def state(self):
        result = np.empty(mujoco.mj_stateSize(self.model, INTEGRATION))
        mujoco.mj_getState(self.model, self.data, result, INTEGRATION)
        return result

    def _check(self):
        if (
            np.any(self.data.qfrc_applied)
            or np.any(self.data.xfrc_applied)
            or np.any(self.model.body_gravcomp)
            or not np.array_equal(self.model.opt.gravity, [0, 0, -9.81])
        ):
            raise ValueError("External motor assistance forbidden")
        if (
            any(w.number for w in self.data.warning)
            or not np.isfinite(self.state()).all()
        ):
            raise FloatingPointError("Invalid shared motor engine")

    def _capture(self):
        references = {
            p: self.drivers[p].sample(self.tick, float(self.data.time)) for p in PLAYERS
        }
        # Capture BOTH actors now. Do not call mj_forward after the physical
        # interval and before this capture: native cached forces/acceleration and
        # keypoint readout belong to the just-completed solver phase.
        return capture_pair(
            self.data, self.maps, references, game_id=self.game_id, tick=self.tick
        )

    def begin_control(self):
        if self.remaining or self.done or self.invalid:
            raise RuntimeError("No motor decision during held command or ending")
        if digest(self.contract) != self.contract_hash:
            raise ValueError("Shared motor contract drift")
        self._check()
        commands = self.teacher.infer_pair(
            self.inputs, game_id=self.game_id, tick=self.tick
        )
        if set(commands) != set(PLAYERS):
            raise ValueError("Both native commands required")
        staged = {}
        for p in PLAYERS:
            value = np.asarray(commands[p])
            if (
                value.dtype != np.float32
                or value.shape != (700,)
                or not np.isfinite(value).all()
                or np.max(abs(value)) > 1
            ):
                raise ValueError("Native muscle command")
            staged[p] = value.copy()
        # Validate both before either owner's control changes.
        self.commands = staged
        for p in PLAYERS:
            self.data.ctrl[self.maps[p].actuator_ids] = excitation(staged[p])
        self.remaining = 20
        return {p: c.copy() for p, c in staged.items()}

    def advance_substep(self, guard):
        if not self.remaining or self.done or self.invalid:
            raise RuntimeError("No held motor interval")
        self._check()
        for p in PLAYERS:
            if not np.array_equal(
                self.data.ctrl[self.maps[p].actuator_ids], excitation(self.commands[p])
            ):
                raise ValueError("Held native muscle drift")
        guard(1)  # Account conservatively BEFORE actual engine entry.
        self.physics_total += 1
        try:
            contacts = step_contacts(self.model, self.data)
            self.tick += 1
            self.remaining -= 1
            self._check()
            self.done = self.tick == self.horizon_ticks
            if not self.remaining:
                self.inputs = self._capture()
            return contacts
        except Exception:  # invalid attempts remain owned by the caller's book
            self.invalid = True
            raise

    def snapshot(self):
        self._check()
        if self.invalid or digest(self.contract) != self.contract_hash:
            raise ValueError("Invalid/drifted motor snapshot")
        payload = {
            "schema": "joxing-shared-motor-replay-v1",
            "contract_hash": self.contract_hash,
            "engine": self.state().tolist(),
            "tick": self.tick,
            "remaining": self.remaining,
            "done": self.done,
            "commands": None
            if self.commands is None
            else {p: a.tolist() for p, a in self.commands.items()},
            "inputs": {p: a.snapshot() for p, a in self.inputs.items()},
            "references": {p: d.snapshot() for p, d in self.drivers.items()},
        }
        return payload | {"sha256": digest(payload)}

    def restore(self, snapshot):
        value = copy.deepcopy(snapshot)
        sha = value.pop("sha256")
        fields = {
            "schema",
            "contract_hash",
            "engine",
            "tick",
            "remaining",
            "done",
            "commands",
            "inputs",
            "references",
        }
        if (
            set(value) != fields
            or digest(value) != sha
            or value["schema"] != "joxing-shared-motor-replay-v1"
            or value["contract_hash"] != self.contract_hash
        ):
            raise ValueError("Foreign motor snapshot")
        tick, remaining = value["tick"], value["remaining"]
        if (
            type(tick) is not int
            or type(remaining) is not int
            or not 0 <= tick <= self.horizon_ticks
            or not 0 <= remaining <= 20
            or (remaining and tick % 20 + remaining != 20)
            or (not remaining and tick % 20)
        ):
            raise ValueError("Motor replay clock")
        if (
            type(value["done"]) is not bool
            or value["done"] != (tick == self.horizon_ticks)
            or (value["done"] and remaining)
        ):
            raise ValueError("Motor replay ending")
        engine = np.asarray(value["engine"], np.float64)
        if engine.shape != self.state().shape or not np.isfinite(engine).all():
            raise ValueError("Motor replay engine")
        data = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, data, engine, INTEGRATION)
        if (
            abs(data.time - tick * 0.001) > 1e-9
            or np.any(data.qfrc_applied)
            or np.any(data.xfrc_applied)
        ):
            raise ValueError("Motor replay assistance/time")
        mujoco.mj_forward(self.model, data)
        if any(w.number for w in data.warning):
            raise ValueError("Motor replay engine warning")
        mujoco.mj_setState(self.model, data, engine, INTEGRATION)
        sample_tick = tick - (tick % 20)
        inputs = {p: MotorPolicyInput.restore(v) for p, v in value["inputs"].items()}
        validate_pair(inputs, self.game_id, sample_tick)
        if set(value["references"]) != set(PLAYERS):
            raise ValueError("Motor replay reference owners")
        drivers = {}
        for p in PLAYERS:
            driver = copy.copy(self.drivers[p])
            driver.tick = 0
            saved = value["references"][p]
            if saved["tick"] != sample_tick:
                raise ValueError("Motor replay reference clock")
            driver.sample(saved["tick"], saved["sample_time"])
            if (
                digest(driver.snapshot()) != digest(saved)
                or reference_sha(driver.reference) != inputs[p].reference_sha256
            ):
                raise ValueError("Motor replay reference identity/targets")
            raw = inputs[p].values
            expected = np.concatenate(
                [
                    driver.reference.qpos,
                    driver.reference.future_qpos.ravel(),
                    driver.reference.keypoints_relative_pelvis.ravel(),
                ]
            ).astype(np.float32)
            if raw[3073:].tobytes() != expected.tobytes():
                raise ValueError("Motor replay self-target input")
            drivers[p] = driver
        commands = value["commands"]
        if commands is None:
            if tick or remaining or np.any(data.ctrl):
                raise ValueError("Motor replay initial command")
        else:
            if set(commands) != set(PLAYERS):
                raise ValueError("Motor replay command owners")
            commands = {p: np.asarray(c, np.float32) for p, c in commands.items()}
            for p, c in commands.items():
                if (
                    c.shape != (700,)
                    or not np.isfinite(c).all()
                    or np.max(abs(c)) > 1
                    or not np.array_equal(
                        data.ctrl[self.maps[p].actuator_ids], excitation(c)
                    )
                ):
                    raise ValueError("Motor replay held command")
        # No step is hidden in restore; all validations precede atomic commit.
        self.data, self.drivers, self.inputs = data, drivers, inputs
        self.tick, self.remaining, self.done = tick, remaining, value["done"]
        self.commands, self.invalid = commands, False
        # Aggregate physics_total deliberately never rewinds on restore.
