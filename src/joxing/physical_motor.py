"""Substep physical continuation of a frozen walking controller, never boxing.

Only the named task termination changes. Native controls, rewards, reference
updates and cached actor features keep their original arithmetic and timing.
End-posture telemetry uses scratch kinematics, never a forward call on live data.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from .native_transfer import TransferTask, planned_rows
from .provenance import ROOT, digest
from .support import SupportMonitor

PROTOCOL_SHA = "8c5f14240ec58048fe257bf49af7a07ed138474633096a84f8624553954b20e4"
ROLES = ("right", "left", "nonfoot")
REASONS = ("nonfoot_support", "excess_tilt", "pelvis_drop")
INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION


def protocol():
    return json.loads((ROOT / "experiments/v0.0.8/protocol.json").read_text())


def validate_protocol(p):
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Registered v8 protocol changed")
    return copy.deepcopy(p)


@dataclass(frozen=True)
class PhysicalMotorConfig:
    task_variant: str = "tracking-independent-motor-continuation"
    horizon_s: float = 20.0
    nonfoot_weight_fraction: float = 0.1
    nonfoot_persistence_s: float = 0.05
    tilt_rad: float = np.pi / 3
    tilt_persistence_s: float = 0.1
    drop_m: float = 0.3
    drop_persistence_s: float = 0.1
    finish_held_control_interval_after_latch: bool = True

    def __post_init__(self):
        # SupportMonitor is the existing physical diagnostic, not a tunable game.
        if asdict(self) != {
            "task_variant": "tracking-independent-motor-continuation",
            "horizon_s": 20.0,
            "nonfoot_weight_fraction": 0.1,
            "nonfoot_persistence_s": 0.05,
            "tilt_rad": np.pi / 3,
            "tilt_persistence_s": 0.1,
            "drop_m": 0.3,
            "drop_persistence_s": 0.1,
            "finish_held_control_interval_after_latch": True,
        }:
            raise ValueError("Unknown physical motor task variant")


@dataclass(frozen=True)
class SubstepMotorEvidence:
    start_s: float
    end_s: float
    loads_N: tuple[float, float, float]
    tilt_rad: float
    drop_m: float
    # geom1, geom2, role code, signed world force on owned geom xyz, point xyz.
    contacts: tuple[tuple[float, ...], ...]

    @classmethod
    def from_dict(cls, value):
        v = copy.deepcopy(value)
        v["loads_N"] = tuple(v["loads_N"])
        v["contacts"] = tuple(tuple(c) for c in v["contacts"])
        return cls(**v)


def signed_floor_evidence(model, data, floor, roles):
    """Read the solved interval's force cache without altering live dynamics."""
    loads = np.zeros(3)
    contacts = []
    for slot, c in enumerate(data.contact):
        if c.efc_address < 0 or floor not in (c.geom1, c.geom2):
            continue
        other = int(c.geom2 if c.geom1 == floor else c.geom1)
        role = ROLES.index(roles[other])  # Missing/unknown ownership is a fault.
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, slot, wrench)
        force = c.frame.reshape(3, 3).T @ wrench[:3]
        if c.geom2 == floor:
            force = -force
        if not np.isfinite(np.r_[force, c.pos]).all():
            raise RuntimeError("Nonfinite substep contact evidence")
        loads[role] += max(0.0, float(force[2]))
        contacts.append((int(c.geom1), int(c.geom2), role, *force, *c.pos.copy()))
    return tuple(float(v) for v in loads), tuple(contacts)


class PhysicalEnvironmentMixin:
    def _step_mujoco_simulation(self, ctrl, n_frames):
        task = self._physical_task
        if not task.pending or n_frames != self.frame_skip:
            raise RuntimeError("Missing held native control")
        if not np.array_equal(np.asarray(ctrl), task.excitation):
            raise RuntimeError("Held control changed")
        task.advance_substeps(self.frame_skip - task.substep)
        # Preserve Gym's one final call, not one call at every substep.
        mujoco.mj_rnePostConstraint(self.model, self.data)
        error = float(np.mean(abs(self.data.qpos[3:] - self.qpos_ref[3:])))
        task.control_tracking_error = error
        if error > self.qpos_diff_th and task.first_tracking_failure_s is None:
            task.first_tracking_failure_s = float(self.data.time)

    @property
    def terminated(self):
        return self._physical_task.physical_failure is not None

    def step(self, action):
        obs, reward, terminated, _, info = super().step(action)
        horizon = self.data.time + 1e-9 >= self._physical_task.physical.horizon_s
        # Native reference/reward updates above remain intact. The diagnostic's
        # explicit physical horizon replaces the old tracking-task time limit.
        return obs, reward, terminated, bool(horizon and not terminated), info


class PhysicalMotorTask(TransferTask):
    def __init__(self, config, variant):
        self.physical = PhysicalMotorConfig()
        super().__init__(config, variant, environment_mixin=PhysicalEnvironmentMixin)
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("Physical evidence requires registered Euler integrator")
        self.env._physical_task = self
        self.scratch = mujoco.MjData(self.model)
        self.evidence = []
        self.capture_mid_snapshot = False
        self.registered_mid_snapshot = None

    def specification(self):
        result = super().specification()
        result.update(
            physical_motor=asdict(self.physical),
            support_telemetry="every physics interval; signed solved force at interval start, scratch end-qpos posture",
            floor_roles={str(k): v for k, v in self.roles.items()},
            cache_contract="live caches untouched; held action resumes without a new policy decision",
        )
        return result

    def reset(self, seed):
        obs = super().reset(seed)
        self.pending = False
        self.action = self.excitation = self.decision_qpos = None
        self.substep = self.completed_controls = 0
        self.control_start_s = float(self.data.time)
        self.first_tracking_failure_s = None
        self.physical_failure = self.physical_failure_s = None
        self.control_tracking_error = float(
            np.mean(abs(self.data.qpos[3:] - self.env.qpos_ref[3:]))
        )
        self.interval_evidence = []
        self.evidence = []
        self.registered_mid_snapshot = None
        self.capture_mid_snapshot = False
        return obs

    def begin_control(self, action):
        if self.done or self.pending:
            raise RuntimeError("Reset or resume pending native control required")
        self.check_assistance()
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (700,) or not np.isfinite(a).all() or np.max(abs(a)) > 1:
            raise ValueError("Malformed native muscle action")
        self.action = a.copy()
        self.excitation = 1 / (1 + np.exp(-5 * (a - 0.5)))
        self.decision_qpos = self.data.qpos.copy()
        self.control_start_s = float(self.data.time)
        self.data.ctrl[:] = self.excitation
        self.substep = 0
        self.interval_evidence = []
        self.pending = True

    def advance_substeps(self, count):
        if (
            not self.pending
            or type(count) is not int
            or not 0 <= count <= self.env.frame_skip - self.substep
        ):
            raise ValueError("Invalid pending substep count")
        for _ in range(count):
            self.check_assistance()
            if not np.array_equal(self.data.ctrl, self.excitation):
                raise RuntimeError("Held control changed")
            warnings = np.array([w.number for w in self.data.warning])
            before = float(self.data.time)
            self.physics_total += 1  # Count attempted physics even on a fault.
            mujoco.mj_step(self.model, self.data)
            if (
                abs(self.data.time - before - self.model.opt.timestep) > 1e-9
                or np.any(np.array([w.number for w in self.data.warning]) > warnings)
                or not np.isfinite(
                    np.r_[self.data.qpos, self.data.qvel, self.data.act]
                ).all()
            ):
                raise RuntimeError("Invalid physical motor substep")
            self.scratch.qpos[:] = self.data.qpos
            mujoco.mj_kinematics(self.model, self.scratch)
            up = self.scratch.xmat[self.pelvis].reshape(3, 3)[:, 1]
            tilt = float(np.arccos(np.clip(up[2], -1, 1)))
            drop = self.initial_height - float(self.scratch.xpos[self.pelvis, 2])
            loads, contacts = signed_floor_evidence(
                self.model, self.data, self.floor, self.roles
            )
            row = SubstepMotorEvidence(
                before, float(self.data.time), loads, tilt, drop, contacts
            )
            reason = self.monitor.update(
                dict(zip(ROLES, loads, strict=True)),
                self.weight,
                tilt,
                drop,
                float(self.model.opt.timestep),
            )
            self.max_tilt = max(self.max_tilt, tilt)
            if reason is not None and self.physical_failure is None:
                self.physical_failure = reason
                self.physical_failure_s = row.end_s
            self.sampled_failure = self.physical_failure
            self.interval_evidence.append(row)
            self.evidence.append(row)
            self.substep += 1
            if (
                self.capture_mid_snapshot
                and self.registered_mid_snapshot is None
                and self.completed_controls == 10
                and self.substep == 3
            ):
                self.registered_mid_snapshot = self.snapshot()

    def resume_control(self):
        if not self.pending:
            raise RuntimeError("No pending native control")
        self.check_assistance()
        obs, reward, terminated, truncated, info = self.env.step(self.excitation)
        if (
            abs(self.data.time - self.control_start_s - self.env.dt) > 1e-9
            or not np.isfinite(np.r_[obs, reward]).all()
        ):
            raise RuntimeError("Invalid physical motor control boundary")
        self.last_obs = obs.copy()
        self.done = bool(terminated or truncated)
        self.completed_controls += 1
        self.pending = False
        self.substep = 0
        self.control_start_s = float(self.data.time)
        self.action = self.excitation = self.decision_qpos = None
        return (
            obs.copy(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
            self.telemetry(),
        )

    def step(self, action):
        self.begin_control(action)
        return self.resume_control()

    def telemetry(self):
        if not self.interval_evidence:
            raise RuntimeError("No substep evidence at this boundary")
        row = self.interval_evidence[-1]
        # Scratch-only velocities at the end state; no actor feature/cache update.
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = self.data.qvel
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        mujoco.mj_comVel(self.model, self.scratch)
        velocities = {}
        for name in ("pelvis", "head_neck", "sternum", "hand_r", "hand_l"):
            value = np.empty(6)
            mujoco.mj_objectVelocity(
                self.model,
                self.scratch,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.body(name).id,
                value,
                0,
            )
            velocities[name] = value.tolist()
        return {
            "time_s": row.end_s,
            "tilt_rad": row.tilt_rad,
            "drop_m": row.drop_m,
            "floor_loads_N": dict(zip(ROLES, row.loads_N, strict=True)),
            "physical_failure": self.physical_failure,
            "physical_failure_s": self.physical_failure_s,
            "sampled_failure": self.physical_failure,
            "first_native_tracking_failure_s": self.first_tracking_failure_s,
            "native_qpos_error_before_reference_update": self.control_tracking_error,
            "native_qpos_mean_error": float(
                np.mean(abs(self.data.qpos[3:] - self.env.qpos_ref[3:]))
            ),
            "end_body_spatial_velocity_world": velocities,
        }

    def snapshot(self):
        result = super().snapshot()
        result["physical_control"] = {
            "pending": self.pending,
            "substep": self.substep,
            "completed_controls": self.completed_controls,
            "control_start_s": self.control_start_s,
            "action": self.action.tolist() if self.pending else None,
            "decision_qpos": self.decision_qpos.tolist() if self.pending else None,
            "first_tracking_failure_s": self.first_tracking_failure_s,
            "physical_failure": self.physical_failure,
            "physical_failure_s": self.physical_failure_s,
            "control_tracking_error": self.control_tracking_error,
            "interval_evidence": [asdict(r) for r in self.interval_evidence],
        }
        return result

    def extract_state(self, state, component):
        value = np.empty(mujoco.mj_stateSize(self.model, component))
        mujoco.mj_extractState(self.model, state, INTEGRATION, value, component)
        return value

    def restore(self, snapshot):
        """Validate everything on a separate task before touching live state.

        A mid-command cached actor input belongs to the previous decision, not
        the current qpos. Transport through NativeTask's boundary validator only
        after validating that decision pose and the exact held excitation.
        """
        s = copy.deepcopy(snapshot)
        if s.get("contract_hash") != self.contract_hash:
            raise ValueError("Snapshot contract mismatch")
        c = s.get("physical_control", {})
        if set(s["application"]) != {
            "qpos_ref",
            "xpos_ref",
            "qpos_ref_future",
            "init_time",
            "terminate_time",
            "current_traj_index",
        }:
            raise ValueError("Unknown native application field")
        state = np.asarray(s["state"], dtype=float)
        if (
            state.shape != (mujoco.mj_stateSize(self.model, INTEGRATION),)
            or not np.isfinite(state).all()
        ):
            raise ValueError("Malformed physical integration state")
        obs = np.asarray(s["last_raw_observation"], dtype=np.float32)
        pending = c.get("pending")
        index, completed = c.get("substep"), c.get("completed_controls")
        if (
            type(pending) is not bool
            or type(index) is not int
            or type(completed) is not int
            or completed < 0
            or not 0 <= index <= self.env.frame_skip
            or (not pending and index != 0)
            or obs.shape != (3601,)
            or not np.isfinite(obs).all()
        ):
            raise ValueError("Malformed physical control state")
        now = float(self.extract_state(state, mujoco.mjtState.mjSTATE_TIME)[0])
        start = c["control_start_s"]
        if (
            not np.isfinite([now, start, c["control_tracking_error"]]).all()
            or start < 0
            or c["control_tracking_error"] < 0
            or abs(start - completed * self.env.dt) > 1e-9
            or abs(now - start - index * self.model.opt.timestep) > 1e-9
        ):
            raise ValueError("Inconsistent physical control clock")
        qpos = self.extract_state(state, mujoco.mjtState.mjSTATE_QPOS)
        if pending:
            a = np.asarray(c["action"], dtype=np.float32)
            decision = np.asarray(c["decision_qpos"], dtype=float)
            if (
                a.shape != (700,)
                or decision.shape != (85,)
                or not np.isfinite(np.r_[a, decision]).all()
                or np.max(abs(a)) > 1
                or not np.array_equal(obs[:85], decision.astype(np.float32))
            ):
                raise ValueError("Malformed held decision/action")
            excitation = 1 / (1 + np.exp(-5 * (a - 0.5)))
            if not np.array_equal(
                self.extract_state(state, mujoco.mjtState.mjSTATE_CTRL), excitation
            ):
                raise ValueError("Held excitation disagrees with action")
        else:
            if c["action"] is not None or c["decision_qpos"] is not None:
                raise ValueError("Unexpected idle action")
            if not np.array_equal(obs[:85], qpos.astype(np.float32)):
                raise ValueError("Inconsistent boundary observation")
        for component in (
            mujoco.mjtState.mjSTATE_QFRC_APPLIED,
            mujoco.mjtState.mjSTATE_XFRC_APPLIED,
        ):
            if np.any(self.extract_state(state, component)):
                raise ValueError("Snapshot contains external assistance")
        for name in ("first_tracking_failure_s", "physical_failure_s"):
            value = c[name]
            if value is not None and (
                not np.isfinite(value) or not 0 <= value <= now + 1e-9
            ):
                raise ValueError("Malformed first failure time")
        reason = c["physical_failure"]
        if (
            reason not in (None, *REASONS)
            or (reason is None) != (c["physical_failure_s"] is None)
            or s["sampled_failure"] != reason
        ):
            raise ValueError("Inconsistent physical failure latch")
        expected_done = not pending and (
            reason is not None or now + 1e-9 >= self.physical.horizon_s
        )
        if type(s["done"]) is not bool or s["done"] != expected_done:
            raise ValueError("Inconsistent physical done state")
        monitor = SupportMonitor(**s["monitor"])
        if (
            not np.isfinite(list(monitor.snapshot().values())).all()
            or min(monitor.snapshot().values()) < 0
            or max(monitor.snapshot().values()) > now + 1e-9
            or (
                reason is None
                and any(
                    value + 1e-12 >= threshold
                    for value, threshold in zip(
                        monitor.snapshot().values(), (0.05, 0.1, 0.1), strict=True
                    )
                )
            )
        ):
            raise ValueError("Malformed support durations")
        rng = copy.deepcopy(self.env.np_random.bit_generator)
        rng.state = copy.deepcopy(s["rng"])  # Validate before any engine mutation.
        rows = [SubstepMotorEvidence.from_dict(r) for r in c["interval_evidence"]]
        expected_rows = index if pending else (self.env.frame_skip if completed else 0)
        if len(rows) != expected_rows:
            raise ValueError("Missing held interval evidence")
        for i, row in enumerate(rows):
            interval_start = start if pending else start - self.env.dt
            if (
                len(row.loads_N) != 3
                or not np.isfinite(
                    [row.start_s, row.end_s, *row.loads_N, row.tilt_rad, row.drop_m]
                ).all()
                or min(row.loads_N) < 0
                or abs(row.start_s - interval_start - i * self.model.opt.timestep)
                > 1e-9
                or abs(row.end_s - row.start_s - self.model.opt.timestep) > 1e-9
            ):
                raise ValueError("Malformed interval evidence")
            reconstructed = np.zeros(3)
            for contact in row.contacts:
                if len(contact) != 9 or not np.isfinite(contact).all():
                    raise ValueError("Malformed contact evidence")
                g1, g2, role = contact[:3]
                if any(int(v) != v for v in (g1, g2, role)) or int(role) not in range(
                    3
                ):
                    raise ValueError("Malformed contact identity")
                if not (0 <= g1 < self.model.ngeom and 0 <= g2 < self.model.ngeom) or (
                    g1 == self.floor
                ) == (g2 == self.floor):
                    raise ValueError("Unknown floor contact ownership")
                other = int(g2 if g1 == self.floor else g1)
                if ROLES[int(role)] != self.roles[other]:
                    raise ValueError("Contact role mismatch")
                reconstructed[int(role)] += max(0.0, contact[5])
            if not np.array_equal(reconstructed, row.loads_N):
                raise ValueError("Contact/load mismatch")
        # NativeTask validates all base shapes/references. A temporary MjData
        # isolates every base validation (including engine forward) from live data.
        transport = copy.deepcopy(s)
        transport["last_raw_observation"][:85] = qpos.astype(np.float32).tolist()
        original_data = self.env.data
        missing = object()
        original_app = {
            k: copy.deepcopy(getattr(self.env, k)) if hasattr(self.env, k) else missing
            for k in s["application"]
        }
        original_task = {
            k: getattr(self, k, missing)
            for k in (
                "last_obs",
                "initial_height",
                "monitor",
                "sampled_failure",
                "max_tilt",
                "done",
            )
        }
        original_rng = copy.deepcopy(self.env.np_random.bit_generator.state)
        try:
            self.env.data = mujoco.MjData(self.model)
            super().restore(transport)
            restored_data = self.env.data
        finally:
            self.env.data = original_data
            for k, v in original_app.items():
                if v is missing:
                    self.env.__dict__.pop(k, None)
                else:
                    setattr(self.env, k, v)
            for k, v in original_task.items():
                if v is missing:
                    self.__dict__.pop(k, None)
                else:
                    setattr(self, k, v)
            self.env.np_random.bit_generator.state = original_rng
        # No validation remains after this commit point.
        self.env.data = restored_data
        for k, v in s["application"].items():
            setattr(self.env, k, np.asarray(v) if k.endswith(("ref", "future")) else v)
        self.env.np_random.bit_generator.state = copy.deepcopy(s["rng"])
        self.last_obs, self.initial_height = obs.copy(), s["initial_height"]
        self.monitor, self.sampled_failure = monitor, reason
        self.max_tilt, self.done = s["max_tilt"], s["done"]
        self.pending, self.substep, self.completed_controls = pending, index, completed
        self.control_start_s = start
        self.action = a.copy() if pending else None
        self.excitation = excitation if pending else None
        self.decision_qpos = decision.copy() if pending else None
        self.first_tracking_failure_s = c["first_tracking_failure_s"]
        self.physical_failure, self.physical_failure_s = reason, c["physical_failure_s"]
        self.control_tracking_error = c["control_tracking_error"]
        self.interval_evidence, self.evidence = rows, []
        self.registered_mid_snapshot = None
        return obs.copy()

    def dump_substeps(self, path):
        rows = self.evidence
        contacts = [c for row in rows for c in row.contacts]
        packed = np.asarray(contacts, dtype=float).reshape(-1, 9)
        offsets = np.r_[0, np.cumsum([len(r.contacts) for r in rows])]
        np.savez_compressed(
            Path(path),
            intervals_s=np.asarray([[r.start_s, r.end_s] for r in rows]).reshape(-1, 2),
            loads_N=np.asarray([r.loads_N for r in rows]).reshape(-1, 3),
            tilt_drop=np.asarray([[r.tilt_rad, r.drop_m] for r in rows]).reshape(-1, 2),
            contact_offsets=offsets,
            contact_geom_ids=packed[:, :2].astype(np.int32),
            contact_role=packed[:, 2].astype(np.int8),
            contact_force_N=packed[:, 3:6],
            contact_position_m=packed[:, 6:9],
        )


def summarize(rows):
    p = protocol()
    expected = {r["attempt_id"] for r in planned_rows(p)}
    if len(rows) != 40 or {r["attempt_id"] for r in rows} != expected:
        raise ValueError("Physical diagnostic denominator changed")
    variants = {}
    for variant in p["variants"]:
        subset = [r for r in rows if r["variant"] == variant["id"]]
        completed = [r for r in subset if r["status"] == "completed"]
        variants[variant["id"]] = {
            "planned": len(subset),
            "completed": len(completed),
            **{
                status: sum(r["status"] == status for r in subset)
                for status in ("invalid", "incomplete", "not_run")
            },
            "physical_horizon_completions": sum(
                r["horizon_reached"] for r in completed
            ),
            "mean_elapsed_sim_s": float(
                np.mean([r["elapsed_sim_s"] for r in completed])
            )
            if completed
            else None,
        }
    all_complete = all(v["completed"] == 20 for v in variants.values())
    return {
        "attempts": 40,
        "variants": variants,
        "invalid": sum(r["status"] == "invalid" for r in rows),
        "all_attempts_completed": all_complete,
        "physical_completion_threshold_met": all_complete
        and all(v["physical_horizon_completions"] >= 16 for v in variants.values()),
        "new_training": False,
        "boxing_or_stance_qualified": False,
    }


def equivalence_oracle(config, variants, seed=75091, controls=6):
    """Short, pre-outcome unbatched/cache oracle against the native v7 path."""
    import traceback

    result = []
    physics = 0
    for variant in variants:
        old = new = None
        try:
            old = TransferTask(config, variant)
            new = PhysicalMotorTask(config, variant)
            old.reset(seed)
            new.reset(seed)
            if digest(old.paired_fields()) != digest(new.paired_fields()):
                raise RuntimeError("Physical reset pairing failed")
            state_error = feature_error = reward_error = 0.0
            steps = 0
            for _ in range(controls):
                command = np.tanh(np.resize(old.last_obs, 700) * 0.001)
                a, b = old.step(command), new.step(command)
                steps += 1
                state_error = max(
                    state_error,
                    float(
                        np.max(
                            abs(
                                np.asarray(old.snapshot()["state"])
                                - new.snapshot()["state"]
                            )
                        )
                    ),
                )
                feature_error = max(feature_error, float(np.max(abs(a[0] - b[0]))))
                reward_error = max(reward_error, abs(a[1] - b[1]))
                if state_error or feature_error or reward_error:
                    raise RuntimeError("Substep evidence changed native transition")
                if old.done:
                    if new.first_tracking_failure_s != float(old.data.time):
                        raise RuntimeError(
                            "Original tracking failure was not preserved"
                        )
                    break
            result.append(
                {
                    "variant": variant["id"],
                    "controls": steps,
                    "max_state_error": state_error,
                    "max_feature_error": feature_error,
                    "max_reward_error": reward_error,
                    "old_task_done": old.done,
                    "physical_task_done": new.done,
                    "passed": True,
                }
            )
        except Exception as error:  # noqa: BLE001 -- retain attempted oracle work
            result.append(
                {
                    "variant": variant["id"],
                    "passed": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            break
        finally:
            for task in (old, new):
                if task is not None:
                    physics += task.physics_total
                    task.close()
    return {
        "passed": len(result) == len(variants) and all(r["passed"] for r in result),
        "variants": result,
        "physics_steps": physics,
    }


def source_host_oracle(p, inputs):
    """Reproduce both actual v7 first scenarios with unchanged policy commands."""
    import traceback

    from .native_transfer import load_controller, predict
    from .stance_train import file_hash

    old_run = Path(p["prerequisite"]["v7_run"])
    audit_path = old_run.parent / "audit-linux.json"
    findings, tasks = [], []
    try:
        audit = json.loads(audit_path.read_text())
        if not audit["passed"] or not audit["reference_replay_passed"]:
            raise RuntimeError("Actual v7 Linux audit missing")
        model, normalizer = load_controller(Path(inputs))
        old_protocol = json.loads((old_run / "protocol.json").read_text())
        if old_protocol["environment"] != p["environment"]:
            raise RuntimeError("v7 native configuration changed")
        seed = old_protocol["replay"]["seed"]
        for variant in p["variants"]:
            folder = old_run / variant["id"]
            rec = json.loads((folder / "replay.json").read_text())
            old = TransferTask(p["environment"], variant)
            tasks.append(old)
            new = PhysicalMotorTask(p["environment"], variant)
            tasks.append(new)
            if old.contract != json.loads((folder / "specification.json").read_text()):
                raise RuntimeError("Actual v7 compiled contract changed")
            old.reset(seed)
            new.reset(seed)
            start = json.loads((folder / f"starts/{seed}.json").read_text())
            if (
                old.snapshot()["state"] != start["state"]
                or new.snapshot()["state"] != start["state"]
            ):
                raise RuntimeError("Actual v7 initial state changed")
            with np.load(folder / rec["actions_file"]) as z:
                actions = z["actions"]
            for index, command in enumerate(actions):
                predicted, _ = predict(model, normalizer, new.last_obs)
                if not np.array_equal(command, predicted):
                    raise RuntimeError("Substep evidence changed actual policy command")
                a, b = old.step(command), new.step(command)
                if (
                    a[1] != b[1]
                    or not np.array_equal(a[0], b[0])
                    or old.snapshot()["state"] != new.snapshot()["state"]
                ):
                    raise RuntimeError("Actual v7 state/feature/reward changed")
                if (
                    index + 1 == rec["continuation_start"]
                    and old.snapshot()["state"] != rec["snapshot"]["state"]
                ):
                    raise RuntimeError("Actual v7 boundary replay changed")
                if old.done and index + 1 != len(actions):
                    raise RuntimeError("Actual v7 stopped earlier than recorded")
            if (
                not old.done
                or old.snapshot()["state"] != rec["final_snapshot"]["state"]
                or not np.array_equal(
                    new.last_obs, rec["final_snapshot"]["last_raw_observation"]
                )
            ):
                raise RuntimeError("Actual v7 final reference changed")
            if new.first_tracking_failure_s != float(old.data.time):
                raise RuntimeError("Actual v7 tracking onset changed")
            findings.append(
                {
                    "variant": variant["id"],
                    "seed": seed,
                    "controls": len(actions),
                    "max_state_feature_reward_action_error": 0.0,
                    "original_tracking_failure_s": new.first_tracking_failure_s,
                    "physical_failure": new.physical_failure,
                    "source_replay_sha256": file_hash(folder / "replay.json"),
                    "source_actions_sha256": file_hash(folder / rec["actions_file"]),
                    "passed": True,
                }
            )
        normalizer.check()
        return {
            "passed": True,
            "findings": findings,
            "physics_steps": sum(t.physics_total for t in tasks),
            "v7_audit_sha256": file_hash(audit_path),
        }
    except Exception as error:  # noqa: BLE001 -- failed prerequisites are evidence
        return {
            "passed": False,
            "findings": findings,
            "physics_steps": sum(t.physics_total for t in tasks),
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        for task in tasks:
            task.close()


def run_physical(protocol_path, archive, out):
    from .native_transfer import run_motor_experiment

    p = validate_protocol(json.loads(Path(protocol_path).read_text()))
    return run_motor_experiment(p, protocol_path, archive, out)
