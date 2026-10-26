"""Feedback muscle control under gravity. This motor diagnostic is not a match."""

import copy
from dataclasses import asdict
from typing import ClassVar

import gymnasium as gym
import mujoco
import numpy as np

from .balance import BalanceConfig, BalanceModel
from .groups import MuscleGroups
from .physics import step_contacts
from .provenance import digest
from .support import SupportMonitor, floor_loads


class StanceEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, config=None, training=True):
        super().__init__()
        config = config or BalanceConfig()
        self.rig = BalanceModel(config)
        self.model = self.rig.model
        self.config = config
        self.groups = MuscleGroups(self.model)
        self.training = training
        self.action_space = gym.spaces.Box(-1.0, 1.0, (138,), dtype=np.float32)
        self.hand_ids = [self.model.body("hand_" + s).id for s in ("r", "l")]
        self.trunk = self.model.body("thoracic12").id
        self.cache = {}
        self.physics_total = 0
        self.preparation_total = 0
        self.reset(seed=0, options={"scenario_id": 30000})
        obs = self.observe()
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, obs.shape, dtype=np.float32
        )
        self.observation_hash = digest(
            {
                "blocks": self.obs_slices,
                "mapping": "relative-qpos,tanh(qvel/10),native-activation,world-up,relative-pelvis,relative-hand-error,weight-normalized-support,previous-normalized-action-v1",
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"scenario_id", "push_Ns", "push_direction_rad"}:
            raise ValueError("Unknown reset option")
        scenario = int(
            options.get("scenario_id", 100000 + int(self.np_random.integers(64)))
        )
        self.push_Ns = float(options.get("push_Ns", 0.0))
        self.push_direction = float(options.get("push_direction_rad", 0.0))
        if (
            not np.isfinite([self.push_Ns, self.push_direction]).all()
            or not 0 <= self.push_Ns <= 4
        ):
            raise ValueError("Invalid prescribed push")
        if scenario not in self.cache:
            data, audit = self.rig.prepare(scenario)
            state = np.empty(
                mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
            )
            mujoco.mj_getState(
                self.model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION
            )
            self.cache[scenario] = (state, audit)
            self.preparation_total += round(
                self.config.preparation_s / self.config.dt_s
            )
        state, self.reset_audit = self.cache[scenario]
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setState(
            self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(self.model, self.data)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_pelvis = self.data.xpos[self.rig.pelvis].copy()
        self.initial_hands = self.data.xpos[self.hand_ids].copy()
        self.previous_action = np.full(138, 0.5 + np.log(0.02 / 0.98) / 5)
        self.monitor = SupportMonitor()
        self.tick = 0
        self.done = False
        self.failure = None
        self.loads = {"right": 0.0, "left": 0.0, "nonfoot": 0.0}
        self.push_delivered = np.zeros(3)
        self.stable_s = 0.0
        self.max_tilt = 0.0
        self.last_outside_s = None
        self.scenario_id = scenario
        self.trace = []
        return self.observe(), {"reset_audit": copy.deepcopy(self.reset_audit)}

    def kinematics(self):
        up = self.data.xmat[self.rig.pelvis].reshape(3, 3)[:, 1]
        tilt = float(np.arccos(np.clip(up[2], -1, 1)))
        drift = float(
            np.linalg.norm(
                self.data.xpos[self.rig.pelvis, :2] - self.initial_pelvis[:2]
            )
        )
        drop = float(self.initial_pelvis[2] - self.data.xpos[self.rig.pelvis, 2])
        return up, tilt, drift, drop

    def observe(self):
        up, _, _, _ = self.kinematics()
        blocks = {
            "relative_qpos": self.data.qpos - self.initial_qpos,
            "velocity": np.tanh(self.data.qvel / 10),
            "activation": self.data.act,
            "pelvis_up": up,
            "relative_pelvis": self.data.xpos[self.rig.pelvis] - self.initial_pelvis,
            "hand_error": (self.data.xpos[self.hand_ids] - self.initial_hands).ravel(),
            "support": np.array([self.loads[k] for k in ("right", "left", "nonfoot")])
            / self.rig.weight_N,
            "previous_action": self.previous_action,
        }
        self.obs_slices = {}
        offset = 0
        for key, value in blocks.items():
            self.obs_slices[key] = [offset, offset + len(value)]
            offset += len(value)
        obs = np.concatenate(list(blocks.values())).astype(np.float32)
        if not np.isfinite(obs).all():
            raise RuntimeError("Invalid nonfinite observation; quarantine episode")
        return obs

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset required after motor episode ends")
        if not np.array_equal(self.model.opt.gravity, [0.0, 0.0, -9.81]) or np.any(
            self.model.body_gravcomp
        ):
            raise RuntimeError("Unexpected gravity assistance")
        ctrl = self.groups.expand(action)
        self.data.ctrl[:] = ctrl
        smoothness = float(np.mean((np.asarray(action) - self.previous_action) ** 2))
        self.previous_action = np.asarray(action, dtype=float).copy()
        reward = 0.0
        for _ in range(round(self.config.control_s / self.config.dt_s)):
            if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
                raise RuntimeError("Unexpected external assistance")
            start = round(5 / self.config.dt_s)
            end = start + round(0.05 / self.config.dt_s)
            force = np.zeros(3)
            if start <= self.tick < end:
                force = (
                    self.push_Ns
                    / 0.05
                    * np.array(
                        [np.cos(self.push_direction), np.sin(self.push_direction), 0]
                    )
                )
                self.data.xfrc_applied[self.trunk, :3] = force
            contacts = step_contacts(self.model, self.data)
            self.data.xfrc_applied[:] = 0
            self.push_delivered += force * self.config.dt_s
            self.tick += 1
            self.physics_total += 1
            self.loads = floor_loads(contacts, self.rig.floor, self.rig.roles)
            mujoco.mj_kinematics(self.model, self.data)
            _, tilt, drift, drop = self.kinematics()
            self.max_tilt = max(self.max_tilt, tilt)
            self.failure = self.monitor.update(
                self.loads, self.rig.weight_N, tilt, drop, self.config.dt_s
            )
            stable = tilt < np.deg2rad(20) and drift < 0.15
            if stable:
                self.stable_s += self.config.dt_s
            else:
                self.last_outside_s = self.tick * self.config.dt_s
            hand_error = float(
                np.mean(
                    np.linalg.norm(
                        self.data.xpos[self.hand_ids] - self.initial_hands, axis=1
                    )
                )
            )
            rate = (
                np.exp(-4 * tilt**2)
                + np.exp(-20 * drift**2)
                + 0.5 * np.exp(-20 * hand_error**2)
                - 0.01 * float(np.mean(ctrl**2))
                - 0.01 * smoothness
            )
            reward += float(rate * self.config.dt_s)
            if self.failure or self.tick >= round(
                self.config.episode_s / self.config.dt_s
            ):
                break
        terminated = self.failure is not None
        truncated = not terminated and self.tick >= round(
            self.config.episode_s / self.config.dt_s
        )
        self.done = terminated or truncated
        info = self.result()
        self.trace.append(
            [
                self.data.time,
                tilt,
                drift,
                *(self.loads[k] for k in ("right", "left", "nonfoot")),
                float(np.linalg.norm(self.push_delivered)),
            ]
        )
        return self.observe(), reward, terminated, truncated, info

    def result(self):
        time_s = self.tick * self.config.dt_s
        last_outside = self.last_outside_s if self.last_outside_s is not None else 0.0
        quiet_success = (
            self.done
            and self.failure is None
            and last_outside <= self.config.episode_s - 2
        )
        recovery_success = (
            quiet_success
            and last_outside <= 7.05
            and abs(np.linalg.norm(self.push_delivered) - self.push_Ns) < 1e-6
        )
        reason = self.failure or (
            "target_tolerance" if self.done and not quiet_success else None
        )
        return {
            "scenario_id": self.scenario_id,
            "status": "physical_failure"
            if reason
            else "success"
            if self.done
            else "running",
            "reason": reason,
            "elapsed_sim_s": time_s,
            "physics_steps": self.tick,
            "stable_s": self.stable_s,
            "max_tilt_rad": self.max_tilt,
            "quiet_success": bool(quiet_success),
            "recovery_success": bool(recovery_success),
            "push_delivered_Ns": self.push_delivered.tolist(),
            "push_prescribed_Ns": self.push_Ns,
        }

    def snapshot(self):
        state = np.empty(
            mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        )
        mujoco.mj_getState(
            self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        keys = (
            "tick",
            "done",
            "failure",
            "loads",
            "push_Ns",
            "push_direction",
            "push_delivered",
            "stable_s",
            "max_tilt",
            "last_outside_s",
            "scenario_id",
            "trace",
            "initial_qpos",
            "initial_pelvis",
            "initial_hands",
            "previous_action",
            "reset_audit",
        )
        application = {k: copy.deepcopy(getattr(self, k)) for k in keys}
        for k, v in application.items():
            if isinstance(v, np.ndarray):
                application[k] = v.tolist()
        return {
            "model_hash": self.rig.hash,
            "action_hash": self.groups.hash,
            "observation_hash": self.observation_hash,
            "state": state.tolist(),
            "application": application,
            "monitor": self.monitor.snapshot(),
            "rng": copy.deepcopy(self.np_random.bit_generator.state),
        }

    def restore(self, snapshot):
        if any(
            snapshot[k] != value
            for k, value in (
                ("model_hash", self.rig.hash),
                ("action_hash", self.groups.hash),
                ("observation_hash", self.observation_hash),
            )
        ):
            raise ValueError("Incompatible balance snapshot")
        array_keys = {
            "push_delivered",
            "initial_qpos",
            "initial_pelvis",
            "initial_hands",
            "previous_action",
        }
        state = np.asarray(snapshot["state"], dtype=float)
        expected = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        if state.shape != (expected,) or not np.isfinite(state).all():
            raise ValueError("Invalid snapshot engine state")
        application = snapshot["application"]
        expected_keys = {
            "tick",
            "done",
            "failure",
            "loads",
            "push_Ns",
            "push_direction",
            "push_delivered",
            "stable_s",
            "max_tilt",
            "last_outside_s",
            "scenario_id",
            "trace",
            "initial_qpos",
            "initial_pelvis",
            "initial_hands",
            "previous_action",
            "reset_audit",
        }
        if set(application) != expected_keys:
            raise ValueError("Unknown or missing snapshot application field")
        shapes = {
            "push_delivered": (3,),
            "initial_qpos": (self.model.nq,),
            "initial_pelvis": (3,),
            "initial_hands": (2, 3),
            "previous_action": (138,),
        }
        for key, shape in shapes.items():
            value = np.asarray(application[key], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid snapshot {key}")
        tick = application["tick"]
        if (
            type(tick) is not int
            or tick < 0
            or abs(state[0] - tick * self.config.dt_s) > 1e-9
        ):
            raise ValueError("Invalid snapshot clock")
        if (
            set(application["loads"]) != {"right", "left", "nonfoot"}
            or not np.isfinite(list(application["loads"].values())).all()
        ):
            raise ValueError("Invalid snapshot support channels")
        monitor = SupportMonitor(**snapshot["monitor"])
        if (
            not np.isfinite(list(monitor.snapshot().values())).all()
            or min(monitor.snapshot().values()) < 0
        ):
            raise ValueError("Invalid snapshot persistence")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        candidate = mujoco.MjData(self.model)
        mujoco.mj_setState(
            self.model,
            candidate,
            state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        mujoco.mj_forward(self.model, candidate)
        for k, v in application.items():
            setattr(self, k, np.array(v) if k in array_keys else copy.deepcopy(v))
        self.monitor = monitor
        self.np_random.bit_generator.state = rng.bit_generator.state
        self.data = candidate

    def specification(self):
        return {
            "balance_config": asdict(self.config),
            "model_hash": self.rig.hash,
            "action_hash": self.groups.hash,
            "observation_hash": self.observation_hash,
            "observation_slices": self.obs_slices,
        }
