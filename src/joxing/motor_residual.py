"""Frozen native teacher plus an explicit deterministic138-action motor residual.

The new SAC objective is over residual actions, not a density over700 muscles.
No competitive learning or default physical task behavior is enabled here.
"""

import copy
import hashlib
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np

from .native_transfer import load_controller, predict
from .physical_motor import PhysicalMotorTask
from .provenance import digest
from .reference_seam import VARIANT, UnwrappedSeamTrajectory
from .wave import canonical_actor_hash

AMPLITUDE = 0.1
REWARD_ID = "physical_alive_seconds_v1"


def reference_hash(manager):
    return digest(
        [
            {
                key: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
                if isinstance(value, np.ndarray)
                else float(value)
                for key, value in trajectory.items()
            }
            for trajectory in manager.trajectories
        ]
    )


def validate_groups(groups):
    result = tuple(tuple(group) for group in groups)
    if len(result) != 138 or any(not group for group in result):
        raise ValueError("Expected138 nonempty native groups")
    indices = [i for group in result for i in group]
    if any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
        for i in indices
    ) or sorted(indices) != list(range(700)):
        raise ValueError("Groups must partition all700 muscles exactly once")
    return tuple(tuple(int(i) for i in group) for group in result)


def action(value, size):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (size,) or not np.isfinite(a).all() or np.any(abs(a) > 1):
        raise ValueError(f"Expected finite bounded action{size}")
    return a.copy()


def expand_groups(residual, groups):
    residual = action(residual, 138)
    groups = validate_groups(groups)
    result = np.empty(700, dtype=np.float32)
    for i, group in enumerate(groups):
        result[list(group)] = residual[i]
    return result


def decode_command(teacher, residual, groups, amplitude=AMPLITUDE):
    if amplitude != AMPLITUDE:
        raise ValueError("Unregistered residual amplitude")
    teacher = action(teacher, 700)
    correction = expand_groups(residual, groups)
    return np.clip(teacher + np.float32(amplitude) * correction, -1, 1).astype(
        np.float32, copy=True
    )


def physical_reward(start, end, failure):
    values = [start, end] + ([] if failure is None else [failure])
    if (
        not np.isfinite(values).all()
        or end < start
        or (failure is not None and not start <= failure <= end)
    ):
        raise ValueError("Invalid motor reward interval/failure")
    alive = max(0.0, min(end, end if failure is None else failure) - start)
    return alive - float(failure is not None), alive


def episode_masks(terminated, truncated):
    return float(not terminated), float(not (terminated or truncated))


class MotorResidualEnv(gym.Env):
    """Own one task and immutable teacher; a pending command never re-predicts."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, task, teacher, normalizer, groups):
        if task.last_obs is not None or getattr(task, "pending", False):
            raise ValueError("Install residual/reference variant before first reset")
        if (task.model.nq, task.model.nv, task.model.nu, task.model.na) != (
            85,
            85,
            700,
            700,
        ) or task.model.opt.timestep != 0.002:
            raise ValueError("Residual task requires all700 native2ms model")
        self.task, self.teacher, self.normalizer = task, teacher, normalizer
        self.groups = validate_groups(groups)
        if not isinstance(task.env.trajectory, UnwrappedSeamTrajectory):
            task.env.trajectory = UnwrappedSeamTrajectory(task.env.trajectory)
        teacher.actor.eval()
        teacher.actor.requires_grad_(False)
        self.teacher_hash = canonical_actor_hash(teacher.actor)
        self.normalizer_hash = normalizer.fingerprint
        self.group_hash = digest(self.groups)
        self.reference_hash = reference_hash(task.env.trajectory)
        self._held = None
        self.last_transition = None
        self.episode_seed = None
        self._np_random = np.random.default_rng(0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3601,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (138,), np.float32)
        self.contract = {
            "schema": "joxing-deterministic-motor-residual-v1",
            "physical_contract": task.contract_hash,
            "reference_variant": VARIANT,
            "reference_content_hash": self.reference_hash,
            "teacher_actor_hash": self.teacher_hash,
            "decoder_amplitude": 0.075,
            "normalizer_hash": self.normalizer_hash,
            "groups_hash": self.group_hash,
            "muscle_names_by_group": [
                [task.model.actuator(i).name for i in group] for group in self.groups
            ],
            "residual_amplitude": AMPLITUDE,
            "reward": REWARD_ID,
            "observation": "frozen-normalized native3601 cached features",
            "action": "latent138 f32; teacher700 plus group correction then legacy excitation once",
        }
        self.contract_hash = digest(self.contract)
        self.check_bundle()

    @classmethod
    def from_inputs(cls, inputs):
        from .physical_motor import protocol

        teacher, normalizer = load_controller(Path(inputs))
        p = protocol()
        task = PhysicalMotorTask(p["environment"], p["variants"][0])
        try:
            return cls(
                task, teacher, normalizer, teacher.actor.dynsyn_layer.muscle_groups
            )
        except Exception:
            task.close()
            raise

    def check_bundle(self):
        self.normalizer.check()
        if (
            self.normalizer.fingerprint != self.normalizer_hash
            or canonical_actor_hash(self.teacher.actor) != self.teacher_hash
            or digest(self.groups) != self.group_hash
        ):
            raise RuntimeError("Frozen teacher/normalizer/groups changed")
        if self.teacher.actor.dynsyn_layer.dynsyn_weight_amp != 0.075:
            raise RuntimeError("Frozen teacher decoder amplitude changed")
        if not isinstance(self.task.env.trajectory, UnwrappedSeamTrajectory):
            raise TypeError("Reference variant changed")
        if (
            reference_hash(self.task.env.trajectory) != self.reference_hash
            or digest(self.contract) != self.contract_hash
        ):
            raise RuntimeError("Reference content or behavior contract changed")

    def observation(self):
        return self.normalizer.apply(self.task.last_obs).copy()

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Unregistered motor reset options")
        super().reset(seed=seed)
        self.check_bundle()
        chosen = int(self.np_random.integers(0, 2**32)) if seed is None else int(seed)
        self.task.reset(chosen)
        self.episode_seed = chosen
        self._held = self.last_transition = None
        return self.observation(), {
            "episode_seed": chosen,
            "bundle_hash": self.contract_hash,
        }

    def begin_control(self, residual):
        import torch

        residual = action(residual, 138)
        if self._held is not None or self.task.pending:
            raise RuntimeError("Residual command already pending")
        with torch.inference_mode():
            teacher, clipped = predict(
                self.teacher, self.normalizer, self.task.last_obs
            )
        self._begin_command(residual, teacher, clipped)

    def _begin_command(self, residual, teacher, clipped):
        command = decode_command(teacher, residual, self.groups)
        self.task.begin_control(command)
        self._held = {
            "residual": residual.tolist(),
            "teacher": teacher.tolist(),
            "command": command.tolist(),
            "start_s": float(self.task.control_start_s),
            "teacher_clipped_fraction": clipped,
        }

    def begin_recorded_neutral(self, command):
        """Qualification-only fixed muscle tape; no actor or fresh residual."""
        if self._held is not None or self.task.pending:
            raise RuntimeError("Residual command already pending")
        normalized = self.observation()
        clipped = float(np.mean(abs(normalized) >= self.normalizer.normalizer.clip_obs))
        self._begin_command(np.zeros(138, np.float32), action(command, 700), clipped)

    def advance_substeps(self, count):
        if self._held is None:
            raise RuntimeError("No residual command to advance")
        self.task.advance_substeps(count)

    def resume_control(self):
        if self._held is None or not self.task.pending:
            raise RuntimeError("No pending residual control")
        held = self._held
        _, native_reward, terminated, truncated, _, telemetry = (
            self.task.resume_control()
        )
        reward, alive = physical_reward(
            held["start_s"], telemetry["time_s"], telemetry["physical_failure_s"]
        )
        obs = self.observation()
        bootstrap, trace = episode_masks(terminated, truncated)
        info = {
            "native_reward": native_reward,
            "alive_seconds": alive,
            "failure_penalty": float(telemetry["physical_failure_s"] is not None),
            "time_s": telemetry["time_s"],
            "physical_failure": telemetry["physical_failure"],
            "physical_failure_s": telemetry["physical_failure_s"],
            "first_tracking_failure_s": telemetry["first_native_tracking_failure_s"],
            "bootstrap_mask": bootstrap,
            "trace_mask": trace,
            "teacher_clipped_fraction": held["teacher_clipped_fraction"],
        }
        if terminated or truncated:
            info["terminal_observation"] = obs.copy()
            self.check_bundle()
        self.last_transition = copy.deepcopy(held) | {
            "telemetry": telemetry,
            "native_reward": native_reward,
        }
        self._held = None
        return obs, reward, terminated, truncated, info

    def step(self, residual):
        self.begin_control(residual)
        return self.resume_control()

    def snapshot(self):
        self.check_bundle()
        return {
            "schema": "joxing-motor-residual-snapshot-v1",
            "bundle_hash": self.contract_hash,
            "physical": self.task.snapshot(),
            "reference_dtypes": {
                key: getattr(self.task.env, key).dtype.str
                for key in ("qpos_ref", "xpos_ref", "qpos_ref_future")
            },
            "held": copy.deepcopy(self._held),
            "episode_seed": self.episode_seed,
            "reset_rng": copy.deepcopy(self.np_random.bit_generator.state),
        }

    def restore(self, snapshot):
        self.check_bundle()
        s = copy.deepcopy(snapshot)
        if (
            set(s)
            != {
                "schema",
                "bundle_hash",
                "physical",
                "reference_dtypes",
                "held",
                "episode_seed",
                "reset_rng",
            }
            or s["schema"] != "joxing-motor-residual-snapshot-v1"
            or s["bundle_hash"] != self.contract_hash
        ):
            raise ValueError("Residual snapshot/bundle mismatch")
        rng = np.random.default_rng()
        rng.bit_generator.state = s["reset_rng"]
        held, physical = s["held"], s["physical"]
        names = ("qpos_ref", "xpos_ref", "qpos_ref_future")
        if set(s["reference_dtypes"]) != set(names):
            raise ValueError("Incomplete reference dtype contract")
        references = {}
        for key in names:
            dtype = np.dtype(s["reference_dtypes"][key])
            if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise ValueError("Unsupported reference dtype")
            value = np.asarray(physical["application"][key], dtype=dtype)
            if not np.array_equal(
                value.astype(np.float64), physical["application"][key]
            ):
                raise ValueError("Reference values do not fit declared dtype")
            references[key] = value.copy()
        if (held is not None) != physical["physical_control"]["pending"]:
            raise ValueError("Residual/physical pending mismatch")
        if held is not None:
            if set(held) != {
                "residual",
                "teacher",
                "command",
                "start_s",
                "teacher_clipped_fraction",
            }:
                raise ValueError("Unknown held residual fields")
            decoded = decode_command(held["teacher"], held["residual"], self.groups)
            if (
                not np.array_equal(decoded, action(held["command"], 700))
                or not np.array_equal(decoded, physical["physical_control"]["action"])
                or held["start_s"] != physical["physical_control"]["control_start_s"]
                or not 0 <= held["teacher_clipped_fraction"] <= 1
            ):
                raise ValueError("Held residual ownership/command mismatch")
        self.normalizer.apply(
            np.asarray(physical["last_raw_observation"], dtype=np.float32)
        )
        self.task.restore(physical)  # Base restore validates atomically before commit.
        for key, value in references.items():
            setattr(self.task.env, key, value)
        self._held, self.episode_seed, self._np_random = held, s["episode_seed"], rng
        self.last_transition = None

    def import_neutral_source(self, physical, seed):
        """Explicit qualification bridge; no cross-bundle production compatibility."""
        source = copy.deepcopy(physical)
        state = source["physical_control"]
        held = None
        if state["pending"]:
            normalized = self.normalizer.apply(
                np.asarray(source["last_raw_observation"], dtype=np.float32)
            )
            held = {
                "residual": [0.0] * 138,
                "teacher": state["action"],
                "command": state["action"],
                "start_s": state["control_start_s"],
                "teacher_clipped_fraction": float(
                    np.mean(abs(normalized) >= self.normalizer.normalizer.clip_obs)
                ),
            }
        rng = np.random.default_rng(seed)
        self.restore(
            {
                "schema": "joxing-motor-residual-snapshot-v1",
                "bundle_hash": self.contract_hash,
                "physical": source,
                "reference_dtypes": {
                    "qpos_ref": np.dtype(np.float32).str,
                    "xpos_ref": np.dtype(np.float32).str,
                    "qpos_ref_future": np.dtype(np.float64).str,
                },
                "held": held,
                "episode_seed": seed,
                "reset_rng": rng.bit_generator.state,
            }
        )

    def close(self):
        self.task.close()
