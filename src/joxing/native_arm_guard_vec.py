"""H35 arm-policy Vec adapter: two46action slots, one shared native world."""

import copy
from dataclasses import asdict

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .motor_residual import action
from .provenance import digest
from .shared_motor_task import PLAYERS


class GuardPairVec(VecEnv):
    def __init__(self, world, *, seed=71000, record_history=False):
        self.world = world
        # Opt-in evidence only. Never read history after a terminal auto-reset.
        self.record_history = bool(record_history)
        self.pending = None
        self.next_seed = int(seed)
        self.waiting = False
        self.render_mode = None
        super().__init__(
            2,
            gym.spaces.Box(-np.inf, np.inf, (world.actor_dimension,), np.float32),
            gym.spaces.Box(-1, 1, (46,), np.float32),
        )
        self.episode_returns = np.zeros(2, float)
        self.episode_lengths = 0

    def seed(self, seed=None):
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        if type(seed) is not int or seed < 0:
            raise ValueError("One explicit joint world seed required")
        self._seeds = [seed, seed]
        return list(self._seeds)

    def reset(self):
        if self.pending is not None:
            raise RuntimeError("No reset over pending joint action")
        if any(self._options):
            raise ValueError("Unregistered paired reset options")
        if self._seeds[0] is not None:
            if self._seeds[0] != self._seeds[1]:
                raise ValueError("Logical slots must share one physical reset seed")
            self.next_seed = int(self._seeds[0])
        obs, info = self.world.reset(seed=self.next_seed)
        self.next_seed += 1
        self.episode_returns = np.zeros(2, float)
        self.episode_lengths = 0
        self.reset_infos = [copy.deepcopy(info), copy.deepcopy(info)]
        self._reset_seeds()
        self._reset_options()
        return np.stack([obs[p] for p in PLAYERS])

    def step_async(self, actions):
        if self.pending is not None:
            raise RuntimeError("Already waiting for paired native actions")
        a = np.asarray(actions)
        if a.shape != (2, 46):
            raise ValueError("Both46actions required atomically")
        staged = {p: action(a[i], 46) for i, p in enumerate(PLAYERS)}
        self.pending = staged
        self.waiting = True

    def step_wait(self):
        if self.pending is None:
            raise RuntimeError("No pending native pair")
        actions = self.pending
        self.pending = None
        self.waiting = False
        before_obs = self.world.observations()
        before_history = (
            {p: self.world.histories[p].snapshot() for p in PLAYERS}
            if self.record_history
            else None
        )
        teacher_inputs = (
            {p: v.values.copy() for p, v in self.world.core.inputs.items()}
            if hasattr(self.world, "core")
            else None
        )
        before_state = (
            self.world.core.state().copy() if hasattr(self.world, "core") else None
        )
        self.world.begin_control(actions)
        metrics, contact_hashes = [], []
        while self.world.core.remaining:
            contacts = self.world.advance_substep()
            metrics.append(
                [
                    [
                        *m["distances_m"],
                        m["face_cos"],
                        m["up_world_z"],
                        m["tilt_rad"],
                        m["head_z_m"],
                        m["head_drop_m"],
                    ]
                    for m in (self.world.last_metrics[p] for p in PLAYERS)
                ]
            )
            contact_hashes.append(
                list(
                    bytes.fromhex(
                        digest(
                            [
                                {
                                    k: v.tolist() if isinstance(v, np.ndarray) else v
                                    for k, v in asdict(c).items()
                                }
                                for c in contacts
                            ]
                        )
                    )
                )
            )
        obs = self.world.observations()
        reward = self.world.last_transition["rewards"]
        terminated = self.world.last_transition["terminated"]
        truncated = self.world.last_transition["truncated"]
        info = self.world.last_transition["info"]
        self.last_transition = {
            "source_observations": before_obs,
            "episode_serial": self.world.episode_serial,
            "logical_seed": self.world.episode_seed,
            "nonce": self.world.nonce,
            "held": {p: self.world.held[p].snapshot() for p in PLAYERS},
            "metrics_1ms": np.asarray(metrics, np.float64),
            "contact_sha256_1ms": np.asarray(contact_hashes, np.uint8),
            "teacher_inputs": teacher_inputs,
            "teacher_commands": {
                p: np.frombuffer(self.world.held[p].teacher_bytes, np.float32).copy()
                for p in PLAYERS
            }
            if self.world.held is not None
            else None,
            "actions": {p: a.copy() for p, a in actions.items()},
            "next_observations": {p: a.copy() for p, a in obs.items()},
            "rewards": dict(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": copy.deepcopy(info),
            "engine_before": before_state,
            "engine_after": self.world.core.state().copy()
            if hasattr(self.world, "core")
            else None,
            "commands": {p: a.copy() for p, a in self.world.core.commands.items()}
            if hasattr(self.world, "core")
            else None,
        }
        if self.record_history:
            self.last_transition.update(
                source_histories=before_history,
                next_histories={p: self.world.histories[p].snapshot() for p in PLAYERS},
            )

        rewards = np.array([reward[p] for p in PLAYERS], np.float32)
        done = bool(terminated or truncated)
        self.episode_returns += rewards
        self.episode_lengths += 1
        infos = [
            {
                **copy.deepcopy(info),
                "player": p,
                "TimeLimit.truncated": bool(truncated and not terminated),
                "joint_world_episode_id": info["episode_id"],
                "independent_games": False,
            }
            for p in PLAYERS
        ]
        if done:
            final = {p: obs[p].copy() for p in PLAYERS}
            for i, p in enumerate(PLAYERS):
                infos[i].update(
                    terminal_observation=final[p],
                    episode={
                        "r": float(self.episode_returns[i]),
                        "l": self.episode_lengths,
                    },
                )
            obs, reset_info = self.world.reset(seed=self.next_seed)
            self.next_seed += 1
            self.reset_infos = [copy.deepcopy(reset_info), copy.deepcopy(reset_info)]
            self.episode_returns[:] = 0
            self.episode_lengths = 0
        return (
            np.stack([obs[p] for p in PLAYERS]),
            rewards,
            np.array([done, done]),
            infos,
        )

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            value = None
        else:
            value = getattr(self.world, attr_name)
        return [copy.deepcopy(value) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise RuntimeError("Cannot mutate one logical slot of a coupled native world")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name not in ("info", "snapshot"):
            raise RuntimeError("Only read methods may address logical slots")
        value = getattr(self.world, method_name)(*method_args, **method_kwargs)
        return [copy.deepcopy(value) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        self.pending = None
        self.waiting = False
        self.world.close()
