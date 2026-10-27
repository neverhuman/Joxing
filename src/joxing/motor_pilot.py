"""D2 worker ownership, reset streams and latent SAC construction.

This module alone does not authorize or launch physical learning.
"""

import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from .motor_storage import RawCredit, append_json, save_npz, write_json
from .provenance import digest

TRAINING_SEEDS = (81011, 81029)
EVALUATION_SEEDS = tuple(range(83000, 83008))
SAC_PARAMETERS = {
    "learning_rate": 1e-4,
    "buffer_size": 16384,
    "learning_starts": 1024,
    "batch_size": 128,
    "ent_coef": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": (4, "step"),
    "gradient_steps": 1,
}


def reset_seed(training_index, worker_index, episode_index):
    """Interleaved, disjoint 100,000-seed ranges; never held-out starts."""
    values = (training_index, worker_index, episode_index)
    if any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
        for x in values
    ):
        raise ValueError("Seed indices must be integers")
    if not (
        0 <= training_index < 2 and 0 <= worker_index < 4 and 0 <= episode_index < 25000
    ):
        raise ValueError("Reset stream exhausted or invalid owner")
    return 8200000 + 100000 * training_index + worker_index + 4 * episode_index


def planned_rows():
    return [
        {
            "training_seed": training_seed,
            "stage": stage,
            "evaluation_seed": seed,
            "status": "not_run",
        }
        for training_seed in TRAINING_SEEDS
        for stage in ("initial", "final")
        for seed in EVALUATION_SEEDS
    ]


def make_learner(env, seed, learner_class=None):
    """Fresh ordinary SAC138; zero deterministic residual before any update."""
    import torch
    from stable_baselines3 import SAC

    if seed not in TRAINING_SEEDS or env.action_space.shape != (138,):
        raise ValueError("Unregistered learner seed/action space")
    if env.observation_space.shape != (3601,):
        raise ValueError("Expected frozen-normalized native3601")
    model = (SAC if learner_class is None else learner_class)(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [256, 256]},
        seed=seed,
        device="cpu",
        **SAC_PARAMETERS,
    )
    with torch.no_grad():
        model.actor.mu.weight.zero_()
        model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(-5)
    return model


class PilotWorker(gym.Wrapper):
    """One persistent physical owner; private chunks and durable fault counters.

    Parent supplies an absolute monotonic deadline shared on one host and owns
    global limits. No worker may extend its 2048-transition allocation. A crash
    leaves the last confirmed counters plus a conservative allocation, not zero.
    """

    def __init__(
        self, env, folder, training_index, worker_index, deadline, raw_bytes=None
    ):
        super().__init__(env)
        reset_seed(training_index, worker_index, 0)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.raw_credit = RawCredit(self.folder, raw_bytes) if raw_bytes else None
        self.training_index, self.worker_index = training_index, worker_index
        self.deadline = float(deadline)
        self.episode_index = -1
        self.transitions = self.completed_episodes = 0
        self.chunk_index = 0
        self.residuals, self.commands, self.teachers, self.rewards = [], [], [], []
        self.trace = self.folder / "transitions.jsonl"
        self.trace.touch(exist_ok=False)
        self.events = self.folder / "episodes.jsonl"
        self.events.touch(exist_ok=False)
        self.step_wall_s = self.reset_wall_s = 0.0
        self.status = "ready"
        self.attempted_residual = None
        self.closed = False
        self._progress()

    def _progress(self):
        from torch import get_num_threads

        from .motor_protocol import pid_identity

        write_json(
            self.folder / "progress.json",
            {
                "pid": os.getpid(),
                "process_identity": pid_identity(),
                "torch_threads": get_num_threads(),
                "thread_limits": {
                    key: os.environ.get(key)
                    for key in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    )
                },
                "started_motor_episodes": self.episode_index + 1,
                "status": self.status,
                "training_seed": TRAINING_SEEDS[self.training_index],
                "worker_index": self.worker_index,
                "episode_index": self.episode_index,
                "transitions": self.transitions,
                "physics_steps": self.env.task.physics_total,
                "maximum_physics_allocation": 20480,
                "completed_motor_episodes": self.completed_episodes,
                "chunks": self.chunk_index,
                "unflushed_transitions": len(self.residuals),
                "monotonic_s": time.monotonic(),
                "step_wall_s": self.step_wall_s,
                "reset_wall_s": self.reset_wall_s,
                "bundle_hash": self.env.contract_hash,
            },
        )

    def _guard(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Per-seed600s or global wall deadline")
        if (self.folder.parent / "STOP").exists():
            raise RuntimeError("Parent stop requested")
        if self.transitions >= 2048 or self.env.task.physics_total + 10 > 20480:
            raise TimeoutError("Worker transition/physics allocation exhausted")

    def _flush(self):
        if not self.residuals:
            return
        target = self.folder / (f"commands-{self.chunk_index:05d}.npz")
        save_npz(
            target,
            residuals=np.asarray(self.residuals, dtype=np.float32),
            commands=np.asarray(self.commands, dtype=np.float32),
            teachers=np.asarray(self.teachers, dtype=np.float32),
            rewards=np.asarray(self.rewards, dtype=np.float64),
        )
        self.chunk_index += 1
        self.residuals.clear()
        self.commands.clear()
        self.teachers.clear()
        self.rewards.clear()

    def _fault(self, error):
        # NaN survives raw NPZ even when a JSON snapshot cannot be encoded.
        import mujoco

        from .physical_motor import INTEGRATION

        if self.raw_credit:
            self.raw_credit.emergency_mode = True
        state = np.empty(mujoco.mj_stateSize(self.env.task.model, INTEGRATION))
        mujoco.mj_getState(self.env.task.model, self.env.task.data, state, INTEGRATION)
        save_npz(
            self.folder / "fault-arrays.npz",
            state=state,
            attempted_residual=np.empty(0, np.float32)
            if self.attempted_residual is None
            else self.attempted_residual,
            held_command=np.asarray(
                (self.env._held or {}).get("command", []), np.float32
            ),
            observation=np.empty(0, np.float32)
            if self.env.task.last_obs is None
            else self.env.task.last_obs,
        )
        self.status = "fault"
        write_json(
            self.folder / "fault.json",
            {
                "type": type(error).__name__,
                "reason": str(error),
                "physics_steps": self.env.task.physics_total,
                "transitions": self.transitions,
            },
        )
        self._flush()
        self._progress()

    def reset(self, *, seed=None, options=None):
        began = time.monotonic()
        try:
            if options:
                raise ValueError("No unregistered worker reset options")
            # SB3 supplies its configured seed once. Physical reset streams are
            # separate, preregistered and all recorded; autoresets pass None.
            expected = TRAINING_SEEDS[self.training_index] + self.worker_index
            if seed is not None and (self.episode_index != -1 or seed != expected):
                raise ValueError("Unexpected vector seed override")
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Worker reset deadline")
            chosen = reset_seed(
                self.training_index, self.worker_index, self.episode_index + 1
            )
            observation, info = self.env.reset(seed=chosen)
            self.episode_index += 1
            self.status = "running"
            append_json(
                self.events,
                {
                    "event": "reset",
                    "episode_index": self.episode_index,
                    "seed": chosen,
                    "sb3_requested_seed": seed,
                    "physics_steps": self.env.task.physics_total,
                },
            )
            write_json(self.folder / "last-valid-snapshot.json", self.env.snapshot())
            self.reset_wall_s += time.monotonic() - began
            self._progress()
            return observation, info
        except Exception as error:
            self._fault(error)
            raise

    def step(self, residual):
        began = time.monotonic()
        try:
            self._guard()
            self.attempted_residual = np.asarray(residual, dtype=np.float32).copy()
            observation, reward, terminated, truncated, info = self.env.step(residual)
            if (
                not np.isfinite(observation).all()
                or not np.isfinite(reward)
                or info["bootstrap_mask"] != float(not terminated)
                or info["trace_mask"] != float(not (terminated or truncated))
            ):
                raise ValueError("Nonfinite transition or incorrect episode masks")
            if terminated or truncated:
                if not np.array_equal(observation, info["terminal_observation"]):
                    raise ValueError("Terminal observation ownership mismatch")
                self.completed_episodes += 1
            transition = self.env.last_transition
            self.residuals.append(np.asarray(transition["residual"], np.float32))
            self.commands.append(np.asarray(transition["command"], np.float32))
            self.teachers.append(np.asarray(transition["teacher"], np.float32))
            self.rewards.append(float(reward))
            self.transitions += 1
            row = {
                "transition": self.transitions,
                "episode_index": self.episode_index,
                "episode_seed": reset_seed(
                    self.training_index, self.worker_index, self.episode_index
                ),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "bootstrap_mask": info["bootstrap_mask"],
                "trace_mask": info["trace_mask"],
                "teacher_clipped_fraction": info["teacher_clipped_fraction"],
                "telemetry": transition["telemetry"],
                "native_reward": transition["native_reward"],
                "physics_steps": self.env.task.physics_total,
            }
            append_json(self.trace, row)
            self.step_wall_s += time.monotonic() - began
            if self.transitions % 64 == 0 or terminated or truncated:
                write_json(
                    self.folder / "last-valid-snapshot.json", self.env.snapshot()
                )
                self._flush()
                self._progress()
            return observation, reward, terminated, truncated, info
        except Exception as error:
            self._fault(error)
            raise

    def close(self):
        if self.closed:
            return
        self._flush()
        if self.status != "fault":
            self.status = "closed"
        self._progress()
        self.env.close()
        self.closed = True


def paired_conclusion(rows):
    """All32 evaluation attempts remain in the denominator, including faults."""
    if len(rows) != 32 or digest(
        [(r["training_seed"], r["stage"], r["evaluation_seed"]) for r in rows]
    ) != digest(
        [(r["training_seed"], r["stage"], r["evaluation_seed"]) for r in planned_rows()]
    ):
        raise ValueError("Evaluation schedule mismatch")
    findings = []
    for training_seed in TRAINING_SEEDS:
        selected = [r for r in rows if r["training_seed"] == training_seed]
        valid = all(r["status"] == "completed" for r in selected)
        pairs = []
        if valid:
            for seed in EVALUATION_SEEDS:
                a = next(
                    r
                    for r in selected
                    if r["stage"] == "initial" and r["evaluation_seed"] == seed
                )
                b = next(
                    r
                    for r in selected
                    if r["stage"] == "final" and r["evaluation_seed"] == seed
                )
                for row in (a, b):
                    if (
                        not np.isfinite(row["survival_s"])
                        or not 0 <= row["survival_s"] <= 20
                    ):
                        raise ValueError("Invalid restricted survival")
                pairs.append(
                    {
                        "evaluation_seed": seed,
                        "initial_s": a["survival_s"],
                        "final_s": b["survival_s"],
                        "delta_s": b["survival_s"] - a["survival_s"],
                    }
                )
        mean = float(np.mean([p["delta_s"] for p in pairs])) if valid else None
        five = sum(p["final_s"] >= 5 for p in pairs)
        findings.append(
            {
                "training_seed": training_seed,
                "all_evaluations_valid": valid,
                "mean_paired_delta_s": mean,
                "final_five_second_survivors": five,
                "directional_criteria_passed": valid and mean >= 0.5 and five >= 4,
                "pairs": pairs,
            }
        )
    return {
        "evaluation_directional_criteria_passed": all(
            r["directional_criteria_passed"] for r in findings
        ),
        "required_closed_loop_replays_still_separate": True,
        "seeds": findings,
    }
