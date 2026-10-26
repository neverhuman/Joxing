"""Bounded episodic policy search on an explicitly constrained muscle reach rig.

This is a fast engineering experiment, not a boxing policy or a motor-competence gate.
The pelvis is fixed, gravity is zero, all 700 muscles remain, and only named right-arm
groups receive learned two-phase excitation pulses. No state is written after reset.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from .provenance import digest, manifest, write_json
from .scene import SOURCE

GROUPS = (
    ("DELT1_r",),
    ("DELT2_r",),
    ("DELT3_r",),
    ("PECM1_r", "PECM2_r", "PECM3_r"),
    ("BIClong_r", "BICshort_r", "BRA_r"),
    ("TRIlong_r", "TRIlat_r", "TRImed_r"),
)


@dataclass(frozen=True)
class ReachConfig:
    seed: int = 7
    generations: int = 16
    population: int = 24
    elite: int = 6
    workers: int = 4
    episode_s: float = 0.35
    dt_s: float = 0.001
    target: tuple[float, float, float] = (0.25, -0.20, 1.10)


class ReachRig:
    def __init__(self, config: ReachConfig):
        self.config = config
        spec = mujoco.MjSpec.from_file(str(SOURCE))
        # Six native root coordinates are removed only in this named fixture variant.
        for key in list(spec.keys):
            spec.delete(key)
        for joint in list(spec.joints):
            if joint.name.startswith("pelvis_"):
                spec.delete(joint)
        spec.option.gravity = [0, 0, 0]
        spec.option.timestep = config.dt_s
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.hash = digest(spec.to_xml())
        self.groups = [
            np.array([self.model.actuator(name).id for name in group])
            for group in GROUPS
        ]
        self.hand_id = self.model.body("hand_r").id
        self.elbow = self.model.joint("elbow_flexion_r").qposadr[0]

    def rollout(self, params, seed: int, record=False):
        if np.shape(params) != (2, len(GROUPS)) or not np.isfinite(params).all():
            raise ValueError("Expected finite two-phase group logits")
        rng = np.random.default_rng(seed)
        target = np.asarray(self.config.target) + rng.uniform(-0.005, 0.005, 3)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.elbow] = 0.05 + rng.uniform(-0.01, 0.01)
        self.data.qvel[:] = rng.uniform(-0.005, 0.005, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        n = round(self.config.episode_s / self.config.dt_s)
        distances, trace, states = [], [], []
        excitations = 1 / (1 + np.exp(-np.clip(params, -12, 12)))
        for k in range(n):
            self.data.ctrl[:] = 0.005
            phase = min(1, 2 * k // n)
            for g, ix in enumerate(self.groups):
                self.data.ctrl[ix] = excitations[phase, g]
            mujoco.mj_step(self.model, self.data)
            if (
                not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
                or any(w.number for w in self.data.warning)
            ):
                raise RuntimeError("Invalid muscle rollout; halt experiment")
            mujoco.mj_kinematics(self.model, self.data)
            hand = self.data.xpos[self.hand_id].copy()
            distances.append(float(np.linalg.norm(hand - target)))
            if record:
                trace.append([self.data.time, *hand.tolist(), distances[-1]])
                states.append(self.data.qpos.copy())
        # Optimize a held terminal window, preventing one lucky transient crossing.
        error = float(np.mean(distances[-max(1, n // 5) :]))
        reward = -error - 0.002 * float(np.mean(excitations**2))
        return {
            "reward": reward,
            "error_m": error,
            "target": target.tolist(),
            "trace": trace,
            "qpos": states,
        }


_worker_rig = None


def _init_worker(config):
    global _worker_rig
    _worker_rig = ReachRig(config)


def _episode(job):
    params, seed = job
    result = _worker_rig.rollout(params, seed)
    return result["reward"], result["error_m"]


def train_reach(config: ReachConfig, out: Path) -> dict:
    if (
        config.workers < 1
        or not 1 < config.elite < config.population
        or config.generations < 1
    ):
        raise ValueError("Invalid CEM resource or population configuration")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", asdict(config))
    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    mean, std = np.full((2, len(GROUPS)), -2.0), np.full((2, len(GROUPS)), 1.5)
    initial = mean.copy()
    history = []
    with ProcessPoolExecutor(
        max_workers=config.workers, initializer=_init_worker, initargs=(config,)
    ) as pool:
        # Common training scenarios per generation; eval scenarios never select candidates.
        for generation in range(config.generations):
            candidates = rng.normal(mean, std, (config.population, *mean.shape))
            candidates[0] = mean
            result = list(
                pool.map(_episode, [(p, 100 + generation) for p in candidates])
            )
            rewards = np.array([r[0] for r in result])
            elite = candidates[np.argsort(rewards)[-config.elite :]]
            mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
            std = np.maximum(0.15, 0.2 * std + 0.8 * elite.std(axis=0))
            row = {
                "generation": generation,
                "episodes": (generation + 1) * config.population,
                "mean_reward": float(rewards.mean()),
                "best_reward": float(rewards.max()),
                "mean_error_m": float(np.mean([r[1] for r in result])),
            }
            history.append(row)
            np.savez_compressed(
                out / f"generation_{generation:03d}.npz",
                candidates=candidates,
                rewards=rewards,
                scenario_seed=100 + generation,
                mean=mean,
                std=std,
            )
            write_json(out / "progress.json", history)
            np.savez_compressed(
                out / "checkpoint.npz", mean=mean, std=std, initial=initial
            )
            print(
                f"generation={generation} mean_error_m={row['mean_error_m']:.4f}",
                flush=True,
            )
        eval_seeds = list(range(10000, 10032))
        before = list(pool.map(_episode, [(initial, s) for s in eval_seeds]))
        after = list(pool.map(_episode, [(mean, s) for s in eval_seeds]))
        # Equal-count fresh samples measure final distribution against random search controls.
        random_params = rng.normal(initial, 1.5, (len(eval_seeds), *mean.shape))
        random_control = list(pool.map(_episode, list(zip(random_params, eval_seeds))))
    errors_before = np.array([r[1] for r in before])
    errors_after = np.array([r[1] for r in after])
    improvement = errors_before - errors_after
    bootstrap = (
        np.random.default_rng(812)
        .choice(improvement, (2000, len(improvement)), replace=True)
        .mean(axis=1)
    )
    rig = ReachRig(config)
    traces = {}
    for name, params in (("initial", initial), ("trained", mean)):
        result = rig.rollout(params, eval_seeds[0], record=True)
        traces[name] = {k: v for k, v in result.items() if k != "qpos"}
        np.savez_compressed(
            out / f"{name}_replay.npz",
            qpos=np.array(result["qpos"]),
            params=params,
            target=result["target"],
        )
    report = manifest() | {
        "experiment": "constrained-muscle-reach-cem-v1",
        "config": asdict(config),
        "model_hash": rig.hash,
        "policy_hash": digest(mean.tolist()),
        "groups": GROUPS,
        "algorithm": "cross-entropy episodic policy search; elite maximum-likelihood Gaussian refit",
        "assistance": ["fixed pelvis", "zero gravity"],
        "action": "two-phase open-loop right-arm muscle pulses",
        "competitive_result": False,
        "training_episodes": config.generations * config.population,
        "evaluation_attempts_per_policy": len(eval_seeds),
        "invalid_episodes": 0,
        "eval_seeds": eval_seeds,
        "initial_error_m": float(errors_before.mean()),
        "trained_error_m": float(errors_after.mean()),
        "random_control_error_m": float(np.mean([r[1] for r in random_control])),
        "paired_error_reduction_m": float(improvement.mean()),
        "paired_scenario_bootstrap_95ci_m": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "wall_s": time.perf_counter() - started,
        "history": history,
        "traces": traces,
        "limitations": [
            "one training seed",
            "one target neighborhood",
            "open-loop controller",
            "no punches, contacts, balance or self-play",
            "CI covers reset variation, not independent training seeds",
        ],
    }
    write_json(out / "result.json", report)
    return report
