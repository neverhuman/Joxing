"""Bounded SAC pilot and immutable paired evaluation for the gravity diagnostic."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .balance import BalanceConfig
from .provenance import digest, manifest, write_json
from .stance_env import StanceEnv


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FaultRecorder(gym.Wrapper):
    def __init__(self, env, path):
        super().__init__(env)
        self.path = Path(path)
        self.attempt = 0

    def step(self, action):
        snapshot = self.env.snapshot()
        self.attempt += 1
        try:
            return self.env.step(action)
        except Exception as error:
            write_json(
                self.path / f"fault-step-{self.attempt}.json",
                {
                    "error": str(error),
                    "attempted_transition": self.attempt,
                    "last_valid": snapshot,
                    "action": np.nan_to_num(action).tolist(),
                    "invalid": True,
                },
            )
            raise


def make_worker(out, index, seed):
    from stable_baselines3.common.monitor import Monitor

    env = StanceEnv()
    env.reset(seed=seed + index)
    return Monitor(
        FaultRecorder(env, Path(out) / f"worker-{index}"),
        str(Path(out) / f"worker-{index}"),
    )


def evaluate_policy(model, env, ids, name, out):
    rows = [
        {
            "attempt_id": f"{name}:{scenario}",
            "policy": name,
            "scenario_id": scenario,
            "status": "not_run",
            "invalid": False,
        }
        for scenario in ids
    ]
    checkpoints = {}
    for index, scenario in enumerate(ids):
        try:
            obs, info = env.reset(options={"scenario_id": scenario})
            initial_hash = digest(env.snapshot()["state"])
            snapshot = None
            while not env.done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _, _ = env.step(action)
                if snapshot is None and env.tick >= 100:
                    snapshot = env.snapshot()
            row = env.result() | {
                "attempt_id": f"{name}:{scenario}",
                "policy": name,
                "initial_state_hash": initial_hash,
                "invalid": False,
                "reset_audit": info["reset_audit"],
            }
            if scenario == ids[0]:
                checkpoints[name] = {
                    "snapshot": snapshot,
                    "final_state": env.snapshot()["state"],
                    "result": env.result(),
                    "trace": env.trace,
                }
        except Exception as error:
            row = {
                "attempt_id": f"{name}:{scenario}",
                "policy": name,
                "scenario_id": scenario,
                "status": "invalid",
                "reason": str(error),
                "invalid": True,
            }
            rows[index] = row
            write_json(out / f"{name}-evaluations.json", rows)
            write_json(out / "evaluation-fault.json", row)
            raise
        rows[index] = row
        write_json(out / f"{name}-evaluations.json", rows)
    write_json(out / f"{name}-replay.json", checkpoints[name])
    return rows


def paired_summary(before, after):
    if [x["scenario_id"] for x in before] != [x["scenario_id"] for x in after]:
        raise ValueError("Unpaired evaluation scenarios")
    if any(x["invalid"] or x["status"] == "not_run" for x in before + after):
        raise ValueError("Invalid evaluation cannot enter valid summary")
    a = np.array([x["elapsed_sim_s"] for x in before])
    b = np.array([x["elapsed_sim_s"] for x in after])
    delta = b - a
    draws = (
        np.random.default_rng(61004)
        .choice(delta, (10000, len(delta)), replace=True)
        .mean(axis=1)
    )
    return {
        "attempts": len(before) + len(after),
        "invalid": 0,
        "initial_mean_survival_s": float(a.mean()),
        "trained_mean_survival_s": float(b.mean()),
        "paired_survival_gain_s": float(delta.mean()),
        "paired_reset_bootstrap_95ci_s": np.quantile(draws, [0.025, 0.975]).tolist(),
        "initial_5s_survivors": int(sum(a >= 5)),
        "trained_5s_survivors": int(sum(b >= 5)),
        "trained_20s_successes": int(sum(x["quiet_success"] for x in after)),
        "pilot_continuation_gate": bool(sum(b >= 5) >= 16 and len(after) == 32),
        "motor_gate_passed": False,
        "scope": "single-seed development pilot; reset CI is not training-seed uncertainty",
    }


def train_stance(
    out: Path, steps=100000, workers=4, seed=73, wall_s=1800.0, eval_count=32
):
    import stable_baselines3
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if (
        not 1 <= workers <= 4
        or steps < 1
        or steps % workers
        or not 1 <= eval_count <= 32
        or not np.isfinite(wall_s)
        or wall_s <= 0
    ):
        raise ValueError("Invalid bounded pilot budget")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    eval_env = StanceEnv(training=False)
    ids = list(range(30000, 30000 + eval_count))
    protocol = manifest() | {
        "experiment": "v0.0.4-native-gravity-sac-development-pilot",
        "seed": seed,
        "steps": steps,
        "workers": workers,
        "wall_cap_s": wall_s,
        "eval_ids": ids,
        "config": asdict(BalanceConfig()),
        "specification": eval_env.specification(),
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "assistance_during_evaluation": [],
        "training_reset_ids": [100000, 100063],
        "algorithm": {
            "name": "SAC",
            "net_arch": [256, 256],
            "learning_rate": 0.0003,
            "learning_starts": 10000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "entropy": "auto",
            "replay_capacity": 100000,
            "gradient_steps_per_vector_step": workers,
        },
        "primary_pilot_metric": "paired development survival seconds",
        "bootstrap_seed": 61004,
        "competitive": False,
        "formal_proof": False,
    }
    protocol["protocol_hash"] = digest(protocol)
    write_json(out / "protocol.json", protocol)
    write_json(out / "model-audit.json", eval_env.rig.audit())
    factories = [partial(make_worker, str(out), i, seed) for i in range(workers)]
    vec = (
        DummyVecEnv(factories)
        if workers == 1
        else SubprocVecEnv(factories, start_method="spawn")
    )
    progress = []

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last = 0.0
            self.stop_reason = None

        def _on_step(self):
            elapsed = time.perf_counter() - started
            if elapsed - self.last >= 15 or self.num_timesteps >= steps:
                self.last = elapsed
                row = {
                    "control_transitions": self.num_timesteps,
                    "wall_s": elapsed,
                    "physics_substeps": int(sum(vec.get_attr("physics_total"))),
                    "preparation_substeps": int(sum(vec.get_attr("preparation_total"))),
                }
                progress.append(row)
                write_json(out / "progress.json", progress)
                temporary = out / "checkpoint.tmp.zip"
                self.model.save(temporary)
                temporary.replace(out / "checkpoint.zip")
                print(json.dumps(row), flush=True)
            if elapsed >= wall_s:
                self.stop_reason = "wall_cap"
                return False
            return True

    try:
        model = SAC(
            "MlpPolicy",
            vec,
            policy_kwargs={"net_arch": [256, 256]},
            learning_rate=0.0003,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=workers,
            ent_coef="auto",
            seed=seed,
            device="cpu",
            verbose=0,
        )
        model.save(out / "initial.zip")
        before = evaluate_policy(model, eval_env, ids, "initial", out)
        callback = Progress()
        model.learn(total_timesteps=steps, callback=callback)
        model.save(out / "final.zip")
        training_physics = int(sum(vec.get_attr("physics_total")))
        preparation_physics = int(sum(vec.get_attr("preparation_total")))
        trained = model.num_timesteps
        after = evaluate_policy(model, eval_env, ids, "trained", out)
        report = manifest() | {
            "protocol_hash": protocol["protocol_hash"],
            "seed": seed,
            "summary": paired_summary(before, after),
            "training_transitions": trained,
            "training_physics_substeps": training_physics,
            "preparation_substeps": preparation_physics,
            "wall_s": time.perf_counter() - started,
            "stop_reason": callback.stop_reason,
            "scheduled_training_complete": trained >= steps
            and callback.stop_reason is None,
            "checkpoint_hashes": {
                n: file_hash(out / f"{n}.zip") for n in ("initial", "final")
            },
            "evaluation_attempts": len(before) + len(after),
            "invalid_attempts": 0,
            "evaluation_physics_substeps": sum(
                x["physics_steps"] for x in before + after
            ),
            "limitations": [
                "development data only; no held-out motor proof",
                "one training seed",
                "incomplete boxing collision envelope",
                "normal gravity, but reset-only preparation/projection",
                "no self-play or qualified GPU physics",
            ],
        }
        write_json(out / "result.json", report)
        return report
    except Exception as error:
        write_json(
            out / "fault.json",
            manifest()
            | {
                "error": str(error),
                "invalid_run": True,
                "wall_s": time.perf_counter() - started,
            },
        )
        raise
    finally:
        vec.close()


def balance_probe(out: Path):
    from .balance import BalanceModel
    from .physics import step_contacts
    from .support import floor_loads

    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rig = BalanceModel()
    resets = []
    for scenario in range(30000, 30032):
        _, audit = rig.prepare(scenario)
        resets.append(audit)
    write_json(out / "resets.json", resets)
    write_json(out / "model-audit.json", rig.audit())
    rows = []
    for dt in (0.002, 0.001, 0.0005):
        probe = BalanceModel(BalanceConfig(dt_s=dt))
        for name, joint, velocity in (
            ("forward", "pelvis_tilt", 1.0),
            ("backward", "pelvis_tilt", -1.0),
            ("right", "pelvis_list", 1.0),
            ("left", "pelvis_list", -1.0),
        ):
            data, audit = probe.prepare(30000)
            data.qvel[probe.model.joint(joint).dofadr[0]] += velocity
            min_sv = 1.0
            peak_load = 0.0
            max_tilt = 0.0
            for tick in range(round(0.8 / dt)):
                contacts = step_contacts(probe.model, data)
                loads = floor_loads(contacts, probe.floor, probe.roles)
                peak_load = max(peak_load, sum(loads.values()))
                mujoco.mj_kinematics(probe.model, data)
                tilt = float(
                    np.arccos(
                        np.clip(data.xmat[probe.pelvis].reshape(3, 3)[2, 1], -1, 1)
                    )
                )
                max_tilt = max(max_tilt, tilt)
                if tick % 20 == 0:
                    jp = np.zeros((3, probe.model.nv))
                    jr = jp.copy()
                    mujoco.mj_jacBody(probe.model, data, jp, jr, probe.pelvis)
                    min_sv = min(
                        min_sv,
                        float(
                            np.linalg.svd(
                                np.concatenate([jp, jr])[:, :6], compute_uv=False
                            )[-1]
                        ),
                    )
            rows.append(
                {
                    "direction": name,
                    "dt_s": dt,
                    "physics_steps": round(0.8 / dt),
                    "min_root_jacobian_singular_value": min_sv,
                    "max_pelvis_tilt_rad": max_tilt,
                    "peak_support_N": peak_load,
                    "invalid": False,
                    "model_hash": probe.hash,
                }
            )
            write_json(out / "root-probes.json", rows)
    report = manifest() | {
        "model": rig.audit(),
        "valid_resets": len(resets),
        "root_probes": rows,
        "wall_s": time.perf_counter() - started,
        "scope": "reset and finite root-stress diagnostics, not motor competence or full singularity exclusion",
    }
    write_json(out / "result.json", report)
    return report
