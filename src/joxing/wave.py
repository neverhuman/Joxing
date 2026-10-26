"""Preregistered v0.0.5 motor experiments; unchanged physical task and gates."""

from __future__ import annotations

import copy
import hashlib
import json
import resource
import shutil
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

from .provenance import digest, manifest, source_hash, write_json
from .stance_env import StanceEnv
from .stance_train import FaultRecorder, file_hash

ARMS = {
    "A": {"entropy": "auto", "reward_scale": 1.0},
    "B": {"entropy": 0.0001, "reward_scale": 1.0},
    "C": {"entropy": "auto", "reward_scale": 1000.0},
    "D": {"entropy": 0.0001, "reward_scale": 1000.0},
}
SPLITS = {
    "diagnostic": list(range(62000, 62008)),
    "screen": list(range(60000, 60032)),
    "confirmation": list(range(61000, 61032)),
    "training": list(range(110000, 110064)),
}


def validate_protocol(value):
    """Reject accidental postregistration changes to the controlled experiment."""
    expected = {
        "schema",
        "description",
        "arms",
        "splits",
        "seeds",
        "learner",
        "caps",
        "selection",
        "constants",
        "profile",
        "experiments",
    }
    if set(value) != expected or value["schema"] != "joxing-wave-0.0.5":
        raise ValueError("Unknown or missing protocol fields")
    if value["arms"] != ARMS or value["splits"] != SPLITS:
        raise ValueError("Registered arms/splits cannot change")
    if value["seeds"] != {"screen": 149, "confirmation": [263, 431]}:
        raise ValueError("Registered seeds cannot change")
    if value["learner"] != {
        "net_arch": [128, 128],
        "batch_size": 64,
        "buffer_size": 50000,
        "learning_starts": 2000,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "tau": 0.005,
        "gradient_steps": 1,
        "workers": 4,
        "steps": 40000,
    }:
        raise ValueError("Registered common learner cannot change")
    if value["caps"] != {
        "run_wall_s": 900,
        "wave_wall_s": 14400,
        "physics_steps": 40000000,
        "raw_bytes": 10737418240,
        "minimum_disk_bytes": 10737418240,
        "minimum_memory_bytes": 4294967296,
    }:
        raise ValueError("Registered resource caps cannot change")
    if value["selection"] != {"gain_s": 0.25, "over_A_s": 0.1, "bootstrap_seed": 61005}:
        raise ValueError("Registered selection cannot change")
    if value["constants"] != [0.005, 0.02, 0.05, 0.1, 0.2, 0.4]:
        raise ValueError("Registered constants cannot change")
    if value["profile"] != {
        "transitions": 256,
        "warmup": 20,
        "updates": 100,
        "repeats": 3,
        "minimum_speedup": 2.0,
    }:
        raise ValueError("Registered profile cannot change")
    ids = [x["id"] for x in value["experiments"]]
    if ids != [
        "B0",
        "P0",
        "S-A",
        "S-B",
        "S-C",
        "S-D",
        "R-A-263",
        "R-W-263",
        "R-A-431",
        "R-W-431",
        "X0",
    ]:
        raise ValueError("Missing or duplicate planned experiment")
    if any(
        not x.get("hypothesis") or not x.get("description")
        for x in value["experiments"]
    ):
        raise ValueError("Experiments need descriptions before execution")
    return copy.deepcopy(value)


def canonical_actor_hash(actor):
    result = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        result.update(json.dumps([name, list(a.shape), str(a.dtype)]).encode())
        result.update(a.tobytes())
    return result.hexdigest()


class TrainingContract(gym.Wrapper):
    """Only training reward units and purpose-separated reset sampling change."""

    def __init__(self, env, scale, seed):
        super().__init__(env)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Reward scale must be positive finite")
        self.scale = scale
        self.reset_rng = np.random.default_rng(seed)
        self.reset_wall_s = 0.0
        self.raw_reward_sum = 0.0
        self.scaled_reward_sum = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.reset_rng = np.random.default_rng(seed)
        if options is None:
            options = {"scenario_id": int(self.reset_rng.choice(SPLITS["training"]))}
        started = time.perf_counter()
        result = self.env.reset(seed=seed, options=options)
        self.reset_wall_s += time.perf_counter() - started
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled = reward * self.scale
        self.raw_reward_sum += reward
        self.scaled_reward_sum += scaled
        return (
            obs,
            scaled,
            terminated,
            truncated,
            info | {"raw_reward": reward, "scaled_reward": scaled},
        )


class ConstantPolicy:
    def __init__(self, excitation):
        self.excitation = excitation
        self.action = np.full(
            138, 0.5 + np.log(excitation / (1 - excitation)) / 5, dtype=np.float32
        )
        if np.any(abs(self.action) > 1):
            raise ValueError("Unreachable constant excitation")

    def predict(self, obs, deterministic=True):
        return self.action.copy(), None


def bootstrap_pair(before, after):
    if not before or len(before) != len(after):
        raise ValueError("Missing paired rows")
    for rows in (before, after):
        ids = [r["scenario_id"] for r in rows]
        if len(set(ids)) != len(ids) or any(
            r.get("invalid") or r["status"] in {"not_run", "invalid"} for r in rows
        ):
            raise ValueError("Duplicate, missing or invalid evaluation")
    if [(r["scenario_id"], r["initial_state_hash"]) for r in before] != [
        (r["scenario_id"], r["initial_state_hash"]) for r in after
    ]:
        raise ValueError("Unpaired physical starts")
    a = np.array([r["elapsed_sim_s"] for r in before])
    b = np.array([r["elapsed_sim_s"] for r in after])
    delta = b - a
    draws = np.random.default_rng(61005).choice(delta, (10000, len(delta))).mean(axis=1)
    return {
        "attempts": len(a) * 2,
        "invalid": 0,
        "initial_mean_survival_s": float(a.mean()),
        "trained_mean_survival_s": float(b.mean()),
        "paired_survival_gain_s": float(delta.mean()),
        "paired_reset_bootstrap_95ci_s": np.quantile(draws, [0.025, 0.975]).tolist(),
        "trained_5s_survivors": int(sum(b >= 5)),
        "pilot_continuation_gate": bool(len(b) == 32 and sum(b >= 5) >= 16),
        "motor_gate_passed": False,
    }


def select_arm(results):
    if set(results) != set(ARMS) or any(
        not r["scheduled_training_complete"] or r["summary"]["invalid"]
        for r in results.values()
    ):
        return {
            "qualified": False,
            "arm": None,
            "reason": "incomplete_or_invalid_screen",
        }
    winner = max(ARMS, key=lambda a: results[a]["summary"]["trained_mean_survival_s"])
    s = results[winner]["summary"]
    over_a = (
        s["trained_mean_survival_s"]
        - results["A"]["summary"]["trained_mean_survival_s"]
    )
    passed = s["paired_survival_gain_s"] >= 0.25 and (winner == "A" or over_a >= 0.1)
    return {
        "qualified": passed,
        "arm": winner,
        "gain_s": s["paired_survival_gain_s"],
        "over_A_s": over_a,
        "reason": "replicate" if passed else "no_meaningful_screen_gain",
    }


def evaluate(policy, env, ids, name, out, actor_hash=None):
    out = Path(out)
    path = out / f"{name}-evaluations.json"
    if path.exists():
        raise FileExistsError(path)
    rows = [
        {
            "attempt_id": f"{name}:{s}",
            "policy": name,
            "scenario_id": s,
            "status": "not_run",
            "invalid": False,
        }
        for s in ids
    ]
    write_json(path, rows)
    for index, scenario in enumerate(ids):
        last_valid = None
        try:
            obs, info = env.reset(options={"scenario_id": scenario})
            start = env.snapshot()
            initial_hash = digest(start["state"])
            root = env.data.qvel[:6].tolist()
            tilt = env.kinematics()[1]
            replay = None
            actions = []
            while not env.done:
                last_valid = env.snapshot()
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, _, _, _ = env.step(action)
                if replay is not None:
                    actions.append(np.asarray(action).tolist())
                if replay is None and env.tick >= 100:
                    replay = env.snapshot()
            rows[index] = env.result() | {
                "attempt_id": f"{name}:{scenario}",
                "policy": name,
                "initial_state_hash": initial_hash,
                "actor_hash": actor_hash,
                "initial_root_qvel": root,
                "initial_tilt_rad": tilt,
                "invalid": False,
                "reset_audit": info["reset_audit"],
            }
            if index == 0:
                write_json(
                    out / f"{name}-replay.json",
                    {
                        "snapshot": replay,
                        "actions": actions,
                        "final_state": env.snapshot()["state"],
                        "result": env.result(),
                    },
                )
        except Exception as error:
            rows[index].update(status="invalid", invalid=True, reason=str(error))
            write_json(path, rows)
            write_json(
                out / f"{name}-fault.json",
                {"row": rows[index], "last_valid": last_valid},
            )
            raise
        write_json(path, rows)
    return rows


def make_worker(out, index, seed, scale):
    from stable_baselines3.common.monitor import Monitor

    env = TrainingContract(
        FaultRecorder(StanceEnv(), Path(out) / f"worker-{index}"), scale, seed + index
    )
    return Monitor(env, str(Path(out) / f"worker-{index}"))


def timed_sac_class():
    from stable_baselines3 import SAC

    class TimedSAC(SAC):
        update_wall_s = 0.0
        inference_wall_s = 0.0

        def train(self, gradient_steps, batch_size=64):
            started = time.perf_counter()
            super().train(gradient_steps, batch_size)
            self.update_wall_s += time.perf_counter() - started

        def predict(self, *args, **kwargs):
            started = time.perf_counter()
            result = super().predict(*args, **kwargs)
            self.inference_wall_s += time.perf_counter() - started
            return result

    return TimedSAC


def diagnostic(model, observations):
    """Sampling diagnostics cannot advance the training RNG stream."""
    import torch

    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(61005)
        obs = torch.as_tensor(observations, device=model.device)
        _, logp = model.actor.action_log_prob(obs)
        alpha = (
            model.log_ent_coef.exp()
            if model.ent_coef_optimizer is not None
            else model.ent_coef_tensor
        )
        return {
            "alpha": float(alpha.item()),
            "mean_log_prob": float(logp.mean().item()),
            "entropy_reward_mean": float((-alpha * logp).mean().item()),
        }


def available_memory():
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    return (
        int(
            next(
                x.split()[1]
                for x in path.read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )


def check_resources(out, caps, start=False):
    free = shutil.disk_usage(out).free
    memory = available_memory()
    multiplier = 2 if start else 1
    if free < multiplier * caps["minimum_disk_bytes"] or (
        memory is not None and memory < multiplier * caps["minimum_memory_bytes"]
    ):
        raise RuntimeError("Shared-host disk/memory cap")
    return {"disk_free_bytes": free, "available_memory_bytes": memory}


def train_run(
    out,
    protocol,
    arm,
    seed,
    ids,
    provenance,
    budget_check,
    *,
    defer_initial_evaluation=False,
):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    out.mkdir(parents=True, exist_ok=False)
    setting = ARMS[arm]
    cfg = protocol["learner"]
    write_json(
        out / "description.json",
        {
            "arm": arm,
            "seed": seed,
            "setting": setting,
            "learner": cfg,
            "evaluation_ids": ids,
            "provenance": provenance,
            "training_contract_hash": digest(
                setting | {"training_ids": SPLITS["training"]}
            ),
        },
    )
    started = time.perf_counter()
    torch.set_num_threads(1)
    env = StanceEnv(training=False)
    write_json(out / "specification.json", env.specification())
    vec = SubprocVecEnv(
        [partial(make_worker, out, i, seed, setting["reward_scale"]) for i in range(4)],
        start_method="spawn",
    )
    model = timed_sac_class()(
        "MlpPolicy",
        vec,
        policy_kwargs={"net_arch": cfg["net_arch"]},
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        train_freq=1,
        gradient_steps=1,
        ent_coef=setting["entropy"],
        seed=seed,
        device="cpu",
        verbose=0,
    )
    progress = []

    def counters():
        return {
            "physics_substeps": int(sum(vec.get_attr("physics_total"))),
            "preparation_substeps": int(sum(vec.get_attr("preparation_total"))),
            "reset_wall_s": float(sum(vec.get_attr("reset_wall_s"))),
            "raw_reward_sum": float(sum(vec.get_attr("raw_reward_sum"))),
            "scaled_reward_sum": float(sum(vec.get_attr("scaled_reward_sum"))),
            "learner_maxrss_native_units": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "available_memory_bytes": available_memory(),
        }

    class Progress(BaseCallback):
        last = 0.0
        stop_reason = None

        def _on_step(self):
            elapsed = time.perf_counter() - started
            if elapsed - self.last >= 30 or self.num_timesteps >= cfg["steps"]:
                self.last = elapsed
                counts = counters()
                budget_check(
                    counts["physics_substeps"] + counts["preparation_substeps"]
                )
                check_resources(out, protocol["caps"])
                row = counts | {
                    "transitions": self.num_timesteps,
                    "wall_s": elapsed,
                    "updates": model._n_updates,
                    "update_wall_s": model.update_wall_s,
                    "inference_wall_s": model.inference_wall_s,
                    "learner": {
                        k: float(v)
                        for k, v in model.logger.name_to_value.items()
                        if k.startswith("train/") and np.isscalar(v)
                    },
                    "diagnostic": diagnostic(model, model._last_obs),
                }
                if not np.isfinite([v for v in row["learner"].values()]).all():
                    raise FloatingPointError("Nonfinite learner losses")
                progress.append(row)
                write_json(out / "progress.json", progress)
                model.save(out / "checkpoint.tmp.zip")
                (out / "checkpoint.tmp.zip").replace(out / "checkpoint.zip")
                print(json.dumps({"run": out.name, **row}), flush=True)
            if elapsed >= protocol["caps"]["run_wall_s"]:
                self.stop_reason = "wall_cap"
                return False
            return True

    try:
        initial_hash = canonical_actor_hash(model.actor)
        model.actor.save(out / "initial-actor.pt")
        # Confirmation scenarios stay unopened until the scheduled final actor
        # is frozen. The initial actor was fixed before any learning.
        confirmation = defer_initial_evaluation or ids == SPLITS["confirmation"]
        before = (
            None
            if confirmation
            else evaluate(model.actor, env, ids, "initial", out, initial_hash)
        )
        callback = Progress()
        model.learn(total_timesteps=cfg["steps"], callback=callback)
        counts = counters()
        final_hash = canonical_actor_hash(model.actor)
        model.actor.save(out / "final-actor.pt")
        if confirmation:
            from stable_baselines3.sac.policies import Actor

            initial_actor = Actor.load(out / "initial-actor.pt", device="cpu")
            before = evaluate(initial_actor, env, ids, "initial", out, initial_hash)
        after = evaluate(model.actor, env, ids, "trained", out, final_hash)
        result = {
            "arm": arm,
            "seed": seed,
            "summary": bootstrap_pair(before, after),
            "scheduled_training_complete": model.num_timesteps >= cfg["steps"]
            and callback.stop_reason is None,
            "stop_reason": callback.stop_reason,
            "training_transitions": model.num_timesteps,
            "counts": counts,
            "evaluation_physics_substeps": sum(
                r["physics_steps"] for r in before + after
            ),
            "evaluation_preparation_substeps": env.preparation_total,
            "wall_s": time.perf_counter() - started,
            "update_wall_s": model.update_wall_s,
            "inference_wall_s": model.inference_wall_s,
            "updates": model._n_updates,
            "initial_actor_hash": initial_hash,
            "final_actor_hash": final_hash,
            "actor_files": {
                n: file_hash(out / f"{n}-actor.pt") for n in ("initial", "final")
            },
            "final_diagnostic": diagnostic(model, np.array([env.observe()])),
            "source_unchanged": source_hash() == provenance["source_hash"],
        }
        if not result["source_unchanged"]:
            raise RuntimeError("Executing source changed")
        write_json(out / "result.json", result)
        return result
    except Exception as error:
        write_json(
            out / "fault.json",
            {
                "error": str(error),
                "wall_s": time.perf_counter() - started,
                "invalid_run": True,
            },
        )
        try:
            model.save(out / "fault-checkpoint.zip")
            write_json(
                out / "fault-states.json",
                {"last_valid_workers": vec.env_method("snapshot")},
            )
        except Exception as capture_error:  # noqa: BLE001 -- preserve original fault
            write_json(out / "fault-capture-error.json", {"error": str(capture_error)})
        raise
    finally:
        vec.close()


def profile_updates(out, protocol):
    """Common real data; time SB3 actor, critic and temperature optimization."""
    import torch
    from stable_baselines3.common.logger import configure

    out.mkdir(exist_ok=False)
    env = StanceEnv(training=False)
    obs, _ = env.reset(options={"scenario_id": 62000})
    dataset = []
    policy = ConstantPolicy(0.02)
    for _ in range(256):
        action, _ = policy.predict(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        dataset.append(
            (
                obs.copy(),
                next_obs.copy(),
                action.copy(),
                reward,
                terminated or truncated,
                {"TimeLimit.truncated": truncated and not terminated},
            )
        )
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(options={"scenario_id": 62000})
    arrays = {
        k: np.array([row[i] for row in dataset])
        for i, k in enumerate(("obs", "next_obs", "actions", "rewards", "dones"))
    }
    np.savez_compressed(out / "common-transitions.npz", **arrays)
    rows = []
    for label, arch, batch in (
        ("reference", [256, 256], 256),
        ("optimized", [128, 128], 64),
    ):
        for repeat in range(3):
            torch.set_num_threads(1)
            model = timed_sac_class()(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": arch},
                batch_size=batch,
                buffer_size=512,
                learning_starts=0,
                seed=149,
                device="cpu",
            )
            model.set_logger(configure(str(out / "logger"), []))
            for a, b, c, d, e, f in dataset:
                model.replay_buffer.add(
                    a[None], b[None], c[None], np.array([d]), np.array([e]), [f]
                )
            before = canonical_actor_hash(model.actor)
            model.train(20, batch)
            started = time.perf_counter()
            model.train(100, batch)
            seconds = time.perf_counter() - started
            losses = {
                k: float(v)
                for k, v in model.logger.name_to_value.items()
                if np.isscalar(v)
            }
            rows.append(
                {
                    "label": label,
                    "repeat": repeat,
                    "net_arch": arch,
                    "batch": batch,
                    "timed_updates": 100,
                    "wall_s": seconds,
                    "updates_per_s": 100 / seconds,
                    "finite": bool(np.isfinite(list(losses.values())).all()),
                    "actor_changed": canonical_actor_hash(model.actor) != before,
                    "losses": losses,
                }
            )
            write_json(out / "samples.json", rows)
    medians = {
        name: float(np.median([r["updates_per_s"] for r in rows if r["label"] == name]))
        for name in ("reference", "optimized")
    }
    speedup = medians["optimized"] / medians["reference"]
    result = {
        "samples": rows,
        "medians": medians,
        "speedup": speedup,
        "passed": speedup >= 2
        and all(r["finite"] and r["actor_changed"] for r in rows),
        "physics_substeps": env.physics_total,
        "preparation_substeps": env.preparation_total,
        "dataset_sha256": file_hash(out / "common-transitions.npz"),
    }
    write_json(out / "result.json", result)
    return result


def export_legacy_actions(out):
    """Preselected v4 snapshots: separate engine replay from closed-loop arithmetic."""
    from stable_baselines3 import SAC

    from .provenance import ROOT

    out.mkdir(exist_ok=False)
    env = StanceEnv(training=False)
    rows = []
    for name, checkpoint in (("initial", "initial"), ("trained", "final")):
        base = ROOT / "artifacts/v0.0.4/pilot"
        record = json.loads((base / f"{name}-replay.json").read_text())
        model = SAC.load(base / f"{checkpoint}.zip", device="cpu")
        env.restore(record["snapshot"])
        actions = []
        while not env.done:
            a, _ = model.predict(env.observe(), deterministic=True)
            actions.append(a.tolist())
            env.step(a)
        actual = np.asarray(env.snapshot()["state"])
        error = float(np.max(abs(actual - np.asarray(record["final_state"]))))
        exported = record | {
            "actions": actions,
            "exported_final_state": actual.tolist(),
            "exported_result": env.result(),
            "source_policy_sha256": file_hash(base / f"{checkpoint}.zip"),
        }
        write_json(out / f"{name}-actions.json", exported)
        env.restore(record["snapshot"])
        for a in actions:
            env.step(a)
        same = float(np.max(abs(np.asarray(env.snapshot()["state"]) - actual)))
        rows.append(
            {
                "policy": name,
                "legacy_reference_error": error,
                "same_host_action_replay_error": same,
                "passed": error < 1e-9 and same < 1e-9,
            }
        )
    result = {
        "policies": rows,
        "passed": all(r["passed"] for r in rows),
        "physics_substeps": env.physics_total,
        "preparation_substeps": env.preparation_total,
    }
    write_json(out / "result.json", result)
    if not result["passed"]:
        raise RuntimeError("Same-host legacy continuation failure")
    return result


def run_wave(protocol_path, out):
    import stable_baselines3
    import torch

    protocol = validate_protocol(json.loads(Path(protocol_path).read_text()))
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    provenance = manifest() | {
        "protocol_hash": digest(protocol),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "resources": check_resources(out, protocol["caps"], True),
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "provenance.json", provenance)
    registry = {
        x["id"]: x | {"status": "not_run", "reason": "pending_dependencies"}
        for x in protocol["experiments"]
    }
    completed_physics = 0

    def event(key, status, **data):
        registry[key].update(status=status, **data)
        write_json(out / "registry.json", registry)
        with (out / "events.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "experiment": key,
                        "status": status,
                        "wall_s": time.perf_counter() - started,
                        **data,
                    }
                )
                + "\n"
            )

    def budget(current=0):
        if (
            time.perf_counter() - started > protocol["caps"]["wave_wall_s"]
            or completed_physics + current > protocol["caps"]["physics_steps"]
        ):
            raise RuntimeError("Wave wall/physics cap")
        if (
            sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            > protocol["caps"]["raw_bytes"]
        ):
            raise RuntimeError("Wave artifact cap")

    def execute(key, operation):
        nonlocal completed_physics
        budget()
        event(key, "running", reason="dependencies_passed")
        write_json(
            out / f"{key}-description.json",
            registry[key]
            | {
                "protocol_hash": provenance["protocol_hash"],
                "source_commit": provenance["code_commit"],
            },
        )
        try:
            result = operation()
            counts = result.get("counts", {})
            completed_physics += (
                counts.get("physics_substeps", result.get("physics_substeps", 0))
                + counts.get(
                    "preparation_substeps", result.get("preparation_substeps", 0)
                )
                + result.get("evaluation_physics_substeps", 0)
                + result.get("evaluation_preparation_substeps", 0)
            )
            status = (
                "completed"
                if result.get("scheduled_training_complete", True)
                else "incomplete"
            )
            event(
                key,
                status,
                result=f"{key}/result.json",
                reason=result.get("stop_reason") or "finished",
                accounted_physics_substeps=completed_physics,
            )
            return result
        except Exception as error:
            event(key, "failed", reason=str(error))
            raise

    def baselines():
        folder = out / "B0"
        folder.mkdir()
        env = StanceEnv(training=False)
        means = []
        for exc in protocol["constants"]:
            rows = evaluate(
                ConstantPolicy(exc), env, SPLITS["diagnostic"], f"exc-{exc}", folder
            )
            means.append(
                {
                    "excitation": exc,
                    "mean_survival_s": float(
                        np.mean([r["elapsed_sim_s"] for r in rows])
                    ),
                }
            )
        chosen = max(means, key=lambda r: r["mean_survival_s"])["excitation"]
        evaluate(
            ConstantPolicy(chosen), env, SPLITS["screen"], "screen-constant", folder
        )
        result = {
            "diagnostic_means": means,
            "chosen_excitation": chosen,
            "physics_substeps": env.physics_total,
            "preparation_substeps": env.preparation_total,
        }
        write_json(folder / "result.json", result)
        return result

    screen = {}
    try:
        baseline = execute("B0", baselines)
        profile = execute("P0", lambda: profile_updates(out / "P0", protocol))
        if profile["passed"]:
            for arm in ARMS:
                screen[arm] = execute(
                    f"S-{arm}",
                    lambda a=arm: train_run(
                        out / f"S-{a}",
                        protocol,
                        a,
                        149,
                        SPLITS["screen"],
                        provenance,
                        budget,
                    ),
                )
            if len({r["initial_actor_hash"] for r in screen.values()}) != 1:
                raise RuntimeError("Unmatched initial screen actors")
            selection = select_arm(screen)
        else:
            selection = {
                "qualified": False,
                "arm": None,
                "reason": "profile_gate_failed",
            }
        write_json(out / "selection.json", selection)
        if selection["qualified"]:
            winner = selection["arm"]
            env = StanceEnv(training=False)
            evaluate(
                ConstantPolicy(baseline["chosen_excitation"]),
                env,
                SPLITS["confirmation"],
                "confirmation-constant",
                out / "B0",
            )
            write_json(
                out / "B0/confirmation-baseline-result.json",
                {
                    "physics_substeps": env.physics_total,
                    "preparation_substeps": env.preparation_total,
                    "chosen_excitation": baseline["chosen_excitation"],
                },
            )
            completed_physics += env.physics_total + env.preparation_total
            for seed in (263, 431):
                reference = execute(
                    f"R-A-{seed}",
                    lambda s=seed: train_run(
                        out / f"R-A-{s}",
                        protocol,
                        "A",
                        s,
                        SPLITS["confirmation"],
                        provenance,
                        budget,
                    ),
                )
                if winner != "A":
                    selected = execute(
                        f"R-W-{seed}",
                        lambda s=seed: train_run(
                            out / f"R-W-{s}",
                            protocol,
                            winner,
                            s,
                            SPLITS["confirmation"],
                            provenance,
                            budget,
                        ),
                    )
                    if (
                        reference["initial_actor_hash"]
                        != selected["initial_actor_hash"]
                    ):
                        raise RuntimeError("Unmatched confirmation actors")
        execute("X0", lambda: export_legacy_actions(out / "X0"))
        reason = selection["reason"]
        for key, row in registry.items():
            if row["status"] == "not_run":
                event(
                    key,
                    "not_run",
                    reason="duplicate_reference_slot"
                    if selection["qualified"] and selection["arm"] == "A"
                    else reason,
                )
        result = {
            "selection": selection,
            "registry": registry,
            "wall_s": time.perf_counter() - started,
            "accounted_physics_substeps": completed_physics,
            "source_unchanged": source_hash() == provenance["source_hash"],
            "scope": "motor objective calibration, no boxing or self-play",
        }
        write_json(out / "result.json", result)
        return result
    except Exception as error:
        for key, row in registry.items():
            if row["status"] == "not_run":
                event(key, "not_run", reason="halted_after_fault")
        write_json(
            out / "fault.json",
            {
                "error": str(error),
                "registry": registry,
                "wall_s": time.perf_counter() - started,
            },
        )
        raise
