"""Small native PPO proof runner; no competition, league promotion or default use."""

import gc
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .native_pair_motor_rl import NativePairMotorEnv, PairMotorConfig
from .native_pair_motor_vec import NativePairMotorVecEnv
from .native_tape_storage import store
from .provenance import digest
from .wave import canonical_actor_hash


def tensor_hash(items):
    h = hashlib.sha256()
    for name, t in sorted(items):
        a = t.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name, a.shape, a.dtype.str]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def actor_hash(policy):
    values = (
        [("mean." + k, v) for k, v in policy.action_net.state_dict().items()]
        + [
            ("pi." + k, v)
            for k, v in policy.mlp_extractor.policy_net.state_dict().items()
        ]
        + [("log_std", policy.log_std)]
    )
    return tensor_hash(values)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


class AuditedPPO(PPO):
    def train(self):
        before = actor_hash(self.policy)
        number = len(self.native_update_rows)
        self.save(self.native_output / f"update-{number:03d}-before")
        torch.save(
            {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                "world": self.env.world.snapshot(),
                "vec_next_seed": self.env.next_seed,
                "episode_returns": self.env.episode_returns,
                "episode_lengths": self.env.episode_lengths,
                "last_obs": self._last_obs,
                "last_episode_starts": self._last_episode_starts,
            },
            self.native_output / f"update-{number:03d}-continuation.pt",
        )
        append(
            self.native_output / "updates.jsonl",
            {
                "phase": "entered",
                "number": number,
                "actor_hash": before,
                "num_timesteps": self.num_timesteps,
            },
        )
        super().train()
        after = actor_hash(self.policy)
        self.native_update_rows.append(
            {
                "number": number,
                "actor_before": before,
                "actor_after": after,
                "optimizer_epochs": self._n_updates,
                "actor_samples": self.num_timesteps,
            }
        )
        self.save(self.native_output / f"update-{number:03d}-after")
        append(
            self.native_output / "updates.jsonl",
            {"phase": "completed", **self.native_update_rows[-1]},
        )


class NativeTransitions(BaseCallback):
    def __init__(self, vec, output, guard):
        super().__init__()
        self.vec, self.output, self.guard = vec, Path(output), guard
        self.rows = []
        self.chain = "0" * 64
        self.arrays = []

    def _on_step(self):
        self.guard(0)
        t = self.vec.last_transition
        row = {
            "joint_step": len(self.rows),
            "episode_id": t["info"]["episode_id"],
            "episode_serial": t["info"]["episode_serial"],
            "end_tick": t["info"]["tick"],
            "physics_total": t["info"]["physics_total"],
            "rewards": t["rewards"],
            "terminated": bool(t["terminated"]),
            "truncated": bool(t["truncated"]),
            "candidate_points": t["info"]["candidate_points"],
            "motor_failures": t["info"]["motor_failures"],
            "source_obs_sha256": {
                p: hashlib.sha256(t["source_observations"][p].tobytes()).hexdigest()
                for p in "AB"
            },
            "next_obs_sha256": {
                p: hashlib.sha256(t["next_observations"][p].tobytes()).hexdigest()
                for p in "AB"
            },
            "applied_actions_sha256": {
                p: hashlib.sha256(t["actions"][p].tobytes()).hexdigest() for p in "AB"
            },
            "commands_sha256": {
                p: hashlib.sha256(t["commands"][p].tobytes()).hexdigest() for p in "AB"
            },
            "engine_before_sha256": hashlib.sha256(
                t["engine_before"].tobytes()
            ).hexdigest(),
            "engine_after_sha256": hashlib.sha256(
                t["engine_after"].tobytes()
            ).hexdigest(),
        }
        self.chain = digest({"previous": self.chain, "row": row})
        row["chain"] = self.chain
        self.rows.append(row)
        append(self.output / "transitions.jsonl", row)
        self.arrays.append(
            {
                "teacher_inputs": np.stack([t["teacher_inputs"][p] for p in "AB"]),
                "teacher_commands": np.stack([t["teacher_commands"][p] for p in "AB"]),
                "source_obs": np.stack([t["source_observations"][p] for p in "AB"]),
                "next_obs": np.stack([t["next_observations"][p] for p in "AB"]),
                "applied_actions": np.stack([t["actions"][p] for p in "AB"]),
                "raw_policy_actions": np.asarray(self.locals["actions"]).copy(),
                "log_probs": self.locals["log_probs"].detach().cpu().numpy().copy(),
                "values": self.locals["values"].detach().cpu().numpy().copy(),
                "commands": np.stack([t["commands"][p] for p in "AB"]),
                "engine_before": t["engine_before"],
                "engine_after": t["engine_after"],
                "rewards": np.array(list(t["rewards"].values()), np.float64),
                "terminated": np.array(t["terminated"]),
                "truncated": np.array(t["truncated"]),
            }
        )
        return True

    def save_arrays(self):
        arrays = {k: np.stack([r[k] for r in self.arrays]) for k in self.arrays[0]}
        return store(self.output / "native-transitions.npz", arrays)


def prove_native_replay(world, *, seed, output):
    world.reset(seed=seed)
    zero = {p: np.zeros(138, np.float32) for p in "AB"}
    for _ in range(10):
        world.step(zero)
    world.begin_control(zero)
    for _ in range(3):
        world.advance_substep()
    snapshot = world.snapshot()
    calls = world.decoder.decisions
    expected = []

    def observation(contacts):
        raw = [
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in asdict(c).items()
            }
            for c in contacts
        ]
        return {
            "tick": world.core.tick,
            "engine": hashlib.sha256(world.core.state().tobytes()).hexdigest(),
            "contacts": digest(raw),
            "commands": {
                p: hashlib.sha256(c.tobytes()).hexdigest()
                for p, c in world.core.commands.items()
            },
        }

    for _ in range(17):
        expected.append(observation(world.advance_substep()))
    world.restore(snapshot)
    actual = [observation(world.advance_substep()) for _ in range(17)]
    if expected != actual or world.decoder.decisions != calls:
        raise ValueError("Actual pretrained native midhold replay/newdecision fault")
    proof = {
        "passed": True,
        "sample_tick": 203,
        "remaining": 17,
        "actor_decisions_unchanged": calls,
        "source": expected,
        "replayed": actual,
        "physics": 237,
        "actual_checkpoint": world.bundle,
        "snapshot_sha256": snapshot["sha256"],
    }
    Path(output, "native-replay.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def evaluate(world, policy, *, seed, output, guard):
    obs, _ = world.reset(seed=seed)
    snap = world.snapshot()
    frames = []
    rows = []

    def frame():
        d = world.core.data
        frames.append(
            {
                "qpos": d.qpos.copy(),
                "qvel": d.qvel.copy(),
                "activation": d.act.copy(),
                "control": d.ctrl.copy(),
                "time": np.array(float(d.time)),
            }
        )

    frame()
    while not world.done:
        actions = {}
        for p in "AB":
            if policy is None:
                actions[p] = np.zeros(138, np.float32)
            else:
                actions[p] = policy.predict(obs[p], deterministic=True)[0]
        obs, rewards, terminated, truncated, info = world.step(actions)
        guard(0)
        rows.append(
            {
                "end_tick": world.core.tick,
                "rewards": rewards,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "metrics": info["motor_metrics"],
                "candidate_points": info["candidate_points"],
                "failures": info["motor_failures"],
            }
        )
        frame()
    receipt = store(
        Path(output) / "frames.npz",
        {k: np.stack([r[k] for r in frames]) for k in frames[0]},
    )
    summary = {
        "seed": seed,
        "frames": len(frames),
        "physics": world.core.physics_total,
        "final_tick": world.core.tick,
        "candidate_points": world.referee.points,
        "motor_failures": world.failures,
        "returns": world.returns,
        "final_metrics": world.last_metrics,
        "sporting_qualified": False,
        "frame_tape": receipt,
    }
    Path(output, "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    Path(output, "transitions.json").write_text(json.dumps(rows, sort_keys=True) + "\n")
    Path(output, "initial-snapshot.json").write_text(
        json.dumps(snap, sort_keys=True) + "\n"
    )
    return summary


def run_native_smoke(
    inputs, output, *, seed=71000, joint_steps=32, horizon_ticks=1000, guard=None
):
    if joint_steps not in (8, 16, 32, 64) or type(seed) is not int or seed < 0:
        raise ValueError("Explicit small native proof budget required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    owned_guard = guard or (lambda n: None)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    world = NativePairMotorEnv.from_inputs(
        Path(inputs),
        config=PairMotorConfig(horizon_ticks=horizon_ticks),
        guard=owned_guard,
    )
    teacher_before = canonical_actor_hash(world.teacher.model.actor)
    result = {
        "schema": "joxing-native-pair-ppo-smoke-v1",
        "seed": seed,
        "joint_steps_registered": joint_steps,
        "status": "running",
        "native_physics": 0,
        "competitive_learning": False,
        "teacher_hash": teacher_before,
        "rows": [
            {"id": "before-eval", "status": "not_run"},
            {"id": "native-ppo", "status": "not_run"},
            {"id": "after-eval", "status": "not_run"},
        ],
    }
    try:
        result["native_replay"] = prove_native_replay(world, seed=seed, output=output)
        result["rows"][0]["status"] = "unconfirmed"
        before = evaluate(
            world, None, seed=seed, output=output / "before-eval", guard=owned_guard
        )
        result["before_eval"] = before
        result["rows"][0]["status"] = "completed"
        vec = NativePairMotorVecEnv(world, seed=seed)
        model = AuditedPPO(
            "MlpPolicy",
            vec,
            n_steps=8,
            batch_size=16,
            n_epochs=2,
            learning_rate=3e-4,
            gamma=0.99,
            seed=seed,
            device="cpu",
            policy_kwargs={
                "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                "log_std_init": -3.0,
                "ortho_init": False,
            },
            verbose=0,
        )
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
        model.native_update_rows = []
        model.native_output = output / "training"
        model.native_output.mkdir()
        model.save(model.native_output / "initial-policy")
        actor_before = actor_hash(model.policy)
        callback = NativeTransitions(vec, model.native_output, owned_guard)
        result["rows"][1]["status"] = "unconfirmed"
        model.learn(total_timesteps=2 * joint_steps, callback=callback)
        model.save(model.native_output / "final-policy")
        tape = callback.save_arrays()
        actor_after = actor_hash(model.policy)
        if (
            len(callback.rows) != joint_steps
            or model.num_timesteps != 2 * joint_steps
            or not model.native_update_rows
            or actor_before == actor_after
        ):
            raise ValueError("Native transition/update/changed actor proof missing")
        result["training"] = {
            "joint_transitions": len(callback.rows),
            "actor_samples": model.num_timesteps,
            "rollout_updates": len(model.native_update_rows),
            "optimizer_epochs": model._n_updates,
            "actor_before": actor_before,
            "actor_after": actor_after,
            "chain": callback.chain,
            "updates": model.native_update_rows,
            "transition_tape": tape,
            "source": "actual NativePairMotorEnv two1400muscle world; verified pretrained teacher",
        }
        result["rows"][1]["status"] = "completed"
        result["rows"][2]["status"] = "unconfirmed"
        result["after_eval"] = evaluate(
            world, model, seed=seed, output=output / "after-eval", guard=owned_guard
        )
        result["rows"][2]["status"] = "completed"
        teacher_after = canonical_actor_hash(world.teacher.model.actor)
        if teacher_after != teacher_before:
            raise ValueError("Pretrained teacher changed")
        result.update(
            status="completed",
            teacher_after=teacher_after,
            world_contract=world.contract,
            world_contract_hash=world.contract_hash,
        )
    except BaseException as error:
        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        for r in result["rows"]:
            if r["status"] == "unconfirmed":
                r["status"] = "failed"
        raise
    finally:
        result.update(
            native_physics=world.physics_total, wall_s=time.monotonic() - started
        )
        Path(output, "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        world.close()
        gc.collect()
    return result
