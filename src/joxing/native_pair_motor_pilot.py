"""Opt-in bounded H31 motor pilot, not sporting self-play or promotion."""

import copy
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .native_pair_motor_evidence import CausalChunkWriter
from .native_pair_motor_rl import NativePairMotorEnv, PairMotorConfig
from .native_pair_motor_train import (
    StreamingNativeTransitions,
    actor_hash,
    prove_native_replay,
)
from .native_pair_motor_vec import NativePairMotorVecEnv
from .native_tape_storage import sha_file, store
from .wave import canonical_actor_hash


def write_binary(budget, path, blob, maximum):
    if len(blob) > maximum:
        raise ValueError("Native pilot bounded serialization first limit")
    path = budget.path(path)
    budget.check(len(blob))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(blob)
    return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


class LimitedBuffer(io.BytesIO):
    def __init__(self, maximum):
        super().__init__()
        self.maximum = maximum

    def write(self, data):
        if self.tell() + len(data) > self.maximum:
            raise ValueError("Native pilot in-memory serialization bound")
        return super().write(data)


def continuation(model):
    vec = model.env
    if vec.pending is not None or vec.world.core.remaining:
        raise ValueError("Pilot checkpoint must be a completed joint control")
    return {
        "schema": "joxing-native-pilot-continuation-v1",
        "contract_hash": vec.world.contract_hash,
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "world": vec.world.snapshot(),
        "world_episode_serial": vec.world.episode_serial,
        "world_completed_episodes": vec.world.completed_episodes,
        "vec": {
            n: copy.deepcopy(getattr(vec, n))
            for n in (
                "next_seed",
                "episode_returns",
                "episode_lengths",
                "reset_infos",
                "_seeds",
                "_options",
            )
        },
        "last_obs": model._last_obs.copy(),
        "last_episode_starts": model._last_episode_starts.copy(),
        "num_timesteps": model.num_timesteps,
    }


class CheckpointBook:
    def __init__(self, output, budget):
        self.output, self.budget = Path(output), budget
        self.output.mkdir(parents=True, exist_ok=False)
        self.checkpoints, self.updates = [], []

    def save(self, model):
        value = continuation(model)
        policy = LimitedBuffer(13 * 1024**2)
        model.save(policy)
        state = LimitedBuffer(1024**2)
        torch.save(value, state)
        a, b = policy.getvalue(), state.getvalue()
        number = len(self.checkpoints)
        row = {
            "number": number,
            "actor_hash": actor_hash(model.policy),
            "actor_samples": model.num_timesteps,
            "world_snapshot_sha256": value["world"]["sha256"],
            "policy": {"bytes": len(a), "sha256": hashlib.sha256(a).hexdigest()},
            "continuation": {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()},
        }
        # Reserve both binary files and their receipt before publishing any.
        self.budget.check(
            len(a) + len(b) + len(json.dumps(row)) + 4096, len(json.dumps(row)) + 4096
        )
        write_binary(
            self.budget, self.output / f"boundary-{number:03d}.zip", a, 13 * 1024**2
        )
        write_binary(self.budget, self.output / f"boundary-{number:03d}.pt", b, 1024**2)
        self.budget.write_json(self.output / f"boundary-{number:03d}.json", row)
        self.checkpoints.append(row)
        return row


def restore_checkpoint(folder, row, vec):
    folder = Path(folder)
    n = row["number"]
    for suffix, key in (("zip", "policy"), ("pt", "continuation")):
        path = folder / f"boundary-{n:03d}.{suffix}"
        if (
            path.stat().st_size != row[key]["bytes"]
            or sha_file(path) != row[key]["sha256"]
        ):
            raise ValueError("Pilot checkpoint external identity")
    # Trusted local provenance was authenticated above; never load an arbitrary
    # untrusted pickle. Exact runtime profile is verified by the task owner.
    value = torch.load(
        folder / f"boundary-{n:03d}.pt", map_location="cpu", weights_only=False
    )
    if (
        value["schema"] != "joxing-native-pilot-continuation-v1"
        or value["contract_hash"] != vec.world.contract_hash
    ):
        raise ValueError("Foreign pilot continuation")
    if value["world"]["sha256"] != row["world_snapshot_sha256"]:
        raise ValueError("Pilot world snapshot identity")
    vec.world.reset(seed=value["world"]["episode_seed"])
    vec.world.restore(value["world"])
    vec.world.episode_serial = value["world_episode_serial"]
    vec.world.completed_episodes = value["world_completed_episodes"]
    for name, item in value["vec"].items():
        if name not in (
            "next_seed",
            "episode_returns",
            "episode_lengths",
            "reset_infos",
            "_seeds",
            "_options",
        ):
            raise ValueError("Foreign VecEnv continuation field")
        setattr(vec, name, copy.deepcopy(item))
    model = PilotAuditedPPO.load(
        folder / f"boundary-{n:03d}.zip", env=vec, device="cpu", force_reset=False
    )
    model._last_obs = value["last_obs"].copy()
    model._last_episode_starts = value["last_episode_starts"].copy()
    if (
        model.num_timesteps != row["actor_samples"]
        or actor_hash(model.policy) != row["actor_hash"]
    ):
        raise ValueError("Pilot actor/counter continuation")
    actual = np.stack([vec.world.observations()[p] for p in "AB"])
    if actual.tobytes() != model._last_obs.tobytes():
        raise ValueError("Pilot VecEnv/world cached observation mismatch")
    # Reset/construct/load may consume RNG. Restore all streams LAST.
    torch.set_rng_state(value["torch"])
    np.random.set_state(value["numpy"])
    random.setstate(value["python"])
    return model


def optimizer_steps(model):
    return max(
        (
            int(s["step"].item())
            for s in model.policy.optimizer.state.values()
            if "step" in s
        ),
        default=0,
    )


class PilotAuditedPPO(PPO):
    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "pilot"]

    def train(self):
        before, steps = actor_hash(self.policy), optimizer_steps(self)
        source = self.pilot.checkpoints[-1]
        super().train()
        target = self.pilot.save(self)
        row = {
            "number": len(self.pilot.updates),
            "behavior_boundary": source["number"],
            "actor_before": before,
            "actor_after": target["actor_hash"],
            "actor_samples": self.num_timesteps,
            "optimizer_epochs": self._n_updates,
            "optimizer_minibatch_steps": optimizer_steps(self) - steps,
        }
        if before != source["actor_hash"] or row["optimizer_minibatch_steps"] <= 0:
            raise ValueError("Pilot behavior/optimizer provenance")
        self.pilot.budget.write_json(
            self.pilot.output / f"update-{row['number']:03d}.json", row
        )
        self.pilot.updates.append(row)


class PilotTransitions(StreamingNativeTransitions):
    def _on_training_start(self):
        self.model.pilot.save(self.model)

    def _on_step(self):
        if self.vec.world.contract_hash != self.writer.contract_hash:
            raise ValueError("Pilot writer/world identity")
        transition = self.vec.last_transition
        boundary = self.model.pilot.checkpoints[-1]
        transition["info"]["behavior"] = {
            "boundary": boundary["number"],
            "policy_sha256": boundary["policy"]["sha256"],
            "actor_hash": boundary["actor_hash"],
        }
        return super()._on_step()


class MotorMetrics:
    """Measure continuous1ms target persistence, never infer it from endpoints."""

    def __init__(self):
        self.rows = []
        self.holds = {p: [0, 0, 0] for p in "AB"}
        self.cycles = {p: 0 for p in "AB"}
        self.expected = {p: 0 for p in "AB"}
        self.previous_phase = {p: None for p in "AB"}

    def observe(self, world):
        row = []
        for p in "AB":
            m = world.last_metrics[p]
            phase = m["phase"]
            if phase == 0 and self.previous_phase[p] == 2:
                self.expected[p] = 0
            self.previous_phase[p] = phase
            distances = m["hand_target_distance_m"]
            good = max(distances) <= 0.15 and m["face_cos_positive"] >= 0.5
            self.holds[p][phase] = self.holds[p][phase] + 1 if good else 0
            for k in range(3):
                if k != phase:
                    self.holds[p][k] = 0
            if self.holds[p][phase] >= 40 and phase == self.expected[p]:
                self.expected[p] += 1
                if self.expected[p] == 3:
                    self.cycles[p] += 1
                    self.expected[p] = 0
            # Require a fresh guard phase after each return; phase order is fixed.
            row.append(
                [
                    *distances,
                    m["face_cos_positive"],
                    float(phase),
                    m["tilt_rad"],
                    m["drop_m"],
                    m["up_world_z"],
                ]
            )
        self.rows.append(row)

    def summary(self, world):
        a = np.array(self.rows, np.float64)
        return {
            p: {
                "mean_target_error_m": float(a[:, j, :2].mean()),
                "reach_return_cycles": self.cycles[p],
                "survival_s": (
                    world.failures[p]["tick"] if world.failures[p] else world.core.tick
                )
                * 0.001,
                "completed_3s": world.failures[p] is None and world.core.tick == 3000,
            }
            for j, p in enumerate("AB")
        }


def evaluate_pilot(world, policy, *, seed, output, budget, guard):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    obs, _ = world.reset(seed=seed)
    snapshot = world.snapshot()
    budget.write_json(output / "initial-snapshot.json", snapshot)
    frames, commands, actions, hashes = [], [], [], []
    metrics = MotorMetrics()

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
        pair = {
            p: np.zeros(138, np.float32)
            if policy is None
            else policy.predict(obs[p], deterministic=True)[0]
            for p in "AB"
        }
        actions.append(np.stack([pair[p] for p in "AB"]))
        world.begin_control(pair)
        commands.append(np.stack([world.core.commands[p] for p in "AB"]))
        while world.core.remaining:
            world.advance_substep()
            metrics.observe(world)
        obs = world.observations()
        hashes.append(hashlib.sha256(world.core.state().tobytes()).digest())
        frame()
        guard(0)
    arrays = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    trace = {
        "actions": np.stack(actions),
        "commands": np.stack(commands),
        "engine_sha256": np.array([np.frombuffer(h, np.uint8) for h in hashes]),
        "metrics_1ms": np.array(metrics.rows, np.float64),
    }
    from .native_tape_storage import plan

    budget.check(plan(arrays)["archive_bytes"] + plan(trace)["archive_bytes"])
    frame_receipt = store(output / "frames.npz", arrays)
    trace_receipt = store(output / "evaluation-trace.npz", trace)
    result = {
        "seed": seed,
        "frames": len(frames),
        "final_tick": world.core.tick,
        "candidate_points": world.referee.points,
        "motor_failures": world.failures,
        "returns": world.returns,
        "metrics": metrics.summary(world),
        "frame_tape": frame_receipt,
        "trace_tape": trace_receipt,
        "initial_snapshot_sha256": snapshot["sha256"],
        "sporting_qualified": False,
    }
    budget.write_json(output / "evaluation.json", result)
    return result


def run_native_pilot(
    inputs,
    output,
    *,
    budget,
    guard,
    seeds=(761, 983),
    joint_steps=1024,
    evaluation_seeds=(71000, 71016, 71032, 71048),
):
    if (
        tuple(seeds) != (761, 983)
        or joint_steps != 1024
        or tuple(evaluation_seeds) != (71000, 71016, 71032, 71048)
    ):
        raise ValueError("Explicit bounded H31pilot profile required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    world = NativePairMotorEnv.from_inputs(
        Path(inputs), config=PairMotorConfig(horizon_ticks=3000), guard=guard
    )
    teacher = canonical_actor_hash(world.teacher.model.actor)
    result = {
        "schema": "joxing-native-pair-motor-pilot-v1",
        "status": "running",
        "contract": world.contract,
        "contract_hash": world.contract_hash,
        "teacher": teacher,
        "seed_rows": [
            {
                "seed": s,
                "status": "not_run",
                "training_status": "not_run",
                "evaluations": {
                    phase: [{"seed": e, "status": "not_run"} for e in evaluation_seeds]
                    for phase in ("before", "after")
                },
            }
            for s in seeds
        ],
        "competitive_learning": False,
    }
    try:
        # Existing proof writes bounded14KiB; reserve/check typed allowance first.
        budget.check(32768, 32768)
        result["native_replay"] = prove_native_replay(world, seed=71000, output=output)
        for row in result["seed_rows"]:
            seed = row["seed"]
            row["status"] = "unconfirmed"
            folder = output / f"seed-{seed}"
            folder.mkdir()
            for e in row["evaluations"]["before"]:
                e["status"] = "unconfirmed"
                e["result"] = evaluate_pilot(
                    world,
                    None,
                    seed=e["seed"],
                    output=folder / f"before-{e['seed']}",
                    budget=budget,
                    guard=guard,
                )
                e["status"] = "completed"
            vec = NativePairMotorVecEnv(world, seed=seed, record_history=True)
            model = PilotAuditedPPO(
                "MlpPolicy",
                vec,
                n_steps=128,
                batch_size=64,
                n_epochs=4,
                learning_rate=3e-4,
                gamma=0.99,
                seed=seed,
                device="cpu",
                verbose=0,
                policy_kwargs={
                    "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                    "log_std_init": -3.0,
                    "ortho_init": False,
                },
            )
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
            model.pilot = CheckpointBook(folder / "checkpoints", budget)
            writer = CausalChunkWriter(
                folder / "training",
                contract_hash=world.contract_hash,
                state_dimension=len(world.core.state()),
                budget=budget,
            )
            before_physics, episodes = world.physics_total, world.completed_episodes
            row["training_status"] = "unconfirmed"
            model.learn(
                total_timesteps=2 * joint_steps,
                callback=PilotTransitions(vec, writer, guard),
            )
            row["training"] = writer.close()
            row["training"].update(
                actor_samples=model.num_timesteps,
                optimizer_epochs=model._n_updates,
                updates=model.pilot.updates,
                checkpoints=model.pilot.checkpoints,
                physics=world.physics_total - before_physics,
                completed_motor_episodes=world.completed_episodes - episodes,
            )
            if (
                len(model.pilot.updates) != 8
                or model.num_timesteps != 2048
                or row["training"]["joint_transitions"] != 1024
            ):
                raise ValueError("Pilot rollout/update denominator")
            row["training_status"] = "completed"
            for e in row["evaluations"]["after"]:
                e["status"] = "unconfirmed"
                e["result"] = evaluate_pilot(
                    world,
                    model,
                    seed=e["seed"],
                    output=folder / f"after-{e['seed']}",
                    budget=budget,
                    guard=guard,
                )
                e["status"] = "completed"
            row["status"] = "completed"
            budget.write_json(folder / "seed-result.json", row)
            del model, vec
        if canonical_actor_hash(world.teacher.model.actor) != teacher:
            raise ValueError("Frozen released teacher changed")
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        for row in result["seed_rows"]:
            if row["status"] == "unconfirmed":
                row["status"] = "failed"
            if row["training_status"] == "unconfirmed":
                row["training_status"] = "failed"
            for phase in row["evaluations"].values():
                for e in phase:
                    if e["status"] == "unconfirmed":
                        e["status"] = "failed"
        raise
    finally:
        result["native_physics"] = world.physics_total
        # Pre-reserved fault publication only, no extra execution under reserve.
        reserve = budget.reserve
        budget.reserve = 0
        try:
            budget.write_json(output / "result.json", result)
        finally:
            budget.reserve = reserve
        world.close()
    return result
