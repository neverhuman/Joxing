"""Exclusive bounded D2 parent; no automatic retry or held-out selection."""

import gc
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .motor_evaluation import collect, replay
from .motor_pilot import (
    TRAINING_SEEDS,
    PilotWorker,
    make_learner,
    paired_conclusion,
    planned_rows,
)
from .motor_protocol import optimizer_hashes, pid_identity
from .motor_residual import MotorResidualEnv
from .motor_storage import RawCredit, append_json, save_npz, write_bytes, write_json
from .provenance import digest, manifest
from .wave import canonical_actor_hash, check_resources

MIB = 1024**2


class ShapeOnly(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3601,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (138,), np.float32)

    def reset(self, **kwargs):
        raise RuntimeError("Shape-only slot must never supply rollout data")

    def step(self, action):
        raise RuntimeError("Shape-only slot must never supply rollout data")


class MeasuredSAC(SAC):
    def train(self, gradient_steps, batch_size=64):
        began = time.monotonic()
        before = self._n_updates
        super().train(gradient_steps, batch_size)
        losses = {
            k: float(v)
            for k, v in self.logger.name_to_value.items()
            if k in {"train/actor_loss", "train/critic_loss", "train/ent_coef"}
        }
        if not np.isfinite(list(losses.values())).all():
            raise FloatingPointError("Nonfinite SAC loss")
        if not hasattr(self, "update_receipts"):
            self.update_receipts = []
        self.update_receipts.append(
            {
                "transitions": self.num_timesteps,
                "before": before,
                "after": self._n_updates,
                "wall_s": time.monotonic() - began,
                "losses": losses,
            }
        )


def close_pool(vec):
    """Bounded cleanup even when one child has died inside step_wait."""
    for remote, proc in zip(vec.remotes, vec.processes, strict=False):
        if proc.is_alive():
            try:
                remote.send(("close", None))
            except (OSError, EOFError):
                pass
    deadline = time.monotonic() + 8
    forced = []
    for proc in vec.processes:
        proc.join(max(0, deadline - time.monotonic()))
    for proc in vec.processes:
        if proc.is_alive():
            forced.append(proc.pid)
            proc.terminate()
    for proc in vec.processes:
        proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
    for remote in vec.remotes:
        remote.close()
    vec.closed = True
    return forced


def worker_factory(inputs, folder, index, worker, deadline, expected_bundle, raw_bytes):
    def make():
        import torch

        torch.set_num_threads(1)
        env = MotorResidualEnv.from_inputs(inputs)
        if env.contract_hash != expected_bundle:
            env.close()
            raise ValueError("Worker bundle drift")
        return PilotWorker(env, folder, index, worker, deadline, raw_bytes)

    return make


def load_motor_env(inputs, factory=MotorResidualEnv.from_inputs):
    """Loading a frozen SAC teacher must not reseed the parent's learner RNG."""
    import torch

    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        return factory(inputs)
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def save_policy(model, folder, stage, bundle):
    import torch

    buffer = io.BytesIO()
    model.save(buffer)
    payload = buffer.getvalue()
    write_bytes(folder / (stage + ".zip"), payload)
    metadata = {
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "optimizer_state_hashes": optimizer_hashes(model),
        "actor_hash": canonical_actor_hash(model.actor),
        "critic_hash": canonical_actor_hash(model.critic),
        "target_hash": canonical_actor_hash(model.critic_target),
        "bundle_hash": bundle,
        "transitions": model.num_timesteps,
        "optimizer_updates": model._n_updates,
        "acting_schedule": "training_batch4_eval_scalar1_v1",
        "numpy_rng_name": np.random.get_state()[0],
        "numpy_rng_position": np.random.get_state()[2],
        "python_rng": random.getstate(),
    }
    write_json(folder / (stage + "-bundle.json"), metadata)
    state = np.random.get_state()
    save_npz(
        folder / (stage + "-rng.npz"),
        numpy_state=state[1],
        numpy_gaussian=np.asarray(state[3:]),
        torch_state=torch.get_rng_state().numpy(),
    )
    return metadata


class ProgressCallback(BaseCallback):
    def __init__(self, parent, row, folder):
        super().__init__()
        self.parent, self.row, self.folder = parent, row, folder

    def _on_step(self):
        self.parent.guard(0)
        if self.num_timesteps % 64 == 0:
            self.row.update(
                transitions=self.num_timesteps,
                optimizer_updates=self.model._n_updates,
                confirmed_worker_progress=read_workers(self.folder),
            )
            self.parent.save("training_progress")
        return True


def read_workers(folder):
    result = []
    for worker in range(4):
        path = folder / (f"worker-{worker}/progress.json")
        result.append(
            json.loads(path.read_text())
            if path.exists()
            else {
                "worker_index": worker,
                "status": "unconfirmed",
                "physics_steps": 0,
                "maximum_physics_allocation": 20480,
            }
        )
    return result


class PilotRun:
    def __init__(self, protocol, inputs, out):
        self.p, self.inputs, self.out = protocol, Path(inputs), Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.credit = RawCredit(
            self.out, protocol["raw_credits"]["parent_bytes"], parent=True
        )
        self.began = time.monotonic()
        self.audit_wall = self.audit_physics = 0
        self.audit_began = None
        self.main_physics = protocol["prior"]["physics_steps"]
        self.active_env = None
        self.training_allocation = 0
        self.training_folder = None
        self.report = manifest() | {
            "pid": os.getpid(),
            "process_identity": pid_identity(),
            "status": "running",
            "protocol": protocol,
            "evaluations": planned_rows(),
            "training": [
                {"training_seed": s, "status": "not_run"} for s in TRAINING_SEEDS
            ],
            "replays": [
                {"training_seed": s, "evaluation_seed": 83000, "status": "not_run"}
                for s in TRAINING_SEEDS
            ],
            "boxing_games": 0,
            "formal_optimizer_updates": 0,
            "physics_accounting_exact": True,
            "protocol_sha256": digest(protocol),
        }
        write_json(self.out / "protocol.json", protocol)
        self.save("created")

    def main_wall(self):
        audit = self.audit_wall + (
            0 if self.audit_began is None else time.monotonic() - self.audit_began
        )
        return self.p["prior"]["formal_wall_s"] + time.monotonic() - self.began - audit

    def physics(self):
        return (
            self.main_physics
            + sum(
                max(
                    0,
                    r.get("physics_steps_upper_bound", 0)
                    - r.get("physics_steps_lower_bound", 0),
                )
                for r in self.report["training"]
            )
            + self.training_allocation
            + (0 if self.active_env is None else self.active_env.task.physics_total)
        )

    def guard(self, next_steps):
        if self.main_wall() >= self.p["caps"]["wall_s"]:
            raise TimeoutError("Whole-version main wall cap")
        if self.physics() + next_steps > self.p["caps"]["physics_steps"]:
            raise TimeoutError("Whole-version main physics cap")
        check_resources(self.out, self.p["caps"])

    def replay_guard(self, next_steps):
        if self.audit_began is None:
            raise RuntimeError("Missing replay reserve ownership")
        if (
            self.audit_wall + time.monotonic() - self.audit_began
            >= self.p["audit_caps"]["wall_s"]
        ):
            raise TimeoutError("Audit wall reserve")
        if (
            self.audit_physics + self.active_env.task.physics_total + next_steps
            > self.p["audit_caps"]["physics_steps"]
        ):
            raise TimeoutError("Audit physics reserve")
        check_resources(self.out, self.p["caps"])

    def confirmed_physics(self):
        workers = (
            0
            if self.training_folder is None
            else sum(r["physics_steps"] for r in read_workers(self.training_folder))
        )
        return (
            self.main_physics
            + workers
            + (0 if self.active_env is None else self.active_env.task.physics_total)
        )

    def save(self, event):
        self.report["formal_optimizer_updates"] = sum(
            r.get("optimizer_updates", 0) for r in self.report["training"]
        )
        self.report.update(
            main_physics_steps_lower_bound=self.confirmed_physics(),
            main_physics_steps_upper_bound=self.physics(),
            main_formal_wall_s=self.main_wall(),
            audit_physics_steps=self.audit_physics,
            audit_wall_s=self.audit_wall,
        )
        write_json(self.out / "result.json", self.report)
        append_json(
            self.out / "events.jsonl",
            {
                "event": event,
                "monotonic_s": time.monotonic(),
                "main_physics_steps_lower_bound": self.confirmed_physics(),
                "optimizer_updates": self.report["formal_optimizer_updates"],
            },
        )

    def evaluation(self, model, index, stage, folder):
        self.guard(0)
        self.active_env = load_motor_env(self.inputs)
        try:
            if self.active_env.contract_hash != self.p["bundle_hash"]:
                raise ValueError("Evaluation bundle drift")
            context = {
                "bundle": self.active_env.contract,
                "groups": [list(g) for g in self.active_env.groups],
                "normalizer": self.active_env.normalizer.export(),
                "physical_specification": self.active_env.task.specification(),
                "floor_geom_id": self.active_env.task.floor,
            }
            context_path = self.out / "frozen-context.json"
            if context_path.exists():
                if digest(context) != digest(json.loads(context_path.read_text())):
                    raise ValueError("Frozen context drift between phases")
            else:
                write_json(context_path, context)
            for i, row in enumerate(self.report["evaluations"]):
                if (
                    row["training_seed"] != TRAINING_SEEDS[index]
                    or row["stage"] != stage
                ):
                    continue
                self.guard(0)
                path = folder / (stage + "-" + str(row["evaluation_seed"]))
                row["status"] = "running"
                self.save("evaluation_start")
                before = self.active_env.task.physics_total
                row.update(physics_steps=0, controls=0)
                try:
                    self.report["evaluations"][i] = collect(
                        self.active_env,
                        model,
                        path,
                        row,
                        self.guard,
                        self.p["raw_credits"]["evaluation_bytes"],
                    )
                except Exception as error:
                    self.report["evaluations"][i] = dict(
                        row,
                        status="incomplete",
                        reason=type(error).__name__ + ": " + str(error),
                        physics_steps=self.active_env.task.physics_total - before,
                        physics_accounting_exact=True,
                    )
                    raise
                self.save("evaluation_completed")
                if self.report["evaluations"][i]["status"] != "completed":
                    raise RuntimeError(
                        "Held-out evaluation halted; retain remaining rows"
                    )
        finally:
            self.main_physics += self.active_env.task.physics_total
            self.active_env.close()
            self.active_env = None

    def training(self, model, index, folder):
        row = self.report["training"][index]
        self.guard(81920)
        reservation = check_resources(self.out, self.p["caps"])
        minimum = self.p["caps"]["minimum_memory_bytes"] + 12 * 1024**3
        if (
            reservation["available_memory_bytes"] is None
            or reservation["available_memory_bytes"] < minimum
        ):
            raise RuntimeError("Four 2GiB workers plus 4GiB learner reservation")
        began = time.monotonic()
        deadline = min(
            began + 600,
            self.began
            + self.p["caps"]["wall_s"]
            - self.p["prior"]["formal_wall_s"]
            + self.audit_wall,
        )
        row.update(
            status="running", allocation_physics=81920, started_monotonic_s=began
        )
        self.training_allocation = 81920
        self.training_folder = folder
        self.save("training_start")
        vec = None
        try:
            vec = SubprocVecEnv.__new__(SubprocVecEnv)
            SubprocVecEnv.__init__(
                vec,
                [
                    worker_factory(
                        self.inputs,
                        folder / f"worker-{w}",
                        index,
                        w,
                        deadline,
                        self.p["bundle_hash"],
                        self.p["raw_credits"]["worker_bytes"],
                    )
                    for w in range(4)
                ],
                start_method="spawn",
            )
            row["worker_pids"] = [p.pid for p in vec.processes]
            row["worker_process_identities"] = [
                pid_identity(p.pid) for p in vec.processes
            ]
            row["startup_s"] = time.monotonic() - began
            self.save("workers_started")
            model.set_env(vec, force_reset=True)
            model.learn(
                total_timesteps=8192, callback=ProgressCallback(self, row, folder)
            )
            assert model.num_timesteps == 8192 and model._n_updates == 448
            if time.monotonic() >= deadline:
                raise TimeoutError("Per-seed wall cap")
            row.update(
                status="completed",
                transitions=model.num_timesteps,
                optimizer_updates=model._n_updates,
                warmup_transitions=1024,
                update_receipts=model.update_receipts,
                optimizer_wall_s=sum(r["wall_s"] for r in model.update_receipts),
            )
        except Exception as error:  # noqa: BLE001 -- all attempts and conservative counters retained
            row.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=type(error).__name__ + ": " + str(error),
                transitions=model.num_timesteps,
                optimizer_updates=model._n_updates,
            )
        finally:
            forced = [] if vec is None else close_pool(vec)
            workers = read_workers(folder)
            exact = (
                not forced
                and len(workers) == 4
                and all(r["status"] in {"closed", "fault"} for r in workers)
            )
            lower = sum(r["physics_steps"] for r in workers)
            upper = lower if exact else 81920
            self.main_physics += lower
            self.training_allocation = 0
            self.training_folder = None
            self.report["physics_accounting_exact"] &= exact
            row.update(
                physics_steps_lower_bound=lower,
                physics_steps_upper_bound=upper,
                workers=workers,
                forced_worker_pids=forced,
                wall_s=time.monotonic() - began,
            )
            if not exact:
                row.update(
                    status="incomplete",
                    reason="Worker counters uncertain; stop, retain allocation upper bound",
                )
            write_json(folder / "training.json", row)
            self.save("training_end")
        if row["status"] != "completed":
            raise RuntimeError("Training attempt halted; no continuation")
        assert lower == 81920

    def run(self):
        import torch

        torch.set_num_threads(1)
        try:
            check_resources(self.out, self.p["caps"], True)
            for index, seed in enumerate(TRAINING_SEEDS):
                folder = self.out / f"seed-{seed}"
                folder.mkdir()
                slots = DummyVecEnv([ShapeOnly] * 4)
                model = make_learner(slots, seed, MeasuredSAC)
                initial = save_policy(model, folder, "initial", self.p["bundle_hash"])
                self.evaluation(model, index, "initial", folder)
                self.training(model, index, folder)
                for module in (model.actor, model.critic, model.critic_target):
                    assert all(torch.isfinite(t).all() for t in module.parameters())
                final = save_policy(model, folder, "final", self.p["bundle_hash"])
                assert initial["actor_hash"] != final["actor_hash"]
                assert initial["critic_hash"] != final["critic_hash"]
                assert initial["target_hash"] != final["target_hash"]
                loaded = MeasuredSAC.load(folder / "final.zip", device="cpu")
                assert canonical_actor_hash(loaded.actor) == final["actor_hash"]
                self.evaluation(loaded, index, "final", folder)
                self.guard(0)
                self.audit_began = time.monotonic()
                self.report["replays"][index]["status"] = "running"
                self.save("replay_start")
                try:
                    self.active_env = load_motor_env(self.inputs)
                    rr = replay(
                        self.active_env,
                        loaded,
                        folder / "final-83000",
                        folder / "replay-83000",
                        self.replay_guard,
                        self.p["raw_credits"]["replay_bytes"],
                    )
                    self.report["replays"][index].update(rr)
                except Exception as error:
                    self.report["replays"][index].update(
                        status="incomplete",
                        reason=type(error).__name__ + ": " + str(error),
                        physics_steps=(
                            0
                            if self.active_env is None
                            else self.active_env.task.physics_total
                        ),
                    )
                    raise
                finally:
                    if self.active_env is not None:
                        self.audit_physics += self.active_env.task.physics_total
                        self.active_env.close()
                    self.audit_wall += time.monotonic() - self.audit_began
                    self.audit_began = None
                    self.active_env = None
                if rr["status"] != "completed":
                    raise RuntimeError("Required frozen-policy replay failed")
                self.save("seed_concluded")
                slots.close()
                del model, loaded
                gc.collect()
            self.guard(0)
            self.report.update(
                status="completed",
                conclusion=paired_conclusion(self.report["evaluations"]),
            )
        except Exception as error:  # noqa: BLE001 -- preserve fault and every future row
            self.credit.emergency_mode = True
            self.report.update(
                status="halted", reason=type(error).__name__ + ": " + str(error)
            )
        finally:
            for group in ("evaluations", "training", "replays"):
                for row in self.report[group]:
                    if row["status"] == "not_run":
                        row["reason"] = "halted_before_attempt"
                    if row["status"] == "running":
                        row.update(
                            status="incomplete", reason="Parent halted during attempt"
                        )
            self.save("finished")
            files = {
                str(path.relative_to(self.out)): {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(self.out.rglob("*"))
                if path.is_file()
                and path.name not in {"raw-manifest.json", "independent-audit.json"}
            }
            write_json(self.out / "raw-manifest.json", {"files": files})
        return self.report
