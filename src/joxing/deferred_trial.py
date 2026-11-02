"""Opt-in evaluation allocation lifecycle; never the default training path."""

import json
import multiprocessing
import resource
import time
from functools import partial

import numpy as np

from .provenance import digest, source_hash, write_json
from .stance_env import StanceEnv
from .stance_train import file_hash
from .wave import (
    ARMS,
    SPLITS,
    available_memory,
    bootstrap_pair,
    canonical_actor_hash,
    check_resources,
    diagnostic,
    evaluate,
    make_worker,
    timed_sac_class,
)


class VectorOwner:
    """Own partially constructed SB3 worlds and retain cleanup errors."""

    def __init__(self):
        self.vector = None
        self.extra = []
        self.closed = False
        self.cleanup_errors = []

    def construct(self, factory, *args, **kwargs):
        prior = {id(p) for p in multiprocessing.active_children()}
        self.vector = factory.__new__(factory)
        try:
            factory.__init__(self.vector, *args, **kwargs)
        except BaseException:
            self.extra = [
                p for p in multiprocessing.active_children() if id(p) not in prior
            ]
            try:
                self.close(partial=True)
            except Exception as cleanup_error:  # noqa: BLE001 -- preserve constructor failure
                self.cleanup_errors.append(str(cleanup_error))
            raise
        return self.vector

    def close(self, partial=False):
        if self.closed or self.vector is None:
            return
        processes = list(getattr(self.vector, "processes", [])) + self.extra
        failure = None
        if not partial:
            try:
                self.vector.close()
            except Exception as error:  # noqa: BLE001 -- still terminate owned processes
                failure = error
        if partial or failure is not None:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
            if process.is_alive():
                raise RuntimeError("Owned vector process survives cleanup")
        for key in ("remotes", "work_remotes"):
            for remote in getattr(self.vector, key, []):
                remote.close()
        self.closed = True
        if failure is not None:
            raise RuntimeError("Vector close failed after owned cleanup") from failure


def train_deferred(
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
    if not (defer_initial_evaluation or ids == SPLITS["confirmation"]):
        raise ValueError("Deferred allocation requires deferred initial evaluation")
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    out.mkdir(parents=True, exist_ok=False)
    setting = ARMS[arm]
    cfg = protocol["learner"]
    write_json(
        out / "description.json",
        {
            "allocation_lifecycle": "deferred-evaluation-v1",
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
    owner = VectorOwner()
    vec = model = env = None
    captured = None
    failure = None
    phase = "vector_construction"
    try:
        vec = owner.construct(
            SubprocVecEnv,
            [
                partial(make_worker, out, i, seed, setting["reward_scale"])
                for i in range(4)
            ],
            start_method="spawn",
        )
        specifications = vec.env_method("specification")
        if len(specifications) != 4 or any(
            s != specifications[0] for s in specifications
        ):
            raise ValueError("Four-world specification mismatch")
        specification = specifications[0]
        write_json(out / "specification.json", specification)
        phase = "sac_construction"
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

        phase = "training"
        initial_hash = canonical_actor_hash(model.actor)
        model.actor.save(out / "initial-actor.pt")
        callback = Progress()
        model.learn(total_timesteps=cfg["steps"], callback=callback)
        counts = counters()
        final_hash = canonical_actor_hash(model.actor)
        model.actor.save(out / "final-actor.pt")
        captured = vec.env_method("snapshot")
        if len(captured) != 4:
            raise ValueError("Missing four training-end states")
        phase = "training_end_captured"
        owner.close()
        write_json(
            out / "training-end-workers.json",
            {"origin": "training-end", "states": captured},
        )
        phase = "evaluation_construction"
        env = StanceEnv(training=False)
        if env.specification() != specification:
            raise ValueError("Evaluation/worker specification mismatch")
        phase = "deferred_evaluation"
        from stable_baselines3.sac.policies import Actor

        initial_actor = Actor.load(out / "initial-actor.pt", device="cpu")
        before = evaluate(initial_actor, env, ids, "initial", out, initial_hash)
        after = evaluate(model.actor, env, ids, "trained", out, final_hash)
        result = {
            "allocation_lifecycle": "deferred-evaluation-v1",
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
        failure = error
        write_json(
            out / "fault.json",
            {
                "error": str(error),
                "phase": phase,
                "wall_s": time.perf_counter() - started,
                "invalid_run": True,
                "cleanup_errors": owner.cleanup_errors,
            },
        )
        try:
            if model is not None:
                model.save(out / "fault-checkpoint.zip")
            write_json(
                out / "fault-states.json",
                {
                    "last_valid_workers": captured
                    if captured is not None
                    else vec.env_method("snapshot")
                    if vec is not None
                    else [],
                    "origin": "training-end"
                    if captured is not None
                    else "live-workers"
                    if vec is not None
                    else "unavailable-partial-construction",
                },
            )
        except Exception as capture_error:  # noqa: BLE001 -- preserve original fault
            write_json(out / "fault-capture-error.json", {"error": str(capture_error)})
        raise
    finally:
        try:
            owner.close(partial=vec is None)
        except Exception as cleanup_error:
            write_json(
                out / "cleanup-fault.json",
                {"error": str(cleanup_error), "phase": phase},
            )
            if failure is None:
                raise
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as cleanup_error:
                    write_json(
                        out / "evaluation-cleanup-fault.json",
                        {"error": str(cleanup_error)},
                    )
                    if failure is None:
                        raise
