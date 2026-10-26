"""Explicitly authorized larger-budget wave, with bounded parallel trials."""

from __future__ import annotations

import contextlib
import copy
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from .provenance import ROOT, digest, manifest, source_hash, write_json
from .stance_env import StanceEnv
from .stance_train import file_hash
from .wave import (
    ARMS,
    ConstantPolicy,
    check_resources,
    evaluate,
    select_arm,
    train_run,
    validate_protocol,
)


def extension_config(value):
    base = validate_protocol(
        json.loads((ROOT / "experiments/v0.0.5/protocol.json").read_text())
    )
    if set(value) != {
        "schema",
        "description",
        "parent_protocol_hash",
        "steps",
        "run_wall_s",
        "parallel_trials",
        "workers_per_trial",
        "screen_seed",
        "confirmation_seeds",
        "screen_ids",
        "confirmation_ids",
        "selection",
        "experiments",
    }:
        raise ValueError("Unknown or missing extension fields")
    if value["schema"] != "joxing-wave-extension-0.0.5" or value[
        "parent_protocol_hash"
    ] != digest(base):
        raise ValueError("Incompatible parent protocol")
    if {
        k: value[k]
        for k in (
            "steps",
            "run_wall_s",
            "parallel_trials",
            "workers_per_trial",
            "screen_seed",
            "confirmation_seeds",
        )
    } != {
        "steps": 160000,
        "run_wall_s": 2700,
        "parallel_trials": 4,
        "workers_per_trial": 4,
        "screen_seed": 547,
        "confirmation_seeds": [761, 983],
    }:
        raise ValueError("Registered extension budget/seeds cannot change")
    if (
        value["screen_ids"] != list(range(64000, 64032))
        or value["confirmation_ids"] != list(range(65000, 65032))
        or value["selection"] != base["selection"]
    ):
        raise ValueError("Registered splits/selection cannot change")
    ids = [x["id"] for x in value["experiments"]]
    expected = [
        "B0",
        "S-A",
        "S-B",
        "S-C",
        "S-D",
        "R-A-761",
        "R-W-761",
        "R-A-983",
        "R-W-983",
    ]
    if ids != expected or any(
        not x.get("hypothesis") or not x.get("description")
        for x in value["experiments"]
    ):
        raise ValueError("Missing/duplicate experiment descriptions")
    config = copy.deepcopy(base)
    config["learner"]["steps"] = value["steps"]
    config["caps"]["run_wall_s"] = value["run_wall_s"]
    config["seeds"] = {
        "screen": value["screen_seed"],
        "confirmation": value["confirmation_seeds"],
    }
    config["splits"]["screen"] = value["screen_ids"]
    config["splits"]["confirmation"] = value["confirmation_ids"]
    return config


def child_trial(job):
    """Each child exclusively owns its trial directory and four world workers."""
    (
        out,
        key,
        config,
        arm,
        seed,
        ids,
        provenance,
        stop,
        counters,
        prior_steps,
        deadline,
    ) = job
    out = Path(out)

    def budget(current):
        counters[key] = current
        if stop.is_set():
            raise RuntimeError("Stopped after peer trial fault")
        if (
            datetime.now(UTC).timestamp() >= deadline
            or prior_steps + sum(counters.values()) > config["caps"]["physics_steps"]
        ):
            raise RuntimeError("Combined wave wall/physics cap")
        if (
            sum(p.stat().st_size for p in out.parent.rglob("*") if p.is_file())
            > config["caps"]["raw_bytes"]
        ):
            raise RuntimeError("Extension artifact cap")

    with (
        (out.parent / f"{key}-output.txt").open("x") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        return train_run(
            out,
            config,
            arm,
            seed,
            ids,
            provenance,
            budget,
            defer_initial_evaluation=key.startswith("R-"),
        )


def run_extension(protocol_path, original, out):
    import stable_baselines3
    import torch

    value = json.loads(Path(protocol_path).read_text())
    config = extension_config(value)
    original = Path(original)
    parent = json.loads((original / "result.json").read_text())
    parent_provenance = json.loads((original / "provenance.json").read_text())
    assert parent_provenance["protocol_hash"] == value["parent_protocol_hash"]
    assert parent["source_unchanged"]
    assert json.loads((original / "P0/result.json").read_text())["passed"]
    # One global four-hour deadline, measured from original launch, is retained.
    deadline = datetime.fromisoformat(parent_provenance["utc"]).timestamp() + 14400
    if datetime.now(UTC).timestamp() >= deadline:
        raise RuntimeError("Combined wall cap already reached")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    provenance = manifest() | {
        "protocol_hash": digest(value),
        "parent_commit": parent_provenance["code_commit"],
        "parent_protocol_hash": value["parent_protocol_hash"],
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "resources": check_resources(out, config["caps"], True),
        "parallel_trials": 4,
        "aggregate_simulation_workers": 16,
    }
    write_json(out / "protocol.json", value)
    write_json(out / "provenance.json", provenance)
    registry = {
        x["id"]: x | {"status": "not_run", "reason": "pending_dependencies"}
        for x in value["experiments"]
    }

    def event(key, status, **extra):
        registry[key].update(status=status, **extra)
        write_json(out / "registry.json", registry)
        with (out / "events.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "experiment": key,
                        "status": status,
                        "wall_s": time.perf_counter() - started,
                        **extra,
                    }
                )
                + "\n"
            )

    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        stop = manager.Event()
        counters = manager.dict()
        # Reserve all supplementary H0/audit work conservatively (1M steps),
        # in addition to exact original and extension counters.
        prior_steps = parent["accounted_physics_substeps"] + 1000000

        def batch(jobs):
            results = {}
            with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
                futures = {}
                for key, arm, seed, ids in jobs:
                    event(key, "running", reason="dependencies_passed")
                    write_json(
                        out / f"{key}-description.json",
                        registry[key]
                        | {
                            "arm": arm,
                            "seed": seed,
                            "ids": ids,
                            "config": config,
                            "provenance": provenance,
                        },
                    )
                    job = (
                        str(out / key),
                        key,
                        config,
                        arm,
                        seed,
                        ids,
                        provenance,
                        stop,
                        counters,
                        prior_steps,
                        deadline,
                    )
                    futures[pool.submit(child_trial, job)] = key
                error = None
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        r = future.result()
                        results[key] = r
                        counters[key] = (
                            r["counts"]["physics_substeps"]
                            + r["counts"]["preparation_substeps"]
                            + r["evaluation_physics_substeps"]
                            + r["evaluation_preparation_substeps"]
                        )
                        event(
                            key,
                            "completed"
                            if r["scheduled_training_complete"]
                            else "incomplete",
                            result=f"{key}/result.json",
                            reason=r["stop_reason"] or "finished",
                        )
                    except Exception as exc:  # noqa: BLE001 -- drain peers, then raise
                        stop.set()
                        event(key, "failed", reason=str(exc))
                        error = exc
                if error is not None:
                    raise error
            return results

        try:
            event("B0", "running", reason="reuse_frozen_constant_only")
            write_json(
                out / "B0-description.json",
                registry["B0"]
                | {"parent_protocol_hash": value["parent_protocol_hash"]},
            )
            folder = out / "B0"
            folder.mkdir()
            old_baseline = json.loads((original / "B0/result.json").read_text())
            chosen = old_baseline["chosen_excitation"]
            diagnostic_path = original / f"B0/exc-{chosen}-evaluations.json"
            (folder / "reused-diagnostic-evaluations.json").write_bytes(
                diagnostic_path.read_bytes()
            )
            env = StanceEnv(training=False)
            evaluate(
                ConstantPolicy(chosen),
                env,
                value["screen_ids"],
                "screen-constant",
                folder,
            )
            baseline = {
                "chosen_excitation": chosen,
                "diagnostic_source": "reused-diagnostic-evaluations.json",
                "reused_diagnostic_sha256": file_hash(diagnostic_path),
                "reused_diagnostic_attempts_not_counted_again": 8,
                "new_evaluation_attempts": 32,
                "physics_substeps": env.physics_total,
                "preparation_substeps": env.preparation_total,
            }
            write_json(folder / "result.json", baseline)
            counters["baseline"] = env.physics_total + env.preparation_total
            event(
                "B0",
                "completed",
                result="B0/result.json",
                reason="32_new_valid_baseline_trials",
            )
            screen_results = batch(
                [(f"S-{a}", a, value["screen_seed"], value["screen_ids"]) for a in ARMS]
            )
            screen = {a: screen_results[f"S-{a}"] for a in ARMS}
            if len({r["initial_actor_hash"] for r in screen.values()}) != 1:
                raise RuntimeError("Unmatched extension initial actors")
            selection = select_arm(screen)
            write_json(out / "selection.json", selection)
            if selection["qualified"]:
                winner = selection["arm"]
                env = StanceEnv(training=False)
                evaluate(
                    ConstantPolicy(chosen),
                    env,
                    value["confirmation_ids"],
                    "confirmation-constant",
                    folder,
                )
                counts = {
                    "physics_substeps": env.physics_total,
                    "preparation_substeps": env.preparation_total,
                }
                write_json(folder / "confirmation-baseline-result.json", counts)
                counters["confirmation-baseline"] = sum(counts.values())
                jobs = []
                for seed in value["confirmation_seeds"]:
                    jobs.append((f"R-A-{seed}", "A", seed, value["confirmation_ids"]))
                    if winner != "A":
                        jobs.append(
                            (f"R-W-{seed}", winner, seed, value["confirmation_ids"])
                        )
                confirmations = batch(jobs)
                for seed in value["confirmation_seeds"]:
                    if (
                        winner != "A"
                        and confirmations[f"R-A-{seed}"]["initial_actor_hash"]
                        != confirmations[f"R-W-{seed}"]["initial_actor_hash"]
                    ):
                        raise RuntimeError("Unmatched confirmation actors")
            for key, row in registry.items():
                if row["status"] == "not_run":
                    event(
                        key,
                        "not_run",
                        reason="duplicate_reference_slot"
                        if selection["qualified"]
                        else selection["reason"],
                    )
            result = {
                "selection": selection,
                "registry": registry,
                "accounted_physics_substeps": sum(counters.values()),
                "original_physics_substeps": parent["accounted_physics_substeps"],
                "supplementary_physics_reserve": 1000000,
                "wall_s": time.perf_counter() - started,
                "source_unchanged": source_hash() == provenance["source_hash"],
                "scope": "User-authorized longer-budget development screen; seed/budget/split differ from original, not a replication of it. No boxing/self-play.",
            }
            if not result["source_unchanged"]:
                raise RuntimeError("Executing extension source changed")
            write_json(out / "result.json", result)
            return result
        except Exception as error:
            stop.set()
            for key, row in registry.items():
                if row["status"] == "not_run":
                    event(key, "not_run", reason="halted_after_fault")
            write_json(out / "fault.json", {"error": str(error), "registry": registry})
            raise
