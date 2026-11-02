"""One guarded native constructor/close, no rollout or SAC construction."""

import argparse
import contextlib
import json
from functools import partial
from pathlib import Path

from .deferred_qualification import instrumented_worker, value_digest, wait_ready


def constructor_worker(*args, folder, **kwargs):
    import torch
    from stable_baselines3.sac.policies import Actor

    from .stance_env import StanceEnv
    from .trial_memory import install_guards

    stack = contextlib.ExitStack()
    install_guards(
        stack, Path(folder) / "guard-events", "A", "world-0", torch, Actor, StanceEnv
    )
    env = instrumented_worker(*args, folder=folder, **kwargs)
    env._constructor_guards = stack
    return env


def run(folder, slot, packet, ledger_path, binding, ready):
    wait_ready(ready, 15)
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from .deferred_trial import VectorOwner
    from .physics_credit import PhysicsCredit
    from .provenance import write_json

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "trial").mkdir()
    owner = VectorOwner()
    vec = None
    try:
        factory = partial(
            constructor_worker,
            str(folder / "trial"),
            0,
            packet["seed"],
            1.0,
            folder=str(folder),
            ledger_path=ledger_path,
            binding=binding,
            slot=slot,
        )
        vec = owner.construct(SubprocVecEnv, [factory], start_method="spawn")
        specifications = vec.env_method("specification")
        snapshots = vec.env_method("snapshot")
        if (
            len(specifications) != 1
            or specifications[0] != packet["expected_specification"]
        ):
            raise ValueError("Native constructor specification drift")
        if vec.get_attr("physics_total") != [0] or vec.get_attr(
            "preparation_total"
        ) != [200]:
            raise ValueError("Unexpected native constructor physics")
        native = [
            json.loads(line)
            for line in (folder / "traces/native-world-0.jsonl")
            .read_text()
            .splitlines()
        ]
        if (
            len(native) != 1
            or native[0]["semantic"]["phase"] != "reset"
            or native[0]["semantic"]["muscles"] != 700
        ):
            raise ValueError("Native constructor trace/muscle denominator")
        worker = native[0]["owner"]
        initial = value_digest(snapshots[0])
        owner.close()
        closed = [
            json.loads(line)
            for line in (folder / "traces/worker-0.jsonl").read_text().splitlines()
        ]
        if (
            len(closed) != 1
            or closed[0]["semantic"]["phase"] != "close"
            or closed[0]["semantic"]["state"] != initial
        ):
            raise ValueError("Native changed between construction and close")
        if list((folder / "guard-events").glob("*.jsonl")):
            raise ValueError("Forbidden constructor operation")
        ledger = PhysicsCredit(ledger_path, binding).snapshot()
        if ledger["used"] != 200 or ledger["pending"]:
            raise ValueError("Native preparation credit mismatch")
        result = {
            "schema": "joxing-native-teardown-v1",
            "worker": worker,
            "muscles": 700,
            "specification": specifications[0],
            "initial_and_close_state_hash": initial,
            "preparation_steps": 200,
            "rollout_steps": 0,
            "learner_updates": 0,
            "actors_constructed": 0,
            "model_hash": specifications[0]["model_hash"],
            "action_hash": specifications[0]["action_hash"],
            "observation_hash": specifications[0]["observation_hash"],
            "normalizer": "none",
        }
        write_json(folder / "constructor-result.json", result)
        return result
    finally:
        owner.close(partial=vec is None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("folder", "slot", "packet", "ledger", "binding", "ready"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    run(
        args.folder,
        args.slot,
        json.loads(Path(args.packet).read_text()),
        args.ledger,
        args.binding,
        args.ready,
    )
