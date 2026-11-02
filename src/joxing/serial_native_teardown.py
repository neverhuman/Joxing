"""Four original guarded constructors; opt-in serial close, no learning."""

import argparse
import contextlib
import json
from functools import partial
from pathlib import Path

from .deferred_qualification import instrumented_worker, value_digest, wait_ready
from .serial_retirement import serial_vec_class


def constructor_worker(out, index, *args, folder, **kwargs):
    import torch
    from stable_baselines3.sac.policies import Actor

    from .stance_env import StanceEnv
    from .trial_memory import install_guards

    stack = contextlib.ExitStack()
    install_guards(
        stack,
        Path(folder) / "guard-events",
        "A",
        f"world-{index}",
        torch,
        Actor,
        StanceEnv,
    )
    env = instrumented_worker(out, index, *args, folder=folder, **kwargs)
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
    failure = None
    try:
        factories = [
            partial(
                constructor_worker,
                str(folder / "trial"),
                i,
                packet["seed"],
                1.0,
                folder=str(folder),
                ledger_path=ledger_path,
                binding=binding,
                slot=slot,
            )
            for i in range(4)
        ]
        adapter = serial_vec_class(
            SubprocVecEnv, folder / "serial-close.jsonl", packet["close_deadline_s"]
        )
        vec = owner.construct(adapter, factories, start_method="spawn")
        specifications = vec.env_method("specification")
        snapshots = vec.env_method("snapshot")
        if len(specifications) != 4 or any(
            s != packet["expected_specification"] for s in specifications
        ):
            raise ValueError("Serial native specification drift")
        if (
            vec.get_attr("physics_total") != [0] * 4
            or vec.get_attr("preparation_total") != [200] * 4
        ):
            raise ValueError("Serial constructor physics denominator")
        workers, hashes = [], [value_digest(s) for s in snapshots]
        for i in range(4):
            native = [
                json.loads(line)
                for line in (folder / f"traces/native-world-{i}.jsonl")
                .read_text()
                .splitlines()
            ]
            if (
                len(native) != 1
                or native[0]["semantic"]["phase"] != "reset"
                or native[0]["semantic"]["muscles"] != 700
            ):
                raise ValueError("Serial native constructor trace/muscle denominator")
            workers.append(native[0]["owner"])
        owner.close()
        for i, state in enumerate(hashes):
            closed = [
                json.loads(line)
                for line in (folder / f"traces/worker-{i}.jsonl")
                .read_text()
                .splitlines()
            ]
            if (
                len(closed) != 1
                or closed[0]["semantic"]["phase"] != "close"
                or closed[0]["owner"] != workers[i]
                or closed[0]["semantic"]["state"] != state
            ):
                raise ValueError("Serial native changed before close")
        if list((folder / "guard-events").glob("*.jsonl")):
            raise ValueError("Forbidden serial constructor operation")
        ledger = PhysicsCredit(ledger_path, binding).snapshot()
        if ledger["used"] != 800 or ledger["pending"]:
            raise ValueError("Serial constructor credit mismatch")
        result = {
            "schema": "joxing-serial-native-teardown-v1",
            "close_transport": "serial-close-v1",
            "workers": workers,
            "muscles_per_worker": [700] * 4,
            "specification": specifications[0],
            "initial_and_close_state_hashes": hashes,
            "preparation_steps": 800,
            "rollout_steps": 0,
            "learner_updates": 0,
            "actors_constructed": 0,
            "policy_inferences": 0,
            "normalizer": "none",
        }
        write_json(folder / "constructor-result.json", result)
        return result
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            owner.close(partial=vec is None)
        except Exception as cleanup_error:
            if failure is None:
                raise
            write_json(
                folder / "cleanup-fault.json",
                {
                    "original": type(failure).__name__ + ": " + str(failure),
                    "cleanup": type(cleanup_error).__name__ + ": " + str(cleanup_error),
                },
            )


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
