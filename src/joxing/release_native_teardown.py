"""ONE native constructor with pre-close proof and child-side graph release."""

import argparse
import contextlib
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

from .deferred_qualification import value_digest, wait_ready
from .physics_credit import process_identity
from .serial_retirement import serial_vec_class


def constructor_worker(*args, folder, **kwargs):
    import gymnasium as gym
    from stable_baselines3.sac.policies import Actor

    from .live_resource_release import release_wrapper_class
    from .serial_native_teardown import constructor_worker as guarded_worker
    from .trial_memory import guard

    stack = contextlib.ExitStack()
    stack.enter_context(
        patch.object(
            Actor,
            "__init__",
            guard(Path(folder) / "guard-events", "none", "world-0", "actor.__init__"),
        )
    )
    env = guarded_worker(*args, folder=folder, **kwargs)
    wrapper = release_wrapper_class(
        gym.Wrapper,
        Path(folder) / "resource-release.jsonl",
        state_hash=value_digest,
        bound=process_identity(),
    )
    result = wrapper(env)
    result._release_guard_stack = stack
    return result


def run(folder, slot, packet, ledger_path, binding, ready):
    wait_ready(ready, 15)
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from .deferred_trial import VectorOwner
    from .physics_credit import PhysicsCredit
    from .provenance import write_json

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "trial").mkdir()
    owner, vec, failure = VectorOwner(), None, None
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
        adapter = serial_vec_class(
            SubprocVecEnv, folder / "serial-close.jsonl", packet["close_deadline_s"]
        )
        vec = owner.construct(adapter, [factory], start_method="spawn")
        specifications, snapshots = (
            vec.env_method("specification"),
            vec.env_method("snapshot"),
        )
        if specifications != [packet["expected_specification"]]:
            raise ValueError("Release constructor specification drift")
        if vec.get_attr("physics_total") != [0] or vec.get_attr(
            "preparation_total"
        ) != [200]:
            raise ValueError("Release constructor physics denominator")
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
            raise ValueError("Release constructor native trace")
        before = {
            "worker": native[0]["owner"],
            "specification": specifications[0],
            "initial_state_hash": value_digest(snapshots[0]),
            "muscles": 700,
            "preparation_steps": 200,
            "rollout_steps": 0,
            "actors_constructed": 0,
            "policy_inferences": 0,
            "learner_updates": 0,
            "normalizer": "none",
        }
        write_json(folder / "before-close.json", before)
        owner.close()
        closed = [
            json.loads(line)
            for line in (folder / "traces/worker-0.jsonl").read_text().splitlines()
        ]
        release = [
            json.loads(line)
            for line in (folder / "resource-release.jsonl").read_text().splitlines()
        ]
        if [r["phase"] for r in release] != [
            "before_close",
            "inner_closed",
            "graph_checked",
            "released",
        ]:
            raise ValueError("Incomplete resource release phases")
        if (
            len(closed) != 1
            or closed[0]["semantic"]["phase"] != "close"
            or closed[0]["owner"] != before["worker"]
            or closed[0]["semantic"]["state"] != before["initial_state_hash"]
            or any(r["identity"] != before["worker"] for r in release)
            or release[0]["state_hash"] != before["initial_state_hash"]
            or release[1]["state_hash"] != before["initial_state_hash"]
            or release[2]["weakrefs_dead"] != {"env": True, "model": True, "data": True}
        ):
            raise ValueError("Native resource release identity/state/lifetime mismatch")
        if list((folder / "guard-events").glob("*.jsonl")):
            raise ValueError("Forbidden release constructor operation")
        credit = PhysicsCredit(ledger_path, binding).snapshot()
        if credit["used"] != 200 or credit["pending"]:
            raise ValueError("Release constructor credit denominator")
        result = before | {
            "schema": "joxing-release-native-teardown-v1",
            "close_transport": "serial-close-v1",
            "resource_release": "live-graph-release-v1",
            "release_phases": release,
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
