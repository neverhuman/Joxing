"""Opt-in clean forkserver and native release for observed SAC parity.

No heavy import, global start-method change, or default training-path change.
"""

import argparse
import json
from functools import partial
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

from .deferred_qualification import value_digest, wait_ready
from .four_native_context import observe_four
from .native_context import (
    NativeContextProof,
    finalizer_witness,
    worker_witness,
)
from .physics_credit import process_identity
from .process_exit_probe import write_json
from .serial_retirement import serial_vec_class

SCHEMA = "joxing-native-learner-context-parity-v1"
PROFILE = "clean-forkserver-live-release-serial-v1"


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def qualify_retirements(facts, watch):
    workers = [w["worker"] for w in facts["context"]["workers"]]
    if len(workers) != 4 or len({(w["pid"], w["start"]) for w in workers}) != 4:
        raise ValueError("Training retirement worker denominator")
    records = [
        watch["retirement_records"].get(f"{w['pid']}:{w['start']}") for w in workers
    ]
    if any(
        r is None
        or not r["released"]
        or r["highwater_charge_bytes"] <= 0
        or r["max_hold_s"] > 0.1
        for r in records
    ):
        raise ValueError("All four actual trained retirements required")
    return records


def training_worker(*args, folder, server, **kwargs):
    """Fresh-child instrumentation, not a constructor-only no-step guard."""
    from multiprocessing.util import Finalize

    import gymnasium as gym

    from .deferred_qualification import instrumented_worker
    from .live_resource_release import release_wrapper_class

    index = args[1]
    world = Path(folder) / f"world-{index}"
    worker_witness(world, server)
    write_json(
        world / "training-factory.json",
        {"module": __name__, "profile": PROFILE, "training_allowed": True},
    )
    bound = process_identity()
    Finalize(None, finalizer_witness, args=(world, bound), exitpriority=5)
    inner = instrumented_worker(*args, folder=folder, **kwargs)
    release = release_wrapper_class(
        gym.Wrapper,
        world / "resource-release.jsonl",
        state_hash=value_digest,
        bound=bound,
    )

    class TrainingRelease(release):
        def context_training_state(self):
            self._require_open()
            core = self.env.unwrapped
            rng = self.env.get_wrapper_attr("reset_rng")
            return {
                "state": value_digest(core.snapshot()),
                "reset_rng": value_digest(rng.bit_generator.state),
                "muscles": int(core.model.nu),
                "physics": int(core.physics_total),
                "preparation": int(core.preparation_total),
            }

    return TrainingRelease(inner)


def validate_close(folder, before):
    """Strict dynamic state/RNG/lifetime proof, independent of training result."""
    folder = Path(folder)
    if len(before["workers"]) != 4 or len(before["states"]) != 4:
        raise ValueError("Training close denominator")
    if len({(w["pid"], w["start"]) for w in before["workers"]}) != 4:
        raise ValueError("Duplicate training worker")
    events = lines(folder / "serial-close.jsonl")
    if [(e["phase"], e.get("worker")) for e in events] != [("bound", None)] + [
        (p, i) for i in range(4) for p in ("close_send", "joined")
    ] + [("closed", None)]:
        raise ValueError("Training serial-close phases")
    duration = events[-2]["monotonic_s"] - events[1]["monotonic_s"]
    if not 0 <= duration <= 2:
        raise ValueError("Training whole close deadline")
    if any(a["monotonic_s"] > b["monotonic_s"] for a, b in pairwise(events)):
        raise ValueError("Training serial event order")
    worlds = []
    for i, (worker, state) in enumerate(
        zip(before["workers"], before["states"], strict=True)
    ):
        trace = lines(folder / f"traces/worker-{i}.jsonl")
        closed = [r for r in trace if r["semantic"]["phase"] == "close"]
        world = folder / f"world-{i}"
        phases, finalizers = (
            lines(world / "resource-release.jsonl"),
            lines(world / "context-finalizer.jsonl"),
        )
        if (
            len(closed) != 1
            or trace[-1] != closed[0]
            or closed[0]["owner"] != worker
            or closed[0]["semantic"]["state"] != state["state"]
            or closed[0]["semantic"]["reset_rng"] != state["reset_rng"]
            or state["muscles"] != 700
            or sum(r["semantic"]["phase"] == "step" for r in trace) != 16
        ):
            raise ValueError("Trained state/RNG/worker trace mismatch")
        if [r["phase"] for r in phases] != [
            "before_close",
            "inner_closed",
            "graph_checked",
            "released",
        ]:
            raise ValueError("Training release phases")
        if any(r["identity"] != worker for r in phases):
            raise ValueError("Training release owner")
        if (
            phases[0]["state_hash"] != state["state"]
            or phases[1]["state_hash"] != state["state"]
            or phases[0]["muscles"] != 700
        ):
            raise ValueError("Training release state")
        if phases[2]["weakrefs_dead"] != {"env": True, "model": True, "data": True}:
            raise ValueError("Trained native graph still referenced")
        for record in (phases[0], phases[2]):
            memory = record["memory"]
            if type(memory["rss_bytes"]) is not int or memory["rss_bytes"] <= 0:
                raise ValueError("Training positive live RSS missing")
            for ident in (memory["before"], memory["after"]):
                if (
                    ident["pid"] != worker["pid"]
                    or ident["start"] != worker["start"]
                    or ident["state"] in ("Z", "X")
                    or ident["flags"] & 4
                ):
                    raise ValueError("Training live RSS identity")
        if (
            phases[2]["observed_rss_change_bytes"]
            != phases[2]["memory"]["rss_bytes"] - phases[0]["memory"]["rss_bytes"]
        ):
            raise ValueError("Training live RSS difference")
        if (
            len(finalizers) != 1
            or finalizers[0]["identity"] != worker
            or finalizers[0]["phase"] != "managed_finalizer"
        ):
            raise ValueError("Training managed finalizer")
        sent, joined = events[1 + 2 * i : 3 + 2 * i]
        if joined["exitcode"] != 0 or any(
            e["identity"]["pid"] != worker["pid"]
            or e["identity"]["start"] != worker["start"]
            for e in (sent, joined)
        ):
            raise ValueError("Training joined identity")
        times = [
            sent["monotonic_s"],
            phases[0]["monotonic_s"],
            closed[0]["monotonic_s"],
            *[p["monotonic_s"] for p in phases[1:]],
            finalizers[0]["monotonic_s"],
            joined["monotonic_s"],
        ]
        if any(a > b for a, b in pairwise(times)):
            raise ValueError("Training close chronology")
        worlds.append(
            {
                "index": i,
                "worker": worker,
                "state": state,
                "release_phases": phases,
                "finalizer": finalizers[0],
            }
        )
    return {"worlds": worlds, "close_to_join_s": duration}


def context_vector_class(base, proof, folder, packet):
    serial = serial_vec_class(
        base, Path(folder) / "serial-close.jsonl", packet["close_deadline_s"]
    )

    class ContextVector(serial):
        def __init__(self, factories, *, start_method):
            if start_method != "spawn" or len(factories) != 4:
                raise ValueError(
                    "Explicit original four-factory spawn request required"
                )
            write_json(
                Path(folder) / "requested-context.json",
                {
                    "requested": start_method,
                    "actual": "forkserver",
                    "profile": packet["execution_profile"],
                },
            )
            super().__init__(factories, start_method="forkserver")
            self.training_context = observe_four(proof, self, folder)
            specifications = self.env_method("specification")
            states = self.env_method("context_training_state")
            if specifications != [packet["expected_specification"]] * 4 or any(
                s["muscles"] != 700 or s["state"] != packet["expected_state_hash"]
                for s in states
            ):
                raise ValueError("Training constructor state/specification drift")
            write_json(Path(folder) / "training-constructor-states.json", states)

        def close(self):
            if self.closed:
                return
            before = {
                "workers": [w["worker"] for w in self.training_context["workers"]],
                "states": self.env_method("context_training_state"),
            }
            write_json(Path(folder) / "trained-before-close.json", before)
            super().close()
            result = validate_close(folder, before)
            write_json(
                Path(folder) / "training-context-result.json",
                result
                | {
                    "schema": SCHEMA,
                    "context": self.training_context,
                    "execution_profile": packet["execution_profile"],
                },
            )

    return ContextVector


def run_slot_context(folder, slot, packet, ledger_path, binding, ready):
    if (
        packet["schema"] != SCHEMA
        or packet["simulation_workers"] != 4
        or packet["close_deadline_s"] != 2
        or packet["execution_profile"] != PROFILE
    ):
        raise ValueError("Explicit learner context packet required")
    wait_ready(ready, 15)
    folder = Path(folder)
    proof = NativeContextProof(Path(str(folder) + "-context"))
    proof.prepare()
    from stable_baselines3.common import vec_env

    from . import deferred_qualification

    adapter = context_vector_class(vec_env.SubprocVecEnv, proof, folder, packet)
    factory = partial(training_worker, server=proof.server)
    try:
        with (
            patch.object(vec_env, "SubprocVecEnv", adapter),
            patch.object(deferred_qualification, "instrumented_worker", factory),
        ):
            deferred_qualification.run_slot(
                folder, slot, packet, ledger_path, binding, ready
            )
        # compare_slots remains the original authority for training semantics.
        result = json.loads((folder / "training-context-result.json").read_text())
        if (
            result["execution_profile"] != packet["execution_profile"]
            or len(result["worlds"]) != 4
        ):
            raise ValueError("Incomplete trained context result")
    except BaseException as error:
        folder.mkdir(parents=True, exist_ok=True)
        write_json(
            folder / "training-context-fault.json",
            {
                "error": type(error).__name__ + ": " + str(error),
                "owner": process_identity(),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("folder", "slot", "packet", "ledger", "binding", "ready"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    p = json.loads(Path(args.packet).read_text())
    slot = next(s for s in p["slots"] if s["id"] == args.slot)
    # Keep the pickled worker factory bound to its importable package module,
    # rather than a cloudpickled __main__ function with parent module globals.
    from .training_context import run_slot_context as execute

    execute(args.folder, slot, p, args.ledger, args.binding, args.ready)
