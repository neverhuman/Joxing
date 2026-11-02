"""H22 explicit four-world constructor/serial release; no training adoption."""

import argparse
import json
from functools import partial
from itertools import pairwise
from pathlib import Path

from .deferred_qualification import value_digest, wait_ready
from .native_context import NativeContextProof, bound_process, constructor_worker
from .physics_credit import process_identity
from .process_exit_probe import facts, validate_server, write_json
from .serial_retirement import serial_vec_class


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def observe_four(proof, vec, folder):
    if len(vec.processes) != 4:
        raise ValueError("Four native context denominator")
    server_after = facts(proof.server, evidence_dir=proof.folder / "proc-server-after")
    workers = []
    for i, process in enumerate(vec.processes):
        bound = bound_process(process.pid)
        world = Path(folder) / f"world-{i}"
        before = json.loads((world / "context-worker-before.json").read_text())
        if (
            before["worker"] != bound
            or before["server"] != proof.server
            or before["method"] != "forkserver"
            or before["proc"]["identity"] != bound
            or before["proc"]["threads"] != 1
            or before["proc"]["after_threads"] != 1
        ):
            raise ValueError("Four native context witness drift")
        after = facts(bound, evidence_dir=world / "proc-context-worker-after")
        validate_server(server_after, after, proof.proof["root"]["pid"])
        if (
            before["proc"]["parent_pid"] != proof.server["pid"]
            or after["threads"] != 1
            or after["after_threads"] != 1
        ):
            raise ValueError("Four native context ancestry/threads")
        workers.append({"index": i, "worker": bound, "before": before, "after": after})
    if len({(w["worker"]["pid"], w["worker"]["start"]) for w in workers}) != 4:
        raise ValueError("Four native context duplicate worker")
    result = proof.proof | {
        "schema": "joxing-four-native-context-proof-v1",
        "workers": workers,
        "server_after": server_after,
    }
    write_json(proof.folder / "after.json", result)
    return result


def validate_world(folder, index, worker, state):
    world = Path(folder) / f"world-{index}"
    closed = lines(world / f"traces/worker-{index}.jsonl")
    phases = lines(world / "resource-release.jsonl")
    finalizer = lines(world / "context-finalizer.jsonl")
    if (
        len(closed) != 1
        or closed[0]["owner"] != worker
        or closed[0]["semantic"]["phase"] != "close"
        or closed[0]["semantic"]["state"] != state
    ):
        raise ValueError("Four native close state/identity mismatch")
    if [p["phase"] for p in phases] != [
        "before_close",
        "inner_closed",
        "graph_checked",
        "released",
    ]:
        raise ValueError("Four native release phases")
    if (
        any(p["identity"] != worker for p in phases)
        or phases[0]["state_hash"] != state
        or phases[1]["state_hash"] != state
    ):
        raise ValueError("Four native release state/identity")
    if phases[0]["muscles"] != 700 or phases[2]["weakrefs_dead"] != {
        "env": True,
        "model": True,
        "data": True,
    }:
        raise ValueError("Four native lifetime/muscles")
    if (
        phases[2]["observed_rss_change_bytes"]
        != phases[2]["memory"]["rss_bytes"] - phases[0]["memory"]["rss_bytes"]
    ):
        raise ValueError("Four native RSS difference mismatch")
    if (
        not phases[0]["monotonic_s"]
        <= closed[0]["monotonic_s"]
        <= phases[1]["monotonic_s"]
    ):
        raise ValueError("Four native close trace chronology")
    if any(a["monotonic_s"] > b["monotonic_s"] for a, b in pairwise(phases)):
        raise ValueError("Four native release chronology")
    for phase in (phases[0], phases[2]):
        memory = phase["memory"]
        if type(memory["rss_bytes"]) is not int or memory["rss_bytes"] <= 0:
            raise ValueError("Four native live RSS")
        for record in (memory["before"], memory["after"]):
            if (
                record["pid"] != worker["pid"]
                or record["start"] != worker["start"]
                or record["state"] in ("Z", "X")
                or record["flags"] & 4
            ):
                raise ValueError("Four native live identity/RSS")
    if (
        len(finalizer) != 1
        or finalizer[0]["phase"] != "managed_finalizer"
        or finalizer[0]["identity"] != worker
        or finalizer[0]["monotonic_s"] < phases[-1]["monotonic_s"]
    ):
        raise ValueError("Four native finalizer")
    if list((world / "guard-events").glob("*.jsonl")):
        raise ValueError("Four native forbidden operation")
    return {
        "index": index,
        "worker": worker,
        "state_hash": state,
        "release_phases": phases,
        "finalizer": finalizer[0],
    }


def run(folder, slot, packet, ledger_path, binding, ready):
    if (
        packet["schema"] != "joxing-four-native-context-v1"
        or packet["simulation_workers"] != 4
    ):
        raise ValueError("Explicit four native packet required")
    wait_ready(ready, 15)
    folder = Path(folder)
    proof = NativeContextProof(Path(str(folder) + "-context"))
    proof.prepare()  # This fresh root/server are still stdlib-only.
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from .deferred_trial import VectorOwner
    from .physics_credit import PhysicsCredit

    folder.mkdir(parents=True, exist_ok=False)
    (folder / "trial").mkdir()
    owner, vec, failure = VectorOwner(), None, None
    try:
        factories = [
            partial(
                constructor_worker,
                str(folder / "trial"),
                i,
                packet["seed"],
                1.0,
                folder=str(folder / f"world-{i}"),
                server=proof.server,
                ledger_path=ledger_path,
                binding=binding,
                slot=slot,
            )
            for i in range(4)
        ]
        adapter = serial_vec_class(
            SubprocVecEnv, folder / "serial-close.jsonl", packet["close_deadline_s"]
        )
        vec = owner.construct(adapter, factories, start_method="forkserver")
        specifications, snapshots = (
            vec.env_method("specification"),
            vec.env_method("snapshot"),
        )
        context = observe_four(proof, vec, folder)
        workers = [w["worker"] for w in context["workers"]]
        hashes = [value_digest(s) for s in snapshots]
        preparation, rollout = (
            vec.get_attr("preparation_total"),
            vec.get_attr("physics_total"),
        )
        native = [
            lines(folder / f"world-{i}/traces/native-world-{i}.jsonl") for i in range(4)
        ]
        muscles = [
            n[0]["semantic"].get("muscles") if len(n) == 1 else None for n in native
        ]
        before = {
            "workers": workers,
            "specifications": specifications,
            "initial_and_close_state_hashes": hashes,
            "muscles_per_worker": muscles,
            "preparation_steps": sum(preparation),
            "rollout_steps": sum(rollout),
            "per_worker_preparation_steps": preparation,
            "per_worker_rollout_steps": rollout,
            "actors_constructed": 0,
            "policy_inferences": 0,
            "learner_updates": 0,
            "normalizer": "none",
        }
        write_json(folder / "before-close.json", before)
        if (
            specifications != [packet["expected_specification"]] * 4
            or hashes != [packet["expected_state_hash"]] * 4
        ):
            raise ValueError("Four native frozen state/specification drift")
        if rollout != [0] * 4 or preparation != [200] * 4:
            raise ValueError("Four native physical denominator")
        for i in range(4):
            n = native[i]
            if (
                len(n) != 1
                or n[0]["owner"] != workers[i]
                or n[0]["semantic"]["phase"] != "reset"
                or muscles[i] != 700
            ):
                raise ValueError("Four native reset/muscle identity")
        owner.close()  # One original serial adapter, one two-second deadline.
        worlds = [
            validate_world(folder, i, worker, hashes[i])
            for i, worker in enumerate(workers)
        ]
        close = lines(folder / "serial-close.jsonl")
        if [(e["phase"], e.get("worker")) for e in close] != [("bound", None)] + [
            (phase, i) for i in range(4) for phase in ("close_send", "joined")
        ] + [("closed", None)]:
            raise ValueError("Four native serial phase denominator")
        if (
            close[-2]["monotonic_s"] - close[1]["monotonic_s"]
            > packet["close_deadline_s"]
        ):
            raise ValueError("Four native whole close deadline")
        for i, world in enumerate(worlds):
            sent, joined = close[1 + 2 * i : 3 + 2 * i]
            if (
                joined["exitcode"] != 0
                or any(
                    e["identity"]["pid"] != workers[i]["pid"]
                    or e["identity"]["start"] != workers[i]["start"]
                    for e in (sent, joined)
                )
                or not sent["monotonic_s"]
                <= world["release_phases"][0]["monotonic_s"]
                <= world["finalizer"]["monotonic_s"]
                <= joined["monotonic_s"]
            ):
                raise ValueError("Four native close/finalizer identity/chronology")
        credit = PhysicsCredit(ledger_path, binding).snapshot()
        if credit["used"] != 800 or credit["pending"]:
            raise ValueError("Four native credit denominator")
        result = before | {
            "schema": packet["schema"],
            "context": context,
            "worlds": worlds,
            "close_transport": "serial-close-v1",
            "resource_release": "live-graph-release-v1",
            "close_to_join_s": close[-2]["monotonic_s"] - close[1]["monotonic_s"],
        }
        write_json(folder / "constructor-result.json", result)
        return result
    except BaseException as error:
        failure = error
        chain, current = [], error
        while current is not None:
            chain.append(type(current).__name__ + ": " + str(current))
            current = current.__cause__
        write_json(
            folder / "constructor-fault.json",
            {"chain": chain, "owner": process_identity()},
        )
        raise
    finally:
        try:
            owner.close(partial=vec is None)
        except Exception as cleanup_error:
            if failure is None:
                raise
            write_json(
                folder / "cleanup-fault.json",
                {"original": str(failure), "cleanup": str(cleanup_error)},
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
