import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from joxing import training_context as training


def packet():
    return {
        "schema": training.SCHEMA,
        "execution_profile": training.PROFILE,
        "simulation_workers": 4,
        "close_deadline_s": 2,
        "expected_specification": {"model": "native"},
        "expected_state_hash": "cold",
    }


def test_stdlib_import_before_barrier():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;import joxing.training_context;assert not {'numpy','torch','mujoco','stable_baselines3'}&sys.modules.keys()",
        ],
        check=True,
        timeout=5,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "old"),
        ("simulation_workers", 1),
        ("close_deadline_s", 3),
        ("execution_profile", "unknown"),
    ],
)
def test_wrong_packet_cannot_reach_readiness(monkeypatch, key, value):
    p = packet()
    p[key] = value
    monkeypatch.setattr(
        training, "wait_ready", lambda *a: pytest.fail("wrong packet started")
    )
    with pytest.raises(ValueError):
        training.run_slot_context(None, None, p, None, None, None)


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def saved_close(folder):
    workers = [{"pid": 20 + i, "start": str(100 + i)} for i in range(4)]
    states = [
        {
            "state": "trained-" + str(i),
            "reset_rng": "rng-" + str(i),
            "muscles": 700,
            "physics": 160,
            "preparation": 400,
        }
        for i in range(4)
    ]
    before = {"workers": workers, "states": states}
    events = [{"phase": "bound", "monotonic_s": 99.0, "workers": workers}]
    for i, (owner, state) in enumerate(zip(workers, states, strict=True)):
        start = 100 + i * 0.1
        memory = {
            "rss_bytes": 1000,
            "before": owner | {"state": "R", "flags": 0},
            "after": owner | {"state": "S", "flags": 0},
        }
        release = [
            {"phase": p, "identity": owner, "monotonic_s": start + j * 0.01}
            for j, p in enumerate(
                ["before_close", "inner_closed", "graph_checked", "released"]
            )
        ]
        release[0].update(memory=memory, state_hash=state["state"], muscles=700)
        release[1]["state_hash"] = state["state"]
        release[2].update(
            memory=memory,
            weakrefs_dead={"env": True, "model": True, "data": True},
            observed_rss_change_bytes=0,
        )
        dump(folder / f"world-{i}/resource-release.jsonl", release)
        dump(
            folder / f"world-{i}/context-finalizer.jsonl",
            [
                {
                    "phase": "managed_finalizer",
                    "identity": owner,
                    "monotonic_s": start + 0.04,
                }
            ],
        )
        trace = [
            {"owner": owner, "monotonic_s": 99, "semantic": {"phase": "step"}}
            for _ in range(16)
        ]
        trace.append(
            {
                "owner": owner,
                "monotonic_s": start + 0.005,
                "semantic": {
                    "phase": "close",
                    "state": state["state"],
                    "reset_rng": state["reset_rng"],
                },
            }
        )
        dump(folder / f"traces/worker-{i}.jsonl", trace)
        events.extend(
            [
                {
                    "phase": "close_send",
                    "worker": i,
                    "identity": owner,
                    "monotonic_s": start - 0.001,
                },
                {
                    "phase": "joined",
                    "worker": i,
                    "identity": owner,
                    "exitcode": 0,
                    "monotonic_s": start + 0.05,
                },
            ]
        )
    events.append({"phase": "closed", "monotonic_s": 100.4})
    dump(folder / "serial-close.jsonl", events)
    return before


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "state",
        "rng",
        "step",
        "weak",
        "finalizer",
        "deadline",
        "identity",
        "rss",
        "exitflag",
        "duplicate",
    ],
)
def test_dynamic_close_rejects_corrupt_or_incomplete_proof(tmp_path, fault):
    before = saved_close(tmp_path)
    world = tmp_path / "world-2"
    if fault in ("state", "rng"):
        before["states"][2]["state" if fault == "state" else "reset_rng"] = "changed"
    elif fault == "step":
        path = tmp_path / "traces/worker-2.jsonl"
        dump(path, training.lines(path)[1:])
    elif fault == "finalizer":
        dump(world / "context-finalizer.jsonl", [])
    elif fault in ("weak", "rss", "exitflag"):
        path = world / "resource-release.jsonl"
        rows = training.lines(path)
        if fault == "weak":
            rows[2]["weakrefs_dead"]["model"] = False
        elif fault == "rss":
            rows[2]["memory"]["rss_bytes"] = 0
        else:
            rows[2]["memory"]["before"]["flags"] = 4
        dump(path, rows)
    elif fault in ("deadline", "identity"):
        path = tmp_path / "serial-close.jsonl"
        rows = training.lines(path)
        if fault == "deadline":
            rows[-2]["monotonic_s"] = 103
        else:
            rows[6]["identity"]["start"] = "wrong"
        dump(path, rows)
    elif fault == "duplicate":
        before["workers"][2] = before["workers"][1]
    if fault == "none":
        result = training.validate_close(tmp_path, before)
        assert len(result["worlds"]) == 4 and result["close_to_join_s"] < 2
    else:
        with pytest.raises(ValueError):
            training.validate_close(tmp_path, before)


@pytest.mark.parametrize(
    "fault", ["none", "missing", "zero", "hold", "unreleased", "duplicate"]
)
def test_all_four_actual_retirements_required(fault):
    workers = [{"pid": 30 + i, "start": str(i)} for i in range(4)]
    facts = {"context": {"workers": [{"worker": w} for w in workers]}}
    watch = {
        "retirement_records": {
            f"{w['pid']}:{w['start']}": {
                "released": True,
                "highwater_charge_bytes": 100,
                "max_hold_s": 0.08,
            }
            for w in workers
        }
    }
    last = watch["retirement_records"]["33:3"]
    if fault == "missing":
        del watch["retirement_records"]["33:3"]
    elif fault == "zero":
        last["highwater_charge_bytes"] = 0
    elif fault == "hold":
        last["max_hold_s"] = 0.10001
    elif fault == "unreleased":
        last["released"] = False
    elif fault == "duplicate":
        facts["context"]["workers"][3] = facts["context"]["workers"][2]
    if fault == "none":
        assert len(training.qualify_retirements(facts, watch)) == 4
    else:
        with pytest.raises(ValueError):
            training.qualify_retirements(facts, watch)


def test_worker_is_training_capable_and_releases_closed_graph(tmp_path, monkeypatch):
    from multiprocessing import util

    import gymnasium as gym
    import numpy as np

    from joxing import deferred_qualification, live_resource_release

    recorded = []
    monkeypatch.setattr(
        training,
        "worker_witness",
        lambda folder, server: Path(folder).mkdir(parents=True),
    )
    monkeypatch.setattr(training, "process_identity", lambda: {"pid": 99, "start": "1"})
    monkeypatch.setattr(util, "Finalize", lambda *a, **k: recorded.append((a, k)))
    original_release = live_resource_release.release_wrapper_class
    memory = {
        "rss_bytes": 2000,
        "before": {"pid": 99, "start": "1", "state": "R", "flags": 0},
        "after": {"pid": 99, "start": "1", "state": "R", "flags": 0},
    }
    monkeypatch.setattr(
        live_resource_release,
        "release_wrapper_class",
        lambda *a, **k: original_release(
            *a, read_memory=lambda b: copy.deepcopy(memory), **k
        ),
    )

    class Resource:
        pass

    class Core(gym.Env):
        def __init__(self):
            self.model, self.data = Resource(), Resource()
            self.model.nu = 700
            self.reset_rng = np.random.default_rng(1097)
            self.physics_total = 0
            self.preparation_total = 200

        def snapshot(self):
            return {"state": self.physics_total}

        def step(self, action):
            self.physics_total += 1
            return self.physics_total

        def close(self):
            pass

    captured = []

    def factory(*a, **k):
        captured.append((a, k))
        return gym.Wrapper(Core())

    monkeypatch.setattr(deferred_qualification, "instrumented_worker", factory)
    env = training.training_worker(
        "out",
        2,
        1097,
        1.0,
        folder=tmp_path,
        server={"pid": 50},
        ledger_path="ledger",
        binding="fixed",
        slot="legacyA",
    )
    before = env.context_training_state()
    assert env.step(np.zeros(138)) == 1
    after = env.context_training_state()
    assert (
        after["state"] != before["state"] and after["reset_rng"] == before["reset_rng"]
    )
    assert (
        captured[0][0] == ("out", 2, 1097, 1.0) and captured[0][1]["binding"] == "fixed"
    )
    assert recorded[0][1]["exitpriority"] == 5
    env.close()
    assert training.lines(tmp_path / "world-2/resource-release.jsonl")[-2][
        "weakrefs_dead"
    ] == {"env": True, "model": True, "data": True}
    with pytest.raises(RuntimeError):
        env.context_training_state()


@pytest.mark.parametrize("fault", [False, True])
def test_scoped_aliases_restore_after_success_or_fault(tmp_path, monkeypatch, fault):
    from stable_baselines3.common import vec_env

    from joxing import deferred_qualification

    events = []
    old_vec, old_worker = (
        vec_env.SubprocVecEnv,
        deferred_qualification.instrumented_worker,
    )

    class Proof:
        def __init__(self, folder):
            self.server = {"pid": 50}

        def prepare(self):
            events.append("prepare")

    monkeypatch.setattr(training, "NativeContextProof", Proof)
    monkeypatch.setattr(training, "wait_ready", lambda *a: events.append("ready"))
    monkeypatch.setattr(training, "context_vector_class", lambda *a: "explicit-adapter")

    def run(*args):
        events.append("run")
        assert vec_env.SubprocVecEnv == "explicit-adapter"
        assert (
            deferred_qualification.instrumented_worker.func is training.training_worker
        )
        if fault:
            raise RuntimeError("retained learner fault")
        (tmp_path / "training-context-result.json").write_text(
            json.dumps({"execution_profile": training.PROFILE, "worlds": [1, 2, 3, 4]})
        )

    monkeypatch.setattr(deferred_qualification, "run_slot", run)
    if fault:
        with pytest.raises(RuntimeError, match="learner fault"):
            training.run_slot_context(
                tmp_path, {}, packet(), "ledger", "binding", "ready"
            )
        assert (tmp_path / "training-context-fault.json").exists()
    else:
        training.run_slot_context(tmp_path, {}, packet(), "ledger", "binding", "ready")
    assert (
        vec_env.SubprocVecEnv is old_vec
        and deferred_qualification.instrumented_worker is old_worker
    )
    assert events == ["ready", "prepare", "run"]


@pytest.mark.parametrize("fault", ["none", "method", "count", "spec", "state"])
def test_vector_keeps_cold_gate_and_persists_trained_state_before_send(
    tmp_path, monkeypatch, fault
):
    p = packet()
    workers = [{"worker": {"pid": 50 + i, "start": str(i)}} for i in range(4)]
    events = []

    class FakeSerial:
        def __init__(self, factories, *, start_method):
            assert start_method == "forkserver"
            self.closed = False
            self.trained = False

        def env_method(self, name):
            if name == "specification":
                return [p["expected_specification"] if fault != "spec" else {}] * 4
            state = "trained" if self.trained else p["expected_state_hash"]
            return [
                {"muscles": 700, "state": "wrong" if fault == "state" else state}
            ] * 4

        def close(self):
            before = json.loads((tmp_path / "trained-before-close.json").read_text())
            assert before["states"][0]["state"] == "trained"
            events.append("persisted_before_send")
            self.closed = True

    monkeypatch.setattr(training, "serial_vec_class", lambda *a: FakeSerial)
    monkeypatch.setattr(training, "observe_four", lambda *a: {"workers": workers})
    monkeypatch.setattr(training, "validate_close", lambda *a: {"worlds": [1, 2, 3, 4]})
    vector = training.context_vector_class(object, None, tmp_path, p)
    if fault == "none":
        env = vector([1] * 4, start_method="spawn")
        env.trained = True
        env.close()
        env.close()
        assert events == ["persisted_before_send"]
        assert (tmp_path / "training-context-result.json").exists()
    else:
        with pytest.raises(ValueError):
            vector(
                [1] * (3 if fault == "count" else 4),
                start_method="fork" if fault == "method" else "spawn",
            )
