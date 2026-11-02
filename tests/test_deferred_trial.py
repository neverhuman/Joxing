"""Allocation order and physical/fault contract controls, before native parity."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from joxing import deferred_trial as d
from joxing import wave


def test_legacy_body_unchanged_and_default_explicit():
    old = subprocess.check_output(
        ["git", "show", "39b3d86:src/joxing/wave.py"], text=True
    )

    def function(raw):
        return next(
            n
            for n in ast.parse(raw).body
            if isinstance(n, ast.FunctionDef) and n.name == "train_run"
        )

    previous = function(old)
    current = function(Path(wave.__file__).read_text())
    assert ast.dump(ast.Module(body=current.body[1:], type_ignores=[])) == ast.dump(
        ast.Module(body=previous.body, type_ignores=[])
    )
    assert current.args.kwonlyargs[-1].arg == "allocation_lifecycle"
    assert ast.literal_eval(current.args.kw_defaults[-1]) == "legacy"


def test_unknown_or_nondeferred_variant_rejected_before_output(tmp_path):
    for variant in ("typo", "deferred-evaluation-v1"):
        with pytest.raises(ValueError):
            wave.train_run(
                tmp_path / "never",
                {},
                "A",
                1097,
                [67010],
                {},
                lambda _: None,
                allocation_lifecycle=variant,
            )
        assert not (tmp_path / "never").exists()


@pytest.fixture
def fixtures(monkeypatch):
    from stable_baselines3.common import vec_env
    from stable_baselines3.sac.policies import Actor

    calls = []
    fault = {"phase": None}
    snapshots = [{"worker": i, "state": [i, 0.25]} for i in range(4)]

    class Vec:
        def __init__(self, workers, **kwargs):
            assert len(workers) == 4 and kwargs == {"start_method": "spawn"}
            calls.append("vector_constructed")
            self.processes = []
            self.closed = False
            if fault["phase"] == "construction":
                raise RuntimeError("constructor fault")

        def env_method(self, name):
            assert not self.closed, "Closed workers queried"
            return [{"fixed_spec": 700}] * 4 if name == "specification" else snapshots

        def get_attr(self, name):
            return [
                160
                if name == "physics_total"
                else 400
                if name == "preparation_total"
                else 0
            ] * 4

        def close(self):
            self.closed = True
            calls.append("workers_closed")

    class Env:
        def __init__(self, **kwargs):
            assert kwargs == {"training": False} and "workers_closed" in calls
            calls.append("evaluation_constructed")
            self.preparation_total = 200

        def specification(self):
            return {"fixed_spec": 700}

        def observe(self):
            return [0]

        def close(self):
            calls.append("evaluation_closed")
            if fault.get("cleanup"):
                raise RuntimeError("cleanup fault")

    class SavedActor:
        def __init__(self, tag="initial"):
            self.tag = tag

        def save(self, path):
            calls.append("save_" + self.tag)
            Path(path).write_bytes(self.tag.encode())

    class SAC:
        def __init__(self, *args, **kwargs):
            assert (
                kwargs["seed"] == 1097
                and kwargs["train_freq"] == kwargs["gradient_steps"] == 1
            )
            calls.append("sac_constructed")
            self.actor = SavedActor()
            self.num_timesteps = 0
            self.update_wall_s = self.inference_wall_s = 0
            self._n_updates = 0

        def learn(self, **kwargs):
            calls.append("learn")
            if fault["phase"] == "learning":
                raise RuntimeError("peer trial fault")
            self.num_timesteps = kwargs["total_timesteps"]
            self._n_updates = 12
            self.actor.tag = "final"

        def save(self, path):
            Path(path).write_bytes(b"fault-checkpoint")

    def evaluate(actor, env, ids, name, out, actor_hash):
        calls.append("evaluate_" + name)
        assert Path(out, "final-actor.pt").exists()
        if fault["phase"] == "evaluation":
            raise RuntimeError("evaluation fault")
        return [{"physics_steps": 1}]

    monkeypatch.setattr(vec_env, "SubprocVecEnv", Vec)
    monkeypatch.setattr(Actor, "load", lambda *a, **kw: SavedActor())
    monkeypatch.setattr(d, "StanceEnv", Env)
    monkeypatch.setattr(d, "timed_sac_class", lambda: SAC)
    monkeypatch.setattr(
        d,
        "canonical_actor_hash",
        lambda actor: hashlib.sha256(actor.tag.encode()).hexdigest(),
    )
    monkeypatch.setattr(d, "evaluate", evaluate)
    monkeypatch.setattr(d, "bootstrap_pair", lambda *a: {"attempts": 2})
    monkeypatch.setattr(d, "diagnostic", lambda *a: {"fixture": True})
    monkeypatch.setattr(d, "source_hash", lambda: "fixed")
    cfg = json.loads(Path("experiments/v0.0.5/protocol.json").read_text())
    cfg["learner"].update(steps=64, learning_starts=16)
    return cfg, calls, fault, snapshots


def run(tmp_path, fixture):
    cfg, *_ = fixture
    return wave.train_run(
        tmp_path / "trial",
        cfg,
        "A",
        1097,
        [67010],
        {"source_hash": "fixed"},
        lambda _: None,
        defer_initial_evaluation=True,
        allocation_lifecycle="deferred-evaluation-v1",
    )


def test_freeze_close_then_allocate_and_evaluate(tmp_path, fixtures):
    result = run(tmp_path, fixtures)
    _cfg, calls, _, states = fixtures
    assert calls == [
        "vector_constructed",
        "sac_constructed",
        "save_initial",
        "learn",
        "save_final",
        "workers_closed",
        "evaluation_constructed",
        "evaluate_initial",
        "evaluate_trained",
        "evaluation_closed",
    ]
    assert result["updates"] == 12 and result["training_transitions"] == 64
    assert result["allocation_lifecycle"] == "deferred-evaluation-v1"
    saved = json.loads((tmp_path / "trial/training-end-workers.json").read_text())
    assert saved == {"origin": "training-end", "states": states}


@pytest.mark.parametrize(
    "phase,origin,error",
    [
        ("learning", "live-workers", "peer trial fault"),
        ("evaluation", "training-end", "evaluation fault"),
        ("construction", "unavailable-partial-construction", "constructor fault"),
    ],
)
def test_all_faults_retain_real_state_origin(tmp_path, fixtures, phase, origin, error):
    fixtures[2]["phase"] = phase
    with pytest.raises(RuntimeError, match=error):
        run(tmp_path, fixtures)
    saved = json.loads((tmp_path / "trial/fault-states.json").read_text())
    assert saved["origin"] == origin
    assert saved["last_valid_workers"] == (
        [] if phase == "construction" else fixtures[3]
    )
    assert "workers_closed" in fixtures[1] or phase == "construction"


def test_partial_constructor_stops_only_new_owned_processes(monkeypatch):
    events = []

    class Process:
        def __init__(self, name):
            self.name = name
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False
            events.append(self.name)

        def join(self, timeout):
            pass

        def kill(self):
            pytest.fail("Unneeded kill")

    foreign, owned = Process("prior"), Process("owned")
    calls = iter([[foreign], [foreign, owned]])
    monkeypatch.setattr(d.multiprocessing, "active_children", lambda: next(calls))

    class Partial:
        def __init__(self):
            self.processes = [owned]
            raise RuntimeError("partial")

    owner = d.VectorOwner()
    with pytest.raises(RuntimeError, match="partial"):
        owner.construct(Partial)
    assert events == ["owned"] and foreign.alive and owner.closed


def test_evaluation_cleanup_fault_does_not_replace_original(tmp_path, fixtures):
    fixtures[2].update(phase="evaluation", cleanup=True)
    with pytest.raises(RuntimeError, match="evaluation fault"):
        run(tmp_path, fixtures)
    assert (
        json.loads((tmp_path / "trial/evaluation-cleanup-fault.json").read_text())[
            "error"
        ]
        == "cleanup fault"
    )
