"""Fail-closed ownership, constructor causality and no-training controls."""

import contextlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from joxing import trial_memory as tm


def fake_proc(root, start=123):
    folder = root / "17"
    folder.mkdir(exist_ok=True)
    fields = ["S", "2"] + ["0"] * 17 + [str(start)]
    (folder / "stat").write_text("17 (name with ) brackets) " + " ".join(fields))
    (folder / "status").write_text("VmRSS:\t100 kB\n")
    (folder / "smaps_rollup").write_text(
        "Pss: 80 kB\nPrivate_Clean: 10 kB\nPrivate_Dirty: 20 kB\nPrivate_Hugetlb: 0 kB\n"
    )
    (folder / "maps").write_text("/usr/lib/libtorch_cpu.so\n")
    return {"pid": 17, "parent_pid": 2, "start_ticks": start, "state": "S"}


def event(phase, role="learner", pid=17, start=123):
    return {
        "identity": {"pid": pid, "start_ticks": start},
        "trial": "A",
        "role": role,
        "phase": phase,
    }


def test_memory_sample_and_pid_reuse(tmp_path):
    bound = fake_proc(tmp_path)
    sample = tm.memory_sample(bound, tmp_path)
    assert sample["rss_bytes"] == 102400
    assert sample["pss_bytes"] == 81920 and sample["private_bytes"] == 30720
    assert sample["libtorch_mapped"]
    fake_proc(tmp_path, start=124)
    assert tm.memory_sample(bound, tmp_path) is None


def test_missing_process_has_no_sample(tmp_path):
    assert tm.memory_sample({"pid": 17, "start_ticks": 123}, tmp_path) is None


def test_phase_order_and_known_roles():
    rows = [
        event(p)
        for p in (
            "evaluation_env_before",
            "evaluation_env_after",
            "vector_before",
            "vector_after",
            "sac_before",
            "sac_after",
            "learn_ready",
        )
    ]
    assert (
        tm.validate_events(rows, [{"pid": 17, "start_ticks": 123}])["17:123"]["role"]
        == "learner"
    )


@pytest.mark.parametrize(
    "rows,owned",
    [
        ([event("learn_ready")], [{"pid": 17, "start_ticks": 123}]),
        ([event("imports_before", role="foreign")], [{"pid": 17, "start_ticks": 123}]),
        ([event("imports_before")], [{"pid": 17, "start_ticks": 124}]),
        ([event("imports_before")], []),
        (
            [event("imports_before"), event("world_before", role="world-0")],
            [{"pid": 17, "start_ticks": 123}],
        ),
        ([event("guard_violation")], [{"pid": 17, "start_ticks": 123}]),
    ],
)
def test_unbound_reused_or_impossible_events_rejected(rows, owned):
    with pytest.raises(ValueError):
        tm.validate_events(rows, owned)


def test_guard_prevents_actual_side_effect_and_restores(monkeypatch, tmp_path):
    journal = []
    monkeypatch.setattr(tm, "emit", lambda *args, **kw: journal.append((args, kw)))
    executed = []

    class Optimizer:
        def step(self):
            executed.append("base")

    class Adam(Optimizer):
        def step(self):
            executed.append("adam")

    class Actor:
        def predict(self):
            executed.append("predict")

        forward = predict
        _predict = predict
        action_log_prob = predict

    class Env:
        def step(self):
            executed.append("physics")

    torch = SimpleNamespace(optim=SimpleNamespace(Optimizer=Optimizer, Adam=Adam))
    with contextlib.ExitStack() as stack:
        tm.install_guards(stack, tmp_path, "A", "learner", torch, Actor, Env)
        for operation in [
            Optimizer().step,
            Adam().step,
            Actor().predict,
            Actor().forward,
            Actor()._predict,
            Actor().action_log_prob,
            Env().step,
        ]:
            with pytest.raises(tm.GuardViolation):
                operation()
    assert not executed and len(journal) == 7
    Adam().step()
    assert executed == ["adam"]


def test_interrupted_event_tail_not_invented(tmp_path):
    row = dict(monotonic_s=1, **event("imports_before"))
    (tmp_path / "events-17-123.jsonl").write_text(json.dumps(row) + '\n{"partial"')
    assert tm.read_events(tmp_path) == [row]


def test_bounded_subprocess_profile_exit_runs_cleanup(tmp_path):
    script = """
import json,sys
from pathlib import Path
from joxing.trial_memory import ProfileExit, GuardViolation
events=[]
try:
    try:
        events.append('constructor_ready')
        raise ProfileExit()
    except Exception:
        events.append('fault_checkpoint')
    finally:
        events.append('close_vector')
except ProfileExit:
    events.append('stopped_before_training')
Path(sys.argv[1]).write_text(json.dumps(events))
assert not issubclass(GuardViolation,Exception)
"""
    result = tmp_path / "cleanup.json"
    subprocess.run([sys.executable, "-c", script, str(result)], check=True, timeout=10)
    assert json.loads(result.read_text()) == [
        "constructor_ready",
        "close_vector",
        "stopped_before_training",
    ]


def test_partial_constructor_counts_are_lower_bounds():
    rows = [
        dict(
            event("evaluation_env_after"),
            muscles=700,
            preparation_substeps=500,
            physics_substeps=0,
        )
    ]
    result = tm.profile_denominator(rows)
    assert result["completed_worlds"] == 1 and not result["all_worlds_constructed"]
    assert result["known_preparation_substeps_lower_bound"] == 500
    assert result["interrupted_constructor_work"] == "unknown"
    with pytest.raises(ValueError):
        tm.profile_denominator(rows + rows)


def test_missing_live_memory_never_silently_counts_as_zero():
    rows = [{"pid": 17, "start_ticks": 123}]
    with pytest.raises(ValueError, match="incomplete"):
        tm.sample_owned(rows, sample=lambda _: None, is_live=lambda _: True)
    assert tm.sample_owned(rows, sample=lambda _: None, is_live=lambda _: False) == []


def test_original_constructor_arguments_and_learn_interception(monkeypatch, tmp_path):
    from stable_baselines3.common import vec_env

    from joxing import wave

    calls, journal = [], []

    class Env:
        def __init__(self, **kwargs):
            calls.append(("env", kwargs))
            self.model = SimpleNamespace(nu=700)
            self.preparation_total, self.physics_total = 500, 0

        def specification(self):
            return {"model_hash": "fixture-only"}

        def step(self, *args):
            pytest.fail("Physics step executed")

    class Vec:
        def __init__(self, workers, **kwargs):
            calls.append(("vec", len(workers), kwargs))
            self.processes = [None] * len(workers)

        def close(self):
            calls.append(("close",))

    class SAC:
        def __init__(self, *args, **kwargs):
            calls.append(("sac", args[0], kwargs))
            self._n_updates = 0
            self.actor = SimpleNamespace(
                save=lambda _: calls.append(("initial_actor_save",))
            )

        def learn(self, *args, **kwargs):
            pytest.fail("Original learning executed")

    monkeypatch.setattr(wave, "StanceEnv", Env)
    monkeypatch.setattr(vec_env, "SubprocVecEnv", Vec)
    monkeypatch.setattr(wave, "timed_sac_class", lambda: SAC)
    monkeypatch.setattr(wave, "canonical_actor_hash", lambda _: "fixture-actor")
    monkeypatch.setattr(tm, "emit", lambda *args, **kw: journal.append((args[3], kw)))
    config = json.loads(Path("experiments/v0.0.5/protocol.json").read_text())
    (tmp_path / "release").touch()
    (tmp_path / "fixtures" / "R-A").mkdir(parents=True)
    (tmp_path / "fixtures" / "R-A" / "initial-actor.pt").write_bytes(b"fixture-only")
    with pytest.raises(tm.ProfileExit), tm.TrialMemoryProfile(tmp_path, "A"):
        wave.train_run(
            tmp_path / "R-A",
            config,
            "A",
            1097,
            [67010],
            {},
            lambda _: None,
            defer_initial_evaluation=True,
        )
    assert calls[0] == ("env", {"training": False})
    assert calls[1] == ("vec", 4, {"start_method": "spawn"})
    assert calls[2][2]["seed"] == 1097 and calls[2][2]["ent_coef"] == "auto"
    assert calls[-2:] == [("initial_actor_save",), ("close",)]
    assert [phase for phase, _ in journal][-1] == "learn_ready"
    assert wave.StanceEnv is Env and vec_env.SubprocVecEnv is Vec
