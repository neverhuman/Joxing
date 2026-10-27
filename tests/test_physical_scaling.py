"""Scaling ownership, exact anchors and partial-step accounting."""

import json
import runpy
import time
from types import SimpleNamespace

import numpy as np

from joxing.provenance import ROOT, digest

helpers = runpy.run_path(str(ROOT / "scripts/benchmark_physical_scaling.py"))


def test_registered_schedule_includes_every_worker_attempt_and_fits_budget():
    p = json.loads((ROOT / "experiments/v0.0.9/scaling-protocol.json").read_text())
    assert digest(p) == helpers["PROTOCOL_SHA"]
    anchors, waves = helpers["schedule"](p)
    assert len(anchors) == 16 and sum(len(w["cases"]) for w in waves) == 62
    assert [w["workers"] for w in waves] == [1, 2, 4, 8, 16, 16, 8, 4, 2, 1]
    assert (16 + 62) * p["control_steps"] * 10 == 93600 <= p["caps"]["physics_steps"]
    ids = [r["case_id"] for r in anchors] + [
        r["case_id"] for w in waves for r in w["cases"]
    ]
    assert len(ids) == len(set(ids))


def fake_worker(monkeypatch, fail=False):
    import torch

    import joxing.wave

    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)

    class Task:
        def __init__(self, *args):
            self.physics_total = 0

        def reset(self, seed):
            self.seed, self.controls = seed, 0
            self.done = False
            self.last_obs = np.full(3601, seed * 1e-5, dtype=np.float32)

        def snapshot(self):
            return {"state": [self.seed, self.controls], "monitor": {"failure": None}}

        def step(self, action):
            assert action.shape == (700,)
            self.physics_total += 3 if fail else 10
            if fail:
                raise RuntimeError("Injected substep fault")
            self.controls += 1
            return self.last_obs, 1.0, False, False, {}, {"control": self.controls}

        def close(self):
            pass

    def check():
        pass

    globals_ = helpers["worker"].__globals__
    monkeypatch.setitem(
        globals_,
        "load_controller",
        lambda _: (
            SimpleNamespace(
                actor=None, policy=SimpleNamespace(set_training_mode=lambda _: None)
            ),
            SimpleNamespace(check=check, fingerprint="frozen"),
        ),
    )
    monkeypatch.setitem(globals_, "PhysicalMotorTask", Task)
    monkeypatch.setitem(
        globals_, "predict", lambda *args: (np.zeros(700, dtype=np.float32), 0.0)
    )
    monkeypatch.setattr(joxing.wave, "canonical_actor_hash", lambda _: "actor")
    return helpers["worker"]


def test_worker_hash_anchors_are_seed_stable_and_order_independent(
    monkeypatch, tmp_path
):
    worker = fake_worker(monkeypatch)
    p = {"control_steps": 3, "progress_controls": 1}
    results = []
    for i, seeds in enumerate(([77000, 77001], [77001, 77000])):
        rows = [{"case_id": str(s), "seed": s, "status": "not_run"} for s in seeds]
        result = worker(
            tmp_path,
            p,
            rows,
            tmp_path / f"{i}.json",
            tmp_path / "stop.json",
            time.monotonic() + 30,
        )
        assert result["phase"] == "completed" and result["physics_steps"] == 60
        results.append({r["seed"]: helpers["fingerprint"](r) for r in result["cases"]})
    assert results[0] == results[1]
    assert results[0][77000] != results[0][77001]


def test_substep_fault_retains_attempted_physics_and_future_denominator(
    monkeypatch, tmp_path
):
    worker = fake_worker(monkeypatch, fail=True)
    rows = [{"case_id": str(s), "seed": s, "status": "not_run"} for s in [77000, 77001]]
    result = worker(
        tmp_path,
        {"control_steps": 3, "progress_controls": 1},
        rows,
        tmp_path / "progress.json",
        tmp_path / "stop.json",
        time.monotonic() + 30,
    )
    assert result["phase"] == "failed" and result["physics_steps"] == 3
    assert (
        result["cases"][0]["status"] == "invalid"
        and result["cases"][0]["physics_steps"] == 3
    )
    assert result["cases"][1]["status"] == "not_run"
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "failed"


def test_parent_stop_prevents_any_physics_and_retains_pending_cases(
    monkeypatch, tmp_path
):
    worker = fake_worker(monkeypatch)
    (tmp_path / "stop.json").write_text("{}")
    rows = [{"case_id": "a", "seed": 77000, "status": "not_run"}]
    result = worker(
        tmp_path,
        {"control_steps": 3, "progress_controls": 1},
        rows,
        tmp_path / "progress.json",
        tmp_path / "stop.json",
        time.monotonic() + 30,
    )
    assert result["phase"] == "failed" and result["physics_steps"] == 0
    assert result["cases"][0]["status"] == "incomplete"
