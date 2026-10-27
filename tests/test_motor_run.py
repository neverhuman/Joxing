"""Parent foundations only; no formal pilot or physical outcome is claimed."""

import time

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from joxing.motor_pilot import make_learner
from joxing.motor_run import MeasuredSAC, ShapeOnly, close_pool, save_policy
from joxing.motor_storage import RawCredit, write_bytes


def test_prewrite_credit_preserves_file_and_fault_reserve(tmp_path):
    c = RawCredit(tmp_path, 4 * 1024**2)
    write_bytes(tmp_path / "existing", b"a" * (1800 * 1024))
    with pytest.raises(TimeoutError):
        write_bytes(tmp_path / "too-big", b"b" * (400 * 1024))
    assert not (tmp_path / "too-big").exists()
    assert (tmp_path / "existing").stat().st_size == 1800 * 1024
    with c.emergency():
        write_bytes(tmp_path / "fault", b"f" * (400 * 1024))
    assert c.used() == 2200 * 1024


def test_shape_slots_handoff_and_measured_checkpoint(tmp_path):
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from test_motor_pilot import VectorFixture

    torch.set_num_threads(1)
    slots = DummyVecEnv([ShapeOnly] * 4)
    with pytest.raises(RuntimeError, match="Shape-only"):
        slots.reset()
    model = make_learner(slots, 81011, MeasuredSAC)
    RawCredit(tmp_path, 128 * 1024**2)
    initial = save_policy(model, tmp_path, "initial", "fixture")
    vec = DummyVecEnv([lambda: VectorFixture(True)] * 4)
    try:
        model.set_env(vec, force_reset=True)
        model.learning_starts = 0
        model.batch_size = 4
        model.learn(16)
        assert model._n_updates == 1 and len(model.update_receipts) == 1
        final = save_policy(model, tmp_path, "final", "fixture")
        assert initial["actor_hash"] != final["actor_hash"]
        loaded = MeasuredSAC.load(tmp_path / "final.zip", device="cpu")
        obs = vec.reset()
        assert np.array_equal(
            model.predict(obs, deterministic=True)[0],
            loaded.predict(obs, deterministic=True)[0],
        )
    finally:
        vec.close()
        slots.close()


def test_subprocess_failure_cleanup_is_bounded(monkeypatch):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from test_motor_pilot import VectorFixture

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")

    class Broken(VectorFixture):
        def step(self, action):
            raise RuntimeError("injected worker fault")

    vec = SubprocVecEnv(
        [lambda: VectorFixture(True), lambda: Broken(True)], start_method="spawn"
    )
    began = time.monotonic()
    try:
        vec.reset()
        with pytest.raises((EOFError, ConnectionResetError, BrokenPipeError)):
            vec.step(np.zeros((2, 138), np.float32))
    finally:
        forced = close_pool(vec)
    assert not forced and all(not p.is_alive() for p in vec.processes)
    assert time.monotonic() - began < 15


def test_teacher_load_preserves_learner_rng_even_on_failure():
    import random

    import torch

    from joxing.motor_run import load_motor_env

    random.seed(81011)
    np.random.seed(81011)
    torch.manual_seed(81011)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()

    def noisy_loader(inputs):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        raise RuntimeError("injected teacher-load failure")

    with pytest.raises(RuntimeError):
        load_motor_env(None, noisy_loader)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_vector_collection_update_cadence_after_1024_warmup():
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from test_motor_pilot import VectorFixture

    class CountOnly(SAC):
        def train(self, gradient_steps, batch_size=64):
            self._n_updates += gradient_steps
            if not hasattr(self, "calls"):
                self.calls = []
            self.calls.append(self.num_timesteps)

    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: VectorFixture(True)] * 4)
    try:
        model = make_learner(env, 81011, CountOnly)
        model.learn(8192)
        assert model.num_timesteps == 8192 and model._n_updates == 448
        assert model.calls == list(range(1040, 8193, 16))
    finally:
        env.close()


def test_parent_keeps_failed_evaluation_attempt_and_actual_steps(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import joxing.motor_run as run

    p = {
        "prior": {"physics_steps": 10394, "formal_wall_s": 25.066974967718124},
        "raw_credits": {"parent_bytes": 192 * 1024**2, "evaluation_bytes": 8 * 1024**2},
        "caps": {
            "wall_s": 1800,
            "physics_steps": 550000,
            "minimum_memory_bytes": 0,
            "minimum_disk_bytes": 0,
        },
        "bundle_hash": "fixture",
    }
    env = SimpleNamespace(
        task=SimpleNamespace(physics_total=0, specification=dict, floor=0),
        normalizer=SimpleNamespace(export=dict),
        groups=[],
        contract={},
        contract_hash="fixture",
        close=lambda: None,
    )
    monkeypatch.setattr(run, "load_motor_env", lambda inputs: env)

    def broken(env, model, folder, *args):
        folder.mkdir(parents=True)
        env.task.physics_total += 3
        raise TimeoutError("injected raw output failure")

    monkeypatch.setattr(run, "collect", broken)
    parent = run.PilotRun(p, tmp_path, tmp_path / "run")
    folder = parent.out / "seed-81011"
    folder.mkdir()
    with pytest.raises(TimeoutError):
        parent.evaluation(None, 0, "initial", folder)
    row = parent.report["evaluations"][0]
    assert row["status"] == "incomplete" and row["physics_steps"] == 3
    assert (
        parent.main_physics == 10397
        and parent.report["evaluations"][1]["status"] == "not_run"
    )
    assert parent.active_env is None
