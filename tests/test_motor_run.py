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
