"""D2 seed isolation, real worker receipts/faults and vector SAC semantics."""

import json
import time

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("torch", reason="install the optional motor extra")
pytest.importorskip("stable_baselines3", reason="install the optional motor extra")

from joxing.motor_pilot import (
    EVALUATION_SEEDS,
    PilotWorker,
    make_learner,
    paired_conclusion,
    planned_rows,
    reset_seed,
)


def test_reset_streams_disjoint_and_exhaustion():
    streams = [
        {reset_seed(t, w, e) for e in range(25000)} for t in range(2) for w in range(4)
    ]
    assert len(set.union(*streams)) == 200000
    assert not set(EVALUATION_SEEDS) & set.union(*streams)
    assert reset_seed(0, 0, 0) == 8200000
    assert reset_seed(1, 3, 24999) == 8399999
    for args in [(0, 4, 0), (2, 0, 0), (0, 0, 25000), (0, 0, -1), (True, 0, 0)]:
        with pytest.raises(ValueError):
            reset_seed(*args)


def test_worker_owned_seeds_chunks_and_raw_fault(tmp_path):
    from test_motor_residual import environment

    worker = PilotWorker(
        environment(), tmp_path / "worker", 0, 0, time.monotonic() + 60
    )
    try:
        _, info = worker.reset(seed=81011)
        assert info["episode_seed"] == 8200000
        worker.step(np.zeros(138, np.float32))
        _, info = worker.reset()
        assert info["episode_seed"] == 8200004
        assert worker.env.task.physics_total == 10
        worker.deadline = time.monotonic() - 1
        with pytest.raises(TimeoutError):
            worker.step(np.zeros(138, np.float32))
    finally:
        worker.close()
    p = tmp_path / "worker"
    result = json.loads((p / "progress.json").read_text())
    assert result["status"] == "fault" and result["transitions"] == 1
    assert (
        result["physics_steps"] == 10 and result["maximum_physics_allocation"] == 20480
    )
    assert result["chunks"] == 1 and result["unflushed_transitions"] == 0
    events = [
        json.loads(line) for line in (p / "episodes.jsonl").read_text().splitlines()
    ]
    assert [r["seed"] for r in events] == [8200000, 8200004]
    with np.load(p / "commands-00000.npz", allow_pickle=False) as a:
        assert a["residuals"].shape == (1, 138) and a["commands"].shape == (1, 700)
        assert a["commands"].dtype == np.float32
    with np.load(p / "fault-arrays.npz", allow_pickle=False) as a:
        assert a["observation"].shape == (3601,) and np.isfinite(a["state"]).all()


class VectorFixture(gym.Env):
    """No physical measurements; terminal obs differs from autoreset obs."""

    def __init__(self, terminal):
        self.terminal = terminal
        self.action_space = gym.spaces.Box(-1, 1, (138,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3601,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3601, np.float32), {}

    def step(self, action):
        return (
            np.full(3601, 0.25, np.float32),
            0.02,
            self.terminal,
            not self.terminal,
            {},
        )


def test_four_worker_sac_latents_updates_timeouts_and_terminal_observation(tmp_path):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: VectorFixture(True), lambda: VectorFixture(False)] * 2)
    model = make_learner(env, 81011)
    assert model.learning_starts == 1024 and model.train_freq.frequency == 4
    assert model.n_envs == 4 and model.gradient_steps == 1
    obs = env.reset()
    initial = model.predict(obs, deterministic=True)[0]
    assert initial.shape == (4, 138) and np.array_equal(initial, np.zeros_like(initial))
    before = [p.detach().clone() for p in model.actor.parameters()]
    # Synthetic integration only; formal configuration and fresh run unaffected.
    model.learning_starts = 0
    model.batch_size = 4
    model.learn(total_timesteps=16)
    assert model.num_timesteps == 16 and model._n_updates == 1
    assert model.replay_buffer.actions[:4].shape == (4, 4, 138)
    assert np.all(model.replay_buffer.dones[:4] == 1)
    assert np.array_equal(
        model.replay_buffer.timeouts[:4], np.array([[0, 1, 0, 1]] * 4)
    )
    assert np.all(model.replay_buffer.next_observations[:4] == 0.25)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, model.actor.parameters(), strict=True)
    )
    model.save(tmp_path / "synthetic.zip")
    loaded = SAC.load(tmp_path / "synthetic.zip", device="cpu")
    assert np.array_equal(
        model.predict(obs, deterministic=True)[0],
        loaded.predict(obs, deterministic=True)[0],
    )
    env.close()


def test_paired_all_attempt_denominator_and_no_pooled_seed_win():
    rows = planned_rows()
    assert len(rows) == 32
    for row in rows:
        row.update(
            status="completed", survival_s=4.0 if row["stage"] == "initial" else 5.0
        )
    assert paired_conclusion(rows)["evaluation_directional_criteria_passed"]
    # One seed improving greatly must not hide the other seed's regression.
    for row in rows:
        if row["training_seed"] == 81029 and row["stage"] == "final":
            row["survival_s"] = 3.0
    assert not paired_conclusion(rows)["evaluation_directional_criteria_passed"]
    rows[0]["status"] = "invalid"
    result = paired_conclusion(rows)
    assert not result["seeds"][0]["all_evaluations_valid"]
    assert result["seeds"][0]["mean_paired_delta_s"] is None
    with pytest.raises(ValueError):
        paired_conclusion(rows[:-1])
