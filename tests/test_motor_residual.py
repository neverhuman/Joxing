"""Residual ownership, neutral physics, pending replay and latent SAC objective."""

import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from joxing.motor_residual import (
    MotorResidualEnv,
    decode_command,
    episode_masks,
    expand_groups,
    physical_reward,
    validate_groups,
)
from joxing.native_transfer import predict
from joxing.physical_motor import PhysicalMotorTask, protocol
from joxing.provenance import digest
from joxing.wave import canonical_actor_hash

torch = pytest.importorskip("torch")


def groups():
    return [a.tolist() for a in np.array_split(np.arange(700), 138)]


class Normalizer:
    fingerprint = "synthetic-frozen-normalizer"
    normalizer = SimpleNamespace(clip_obs=10)

    def apply(self, x):
        assert np.asarray(x).shape == (3601,)
        return np.clip(
            np.asarray(x, dtype=np.float32) * np.float32(0.01), -10, 10
        ).copy()

    def check(self):
        assert self.fingerprint == "synthetic-frozen-normalizer"


class Teacher:
    def __init__(self):
        self.actor = torch.nn.Linear(1, 1)
        self.actor.dynsyn_layer = SimpleNamespace(dynsyn_weight_amp=0.075)
        self.calls = 0

    def predict(self, x, deterministic):
        assert deterministic
        self.calls += 1
        return np.tanh(np.resize(x, 700) * np.float32(0.001)).astype(np.float32), None


def environment():
    p = protocol()
    return MotorResidualEnv(
        PhysicalMotorTask(p["environment"], p["variants"][0]),
        Teacher(),
        Normalizer(),
        groups(),
    )


def test_group_partition_owned_decode_and_independent_channels():
    g = groups()
    assert len(validate_groups(g)) == 138
    u = np.zeros(138, dtype=np.float32)
    u[0], u[-1] = 1, -1
    expanded = expand_groups(u, g)
    assert np.all(expanded[g[0]] == 1) and np.all(expanded[g[-1]] == -1)
    assert np.all(expanded[g[1]] == 0)
    teacher = np.full(700, 0.95, dtype=np.float32)
    result = decode_command(teacher, u, g)
    assert result.dtype == np.float32 and np.all(result[g[0]] == 1)
    u[:] = 0
    teacher[:] = 0
    assert np.all(result[g[-1]] > 0.84)
    for bad in [g[:-1], [[0]] * 138, [[True]] * 138]:
        with pytest.raises(ValueError):
            validate_groups(bad)
    for bad in [np.zeros(700), np.full(138, np.nan), np.full(138, 2)]:
        with pytest.raises(ValueError):
            expand_groups(bad, g)
    with pytest.raises(ValueError, match="amplitude"):
        decode_command(np.zeros(700), np.zeros(138), g, amplitude=0.2)


def test_zero_residual_exact_engine_cached_features_contacts_and_native_reward():
    p = protocol()
    baseline = PhysicalMotorTask(p["environment"], p["variants"][0])
    env = environment()
    try:
        baseline.reset(79001)
        env.reset(seed=79001)
        assert baseline.model.nu == env.task.model.nu == 700
        assert digest(baseline.snapshot()) == digest(env.task.snapshot())
        for _ in range(3):
            command = predict(env.teacher, env.normalizer, baseline.last_obs)[0]
            source = baseline.step(command)
            actual = env.step(np.zeros(138, dtype=np.float32))
            assert digest(baseline.snapshot()) == digest(env.task.snapshot())
            assert baseline.interval_evidence == env.task.interval_evidence
            assert source[1] == actual[4]["native_reward"]
            assert np.array_equal(actual[0], env.normalizer.apply(source[0]))
            assert actual[1] == pytest.approx(0.02)
        assert baseline.physics_total == env.task.physics_total == 30
    finally:
        baseline.close()
        env.close()


def test_mid_control_restores_without_teacher_decision_and_rejects_corruption():
    env = environment()
    try:
        env.reset(seed=79002)
        env.begin_control(np.full(138, 0.2, dtype=np.float32))
        env.advance_substeps(3)
        snapshot = env.snapshot()
        calls = env.teacher.calls
        first = env.resume_control()
        final = env.snapshot()
        env.restore(snapshot)
        second = env.resume_control()
        assert env.teacher.calls == calls
        assert np.array_equal(first[0], second[0]) and first[1:] == second[1:]
        assert digest(final) == digest(env.snapshot())
        before = digest(env.snapshot())
        bad = copy.deepcopy(snapshot)
        bad["held"]["residual"][0] = -0.5
        with pytest.raises(ValueError, match="ownership"):
            env.restore(bad)
        assert digest(env.snapshot()) == before
        bad = copy.deepcopy(snapshot)
        bad["bundle_hash"] = "another-reference-or-normalizer"
        with pytest.raises(ValueError, match="bundle"):
            env.restore(bad)
        assert digest(env.snapshot()) == before
        env.reset(seed=79003)
        assert env._held is None and not env.task.pending
    finally:
        env.close()


def test_assistance_and_decoder_mutation_rejected_without_new_physics():
    env = environment()
    try:
        env.reset(seed=79004)
        env.task.data.xfrc_applied[1, 0] = 1
        with pytest.raises(RuntimeError, match="assistance"):
            env.step(np.zeros(138, dtype=np.float32))
        assert env.task.physics_total == 0
        env.task.data.xfrc_applied[:] = 0
        env.teacher.actor.dynsyn_layer.dynsyn_weight_amp = 0
        with pytest.raises(RuntimeError, match="amplitude"):
            env.reset(seed=79004)
        assert env.task.physics_total == 0
    finally:
        env.close()


def test_first_latch_reward_and_distinct_terminal_time_limit_masks():
    reward, alive = physical_reward(1.0, 1.02, 1.006)
    assert alive == pytest.approx(0.006) and reward == pytest.approx(-0.994)
    assert episode_masks(True, False) == (0, 0)
    assert episode_masks(False, True) == (1, 0)
    assert episode_masks(False, False) == (1, 1)
    with pytest.raises(ValueError):
        physical_reward(1, 1.02, 1.5)


def test_physical_terminal_observation_survives_reset_and_never_bootstraps():
    env = environment()
    try:
        env.reset(seed=79005)
        for _ in range(1001):
            obs, _, terminated, truncated, info = env.step(np.zeros(138, np.float32))
            if terminated or truncated:
                break
        else:
            pytest.fail("Fixture exceeded registered20s physical horizon")
        terminal = info["terminal_observation"].copy()
        assert np.array_equal(terminal, obs)
        assert info["bootstrap_mask"] == float(not terminated)
        assert info["trace_mask"] == 0
        env.reset(seed=79006)
        assert np.array_equal(terminal, info["terminal_observation"])
        assert not np.shares_memory(terminal, env.task.last_obs)
    finally:
        env.close()


class SyntheticEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (3601,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (138,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3601, np.float32), {}

    def step(self, action):
        return np.zeros(3601, np.float32), 0.02, False, False, {}


def test_latent138_sac_finite_update_replay_target_and_roundtrip(tmp_path):
    from stable_baselines3 import SAC

    torch.set_num_threads(1)
    model = SAC(
        "MlpPolicy",
        SyntheticEnv(),
        seed=79111,
        device="cpu",
        learning_rate=1e-4,
        buffer_size=16384,
        learning_starts=1024,
        batch_size=128,
        ent_coef=1e-4,
        gamma=0.99,
        tau=0.005,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
    )
    with torch.no_grad():
        model.actor.mu.weight.zero_()
        model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(-5)
    probe = np.linspace(-0.1, 0.1, 3601, dtype=np.float32)
    assert np.array_equal(
        model.predict(probe, deterministic=True)[0], np.zeros(138, np.float32)
    )
    model._setup_learn(0)
    rng = np.random.default_rng(79112)
    for _ in range(128):
        obs = rng.normal(size=(1, 3601)).astype(np.float32)
        latent = rng.uniform(-1, 1, (1, 138)).astype(np.float32)
        model.replay_buffer.add(
            obs, obs * 0.9, latent, np.array([0.02]), np.array([0]), [{}]
        )
    assert model.replay_buffer.actions.shape[-1] == 138
    before = canonical_actor_hash(model.actor)
    target = canonical_actor_hash(model.critic_target)
    sample, logp = model.actor.action_log_prob(torch.zeros((8, 3601)))
    assert sample.shape == (8, 138) and logp.shape == (8,)
    assert torch.isfinite(sample).all() and torch.isfinite(logp).all()
    model.train(gradient_steps=1, batch_size=128)
    assert model._n_updates == 1 and canonical_actor_hash(model.actor) != before
    assert canonical_actor_hash(model.critic_target) != target
    assert all(
        torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all())
        for p in model.actor.parameters()
    )
    model.save(tmp_path / "residual.zip")
    loaded = SAC.load(tmp_path / "residual.zip", device="cpu")
    assert canonical_actor_hash(loaded.actor) == canonical_actor_hash(model.actor)
    assert np.array_equal(
        model.predict(probe, deterministic=True)[0],
        loaded.predict(probe, deterministic=True)[0],
    )
