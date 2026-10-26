import copy
import json

import gymnasium as gym
import numpy as np
import pytest

from joxing.provenance import ROOT
from joxing.upstream_motor import (
    MODES,
    FrozenNormalizer,
    NativeTask,
    extract_archive,
    load_dynsyn,
    pinned_import,
    set_decoder_mode,
    summarize,
    validate_protocol,
)

torch = pytest.importorskip("torch")
sb3 = pytest.importorskip("stable_baselines3")


def protocol():
    return json.loads((ROOT / "experiments/v0.0.6/protocol.json").read_text())


class SmallEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([1, 2, 3], dtype=np.float32), {}

    def step(self, action):
        return np.array([2, 3, 4], dtype=np.float32), 0.0, False, False, {}


def test_registered_protocol_rejects_silent_retuning():
    p = protocol()
    assert validate_protocol(p) == p
    for block, key, value in [
        ("caps", "physics_steps", 2000000),
        ("environment", "kinematic_play", True),
        ("runtime", "torch", "2.14.1"),
        ("criteria", "same_host_replay_max_error", 1e-3),
    ]:
        altered = copy.deepcopy(p)
        altered[block][key] = value
        with pytest.raises(ValueError):
            validate_protocol(altered)


def test_release_archive_rejected_before_extraction(tmp_path):
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"not the release")
    with pytest.raises(ValueError, match="archive"):
        extract_archive(archive, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_real_normalizer_frozen_and_tampering_detected():
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec = VecNormalize(DummyVecEnv([SmallEnv]), norm_obs=True)
    vec.reset()
    vec.step(np.zeros((1, 4)))
    n = FrozenNormalizer(vec, 3)
    for _ in range(10):
        n.apply(np.ones(3))
    n.check()
    with pytest.raises(ValueError):
        n.apply(np.full(3, np.nan))
    with pytest.raises(ValueError):
        n.apply(np.ones(4))
    vec.obs_rms.mean[0] += 1
    with pytest.raises(RuntimeError, match="changed"):
        n.check()
    vec.close()


def test_decoder_attribute_lost_by_default_but_explicit_bundle_roundtrips(tmp_path):
    from stable_baselines3 import SAC

    cls = pinned_import("DynSyn.SAC_DynSyn").SAC_DynSyn
    aliases = dict(SAC.policy_aliases)
    try:
        model = cls(
            "MlpPolicy",
            SmallEnv(),
            buffer_size=8,
            batch_size=2,
            dynsyn_k=5e-9,
            dynsyn_a=3e7,
            seed=7,
            policy_kwargs={
                "net_arch": [8, 8],
                "dynsyn": [[0, 1], [2, 3]],
                "dynsyn_log_std": 0.5,
            },
        )
    finally:
        SAC.policy_aliases.clear()
        SAC.policy_aliases.update(aliases)
    model.num_timesteps = 45000000
    with torch.no_grad():
        model.actor.mu.weight.zero_()
        model.actor.mu.bias.fill_(0.25)
        model.actor.dynsyn_layer.mu[0].weight.zero_()
        model.actor.dynsyn_layer.mu[0].bias.fill_(1)
    model.actor.dynsyn_layer.update_dynsyn_weight_amp(0.075)
    obs = np.zeros(3, dtype=np.float32)
    intended, _ = model.predict(obs, deterministic=True)
    model.save(tmp_path / "default.zip")
    loaded = load_dynsyn(tmp_path / "default.zip")
    assert SAC.policy_aliases == aliases
    assert loaded.actor.dynsyn_layer.dynsyn_weight_amp is None
    actual, _ = loaded.predict(obs, deterministic=True)
    assert np.max(abs(actual - intended)) > 1e-3
    assert set_decoder_mode(loaded, MODES[1]) == 0.075
    restored, _ = loaded.predict(obs, deterministic=True)
    assert np.array_equal(restored, intended)
    loaded.save(tmp_path / "corrected.zip")
    reloaded = load_dynsyn(tmp_path / "corrected.zip")
    assert reloaded.actor.dynsyn_layer.dynsyn_weight_amp == 0.075
    assert np.array_equal(reloaded.predict(obs, deterministic=True)[0], intended)
    assert SAC.policy_aliases == aliases


@pytest.fixture
def task():
    e = NativeTask(protocol()["environment"])
    e.reset(70001)
    yield e
    e.close()


def test_native_step_uses_muscle_dynamics_and_rejects_assistance(task):
    before = task.data.act.copy()
    task.env.set_state = lambda *args: pytest.fail("Per-step pose write")
    action = np.full(700, -0.7, dtype=np.float32)
    wrapper = pinned_import("wrapper.muscle_norm_wrapper").MuscleNormWrapper
    golden = wrapper.action(None, action)
    _, _, _, _, _, telemetry = task.step(action)
    assert np.array_equal(task.data.ctrl, golden)
    assert task.physics_total == 10
    assert task.data.time == pytest.approx(0.02)
    assert np.max(abs(task.data.act - before)) > 0
    assert np.isfinite(list(telemetry["floor_loads_N"].values())).all()
    task.env.kinematic_play = True
    with pytest.raises(RuntimeError, match="assistance"):
        task.step(np.zeros(700))


def test_native_snapshot_restores_cached_observation_and_feedback_trajectory(task):
    def action(obs):
        return np.tanh(np.resize(obs, 700) * 0.001)

    task.step(action(task.last_obs))
    snapshot = task.snapshot()
    actions = []
    for _ in range(8):
        if task.done:
            break
        a = action(task.last_obs)
        actions.append(a)
        task.step(a)
    expected = task.snapshot()
    obs = task.restore(snapshot)
    assert np.array_equal(
        obs, np.asarray(snapshot["last_raw_observation"], dtype=np.float32)
    )
    for _ in actions:
        task.step(action(task.last_obs))
    assert np.max(abs(np.array(task.snapshot()["state"]) - expected["state"])) < 1e-9
    assert np.array_equal(task.last_obs, expected["last_raw_observation"])
    task.restore(snapshot)
    for a in actions:
        task.step(a)
    assert np.max(abs(np.array(task.snapshot()["state"]) - expected["state"])) < 1e-9


def test_snapshot_rejects_contract_and_nonfinite_state(task):
    s = task.snapshot()
    bad = copy.deepcopy(s)
    bad["contract_hash"] = "other"
    with pytest.raises(ValueError):
        task.restore(bad)
    bad = copy.deepcopy(s)
    bad["state"][0] = np.nan
    with pytest.raises(ValueError):
        task.restore(bad)


def test_all_planned_attempts_stay_in_summary():
    rows = [
        {
            "attempt_id": f"{mode}:{seed}",
            "seed": seed,
            "mode": mode,
            "status": "not_run",
        }
        for seed in range(71000, 71020)
        for mode in MODES
    ]
    summary = summarize(rows)
    assert summary["modes"][MODES[0]]["not_run"] == 20
    assert not summary["native_task_completion_threshold_met"]
    rows[0]["status"] = "invalid"
    assert summarize(rows)["invalid"] == 1
    rows[0] = dict(rows[1])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows)
