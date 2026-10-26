"""Optional motor-extra integration: real updates and time-limit replay semantics."""

import numpy as np
import pytest

sb3 = pytest.importorskip(
    "stable_baselines3", reason="install the optional motor extra"
)

from joxing.balance import BalanceConfig
from joxing.stance_env import StanceEnv


def test_sac_updates_reloads_and_bootstraps_external_horizons(tmp_path):
    import torch

    torch.set_num_threads(1)
    env = StanceEnv(BalanceConfig(episode_s=0.02))
    model = sb3.SAC(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [16, 16]},
        buffer_size=32,
        learning_starts=0,
        batch_size=4,
        seed=8,
        device="cpu",
    )
    before = [p.detach().clone() for p in model.actor.parameters()]
    model.learn(12)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.actor.parameters()))
    assert model.replay_buffer.timeouts[:12].sum() == 6
    assert model.replay_buffer.dones[:12].sum() == 6
    # Terminal next-observation is preserved; reset-relative qpos is not substituted.
    ending = np.flatnonzero(model.replay_buffer.timeouts[:12, 0])
    assert all(np.any(model.replay_buffer.next_observations[i, 0, :85]) for i in ending)
    obs, _ = env.reset(options={"scenario_id": 30000})
    expected, _ = model.predict(obs, deterministic=True)
    model.save(tmp_path / "policy.zip")
    loaded = sb3.SAC.load(tmp_path / "policy.zip", device="cpu")
    actual, _ = loaded.predict(obs, deterministic=True)
    assert actual == pytest.approx(expected, abs=1e-7)
