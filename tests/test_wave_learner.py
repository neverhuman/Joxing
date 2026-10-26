"""Real SB3 integration: RNG isolation, exports and actual replay updates."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3.common.logger import configure
from stable_baselines3.sac.policies import Actor

from joxing.stance_env import StanceEnv
from joxing.wave import canonical_actor_hash, diagnostic, timed_sac_class


def test_real_update_actor_export_seed_and_rng(tmp_path):
    torch.set_num_threads(1)
    env = StanceEnv()
    cls = timed_sac_class()
    a = cls(
        "MlpPolicy",
        env,
        seed=149,
        policy_kwargs={"net_arch": [16, 16]},
        batch_size=4,
        buffer_size=32,
        learning_starts=0,
        device="cpu",
    )
    b = cls(
        "MlpPolicy",
        env,
        seed=149,
        policy_kwargs={"net_arch": [16, 16]},
        batch_size=4,
        buffer_size=32,
        learning_starts=0,
        device="cpu",
        ent_coef=0.0001,
    )
    initial = canonical_actor_hash(a.actor)
    assert initial == canonical_actor_hash(b.actor)
    obs, _ = env.reset(options={"scenario_id": 62000})
    rng = torch.get_rng_state().clone()
    diagnostic(a, obs[None])
    assert torch.equal(torch.get_rng_state(), rng)
    for _ in range(8):
        action, _ = a.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        a.replay_buffer.add(
            obs[None],
            next_obs[None],
            action[None],
            np.array([reward]),
            np.array([terminated or truncated]),
            [{"TimeLimit.truncated": truncated and not terminated}],
        )
        obs = next_obs
    # Explicit timeout transition: SB3 must keep bootstrap through truncation.
    a.replay_buffer.add(
        obs[None],
        obs[None],
        action[None],
        np.array([0.0]),
        np.array([True]),
        [{"TimeLimit.truncated": True}],
    )
    assert a.replay_buffer.timeouts[8, 0] == 1
    sample = a.replay_buffer._get_samples(np.array([8]))
    assert sample.dones.item() == 0
    a.set_logger(configure(str(tmp_path), []))
    a.train(2, 4)
    assert a._n_updates == 2 and a.update_wall_s > 0
    assert canonical_actor_hash(a.actor) != initial
    a.actor.save(tmp_path / "actor.pt")
    restored = Actor.load(tmp_path / "actor.pt", device="cpu")
    assert canonical_actor_hash(restored) == canonical_actor_hash(a.actor)
    np.testing.assert_array_equal(
        a.actor.predict(obs, deterministic=True)[0],
        restored.predict(obs, deterministic=True)[0],
    )
