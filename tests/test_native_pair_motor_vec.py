import numpy as np
import pytest

from joxing.native_pair_motor_vec import NativePairMotorVecEnv


class FakePair:
    actor_dimension = 3

    def __init__(self):
        self.resets = []
        self.steps = 0
        self.tick = 0

    def reset(self, *, seed):
        self.resets.append(seed)
        self.tick = 0
        return {
            "A": np.array([seed, 0, 1], np.float32),
            "B": np.array([seed, 0, 2], np.float32),
        }, {"episode_id": str(seed)}

    def observations(self):
        return {
            p: np.array([self.resets[-1], self.tick, i + 1], np.float32)
            for i, p in enumerate("AB")
        }

    def step(self, actions):
        self.steps += 1
        self.tick += 20
        assert set(actions) == set("AB")
        return (
            {
                "A": np.array([1, self.tick, 1], np.float32),
                "B": np.array([2, self.tick, 2], np.float32),
            },
            {"A": 1.0, "B": 2.0},
            False,
            True,
            {"episode_id": str(self.resets[-1])},
        )

    def close(self):
        pass


def test_pair_vec_one_world_joint_reset_real_terminal_observations():
    w = FakePair()
    v = NativePairMotorVecEnv(w, seed=10)
    v.seed(20)
    o = v.reset()
    assert o.shape == (2, 3) and w.resets == [20]
    v.step_async(np.zeros((2, 138), np.float32))
    assert w.steps == 0
    o, r, d, info = v.step_wait()
    assert w.steps == 1 and w.resets == [20, 21] and np.all(d) and r.tolist() == [1, 2]
    assert info[0]["terminal_observation"].tolist() == [1, 20, 1] and info[1][
        "terminal_observation"
    ].tolist() == [2, 20, 2]
    assert o[:, 0].tolist() == [21, 21] and all(x["TimeLimit.truncated"] for x in info)
    assert (
        info[0]["joint_world_episode_id"] == info[1]["joint_world_episode_id"] == "20"
    )
    assert not info[0]["independent_games"] and not info[1]["independent_games"]
    with pytest.raises(RuntimeError):
        v.step_wait()


@pytest.mark.parametrize(
    "bad", [np.zeros((1, 138)), np.full((2, 138), np.nan), np.full((2, 138), 1.1)]
)
def test_pair_vec_bad_actions_do_not_step_or_queue(bad):
    w = FakePair()
    v = NativePairMotorVecEnv(w)
    v.reset()
    with pytest.raises(ValueError):
        v.step_async(bad)
    assert not w.steps and v.pending is None


def test_pair_vec_no_reset_over_pending_or_mutation_one_corner():
    v = NativePairMotorVecEnv(FakePair())
    v.reset()
    v.step_async(np.zeros((2, 138)))
    with pytest.raises(RuntimeError):
        v.reset()
    with pytest.raises(RuntimeError):
        v.step_async(np.zeros((2, 138)))
    with pytest.raises(RuntimeError):
        v.set_attr("tick", 0, indices=0)
    with pytest.raises(RuntimeError):
        v.env_method("reset")


def test_paired_ppo_fixture_updates_both_slots_not_native_proof():
    from stable_baselines3 import PPO

    from joxing.native_pair_motor_train import actor_hash

    w = FakePair()
    v = NativePairMotorVecEnv(w, seed=1)
    model = PPO(
        "MlpPolicy",
        v,
        n_steps=8,
        batch_size=16,
        n_epochs=1,
        seed=1,
        device="cpu",
        policy_kwargs={"net_arch": [16, 16]},
        verbose=0,
    )
    before = actor_hash(model.policy)
    model.learn(total_timesteps=32)
    assert model.num_timesteps == 32 and w.steps == 16 and model._n_updates == 2
    assert before != actor_hash(model.policy)
