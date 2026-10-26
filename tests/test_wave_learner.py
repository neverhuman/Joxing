"""Real SB3 integration: RNG isolation, exports and actual replay updates."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3.common.logger import configure
from stable_baselines3.sac.policies import Actor

from joxing.stance_env import StanceEnv
from joxing.wave import canonical_actor_hash, diagnostic, timed_sac_class


def test_parallel_extension_trials_defer_initial_evaluation(tmp_path):
    import json
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from joxing.provenance import ROOT, manifest
    from joxing.wave_extension import child_trial, extension_config

    protocol = json.loads(
        (ROOT / "experiments/v0.0.5/extension-protocol.json").read_text()
    )
    config = extension_config(protocol)
    config["learner"].update(steps=64, learning_starts=16)
    config["caps"]["run_wall_s"] = 120
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        stop, counts = manager.Event(), manager.dict()
        jobs = [
            (
                str(tmp_path / f"R-{arm}"),
                f"R-{arm}",
                config,
                arm,
                1097,
                [67010],
                manifest(),
                stop,
                counts,
                0,
                1e12,
            )
            for arm in ("A", "B")
        ]
        with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
            results = list(pool.map(child_trial, jobs))
        assert results[0]["initial_actor_hash"] == results[1]["initial_actor_hash"]
        for arm, result in zip(("A", "B"), results):
            folder = tmp_path / f"R-{arm}"
            assert result["scheduled_training_complete"] and result["updates"] == 12
            assert (folder / "final-actor.pt").stat().st_mtime_ns <= (
                folder / "initial-replay.json"
            ).stat().st_mtime_ns
        stop.set()
        job = (
            str(tmp_path / "R-stop"),
            "R-stop",
            config,
            "B",
            1097,
            [67010],
            manifest(),
            stop,
            counts,
            0,
            1e12,
        )
        with pytest.raises(RuntimeError, match="peer trial fault"):
            child_trial(job)
        assert (tmp_path / "R-stop/fault.json").exists()
        assert (
            len(
                json.loads((tmp_path / "R-stop/fault-states.json").read_text())[
                    "last_valid_workers"
                ]
            )
            == 4
        )


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
