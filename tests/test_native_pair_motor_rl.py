import json
from dataclasses import asdict

import numpy as np
import pytest

from joxing.native_pair_motor_rl import NativePairMotorEnv, PairMotorConfig
from joxing.provenance import ROOT, digest
from joxing.scene import build_scene
from joxing.shared_motor_inputs import FrozenPairTeacher
from joxing.shared_motor_task import NativeReferenceDriver, SharedMotorTask
from joxing.upstream_motor import NativeTask


class FixtureNormalizer:
    def check(self):
        pass

    def apply(self, values):
        return values.copy()


class FixtureActor:
    def __init__(self):
        self.calls = []

    def predict(self, values, *, deterministic):
        assert deterministic
        self.calls.append(values.tobytes())
        v = (-0.2 - 0.01 * np.tanh(values[85:170])[np.arange(700) % 85]).astype(
            np.float32
        )
        v[0] = np.float32(-0.0)
        return v, None


@pytest.fixture(scope="module")
def factory():
    config = json.loads((ROOT / "experiments/v0.0.6/protocol.json").read_text())[
        "environment"
    ]
    donor = NativeTask(config)
    groups = json.loads(
        (ROOT / "third_party/msgym/DynSyn-SAC/configs/locomotionFull.json").read_text()
    )["agent_kwargs"]["policy_kwargs"]["dynsyn"]
    scenes = {
        p: build_scene(spawn_variant="parallel-walk-v1" if p == "parallel" else None)
        for p in ("parallel", "opposed")
    }

    def drivers(seed):
        return {p: NativeReferenceDriver(donor, seed) for p in "AB"}

    def make(*, placement="parallel", horizon=240):
        actor = FixtureActor()
        teacher = FrozenPairTeacher(actor, FixtureNormalizer())
        return NativePairMotorEnv(
            scenes[placement],
            drivers,
            teacher,
            groups,
            {"explicit-native-engine-synthetic-actor": "0" * 64},
            config=PairMotorConfig(placement=placement, horizon_ticks=horizon),
        ), actor

    yield make, donor, groups, scenes, drivers
    donor.close()


def contact_hash(c):
    return digest(
        [
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in asdict(x).items()
            }
            for x in c
        ]
    )


def zeros():
    return {p: np.zeros(138, np.float32) for p in "AB"}


def test_zero_residual_matches_unchanged_shared_native_task(factory):
    make, _, _, scenes, drivers = factory
    env, actor = make(horizon=60)
    obs, _ = env.reset(seed=71000)
    baseline_actor = FixtureActor()
    base = SharedMotorTask(
        scenes["parallel"],
        drivers(71000),
        FrozenPairTeacher(baseline_actor, FixtureNormalizer()),
        horizon_ticks=60,
        bundle_hash="fixture",
    )
    for t in range(60):
        if t % 20 == 0:
            env.begin_control(zeros())
            base.begin_control()
        a = env.advance_substep()
        b = base.advance_substep(lambda n: None)
        assert env.core.state().tobytes() == base.state().tobytes()
        assert contact_hash(a) == contact_hash(b)
    assert actor.calls == baseline_actor.calls and len(actor.calls) == 6
    assert (
        obs["A"].shape == (env.actor_dimension,)
        and env.core.model.nu == env.core.model.na == 1400
    )
    assert np.signbit(env.core.commands["A"][0])


@pytest.mark.parametrize("fault", ["missing", "shape", "nan", "bounds"])
def test_bad_pair_is_atomic_before_teacher_or_engine(factory, fault):
    env, actor = factory[0]()
    env.reset(seed=71000)
    before = env.core.state().tobytes()
    actions = zeros()
    if fault == "missing":
        actions.pop("B")
    elif fault == "shape":
        actions["B"] = np.zeros(137)
    elif fault == "nan":
        actions["B"][0] = np.nan
    else:
        actions["B"][0] = 1.1
    with pytest.raises(ValueError):
        env.begin_control(actions)
    assert (
        not actor.calls
        and not env.core.invalid
        and env.core.state().tobytes() == before
        and env.decoder.pending is None
    )


def test_causal_own_opponent_velocity_history_and_atomic_pair(factory):
    env, actor = factory[0]()
    obs, _ = env.reset(seed=71000)
    initial = obs["A"].copy()
    actions = zeros()
    actions["A"][:] = 0.25
    env.step(actions)
    assert len(actor.calls) == 2 and set(env.core.commands) == set("AB")
    assert np.max(abs(env.core.commands["A"] - env.core.commands["B"])) > 0
    assert (
        len(env.histories["A"]._samples) == 2 and max(env.histories["B"]._samples) == 20
    )
    assert not np.array_equal(initial, env.observations()["A"])
    obs2 = env.observations()
    obs2["A"][:] = 99
    assert not np.any(env.observations()["A"] == 99)
    env.reset(seed=71000)
    assert len(env.histories["A"]._samples) == len(env.histories["B"]._samples) == 1


def test_actual_native_midhold_replay_no_new_teacher_decision(factory):
    env, actor = factory[0]()
    env.reset(seed=71000)
    for _ in range(10):
        env.step(zeros())
    env.begin_control(zeros())
    for _ in range(3):
        env.advance_substep()
    snap = env.snapshot()
    assert snap["core"]["tick"] == 203 and snap["core"]["remaining"] == 17
    count = len(actor.calls)
    expected = []
    for _ in range(17):
        expected.append((env.advance_substep(), env.core.state().tobytes()))
    resumed, other_actor = factory[0]()
    resumed.reset(seed=71000)
    resumed.restore(snap)
    assert not other_actor.calls
    for contacts, state in expected:
        actual = resumed.advance_substep()
        assert resumed.core.state().tobytes() == state and contact_hash(
            actual
        ) == contact_hash(contacts)
    assert len(actor.calls) == count and not other_actor.calls
    assert resumed.observations()["A"].tobytes() == env.observations()["A"].tobytes()
    assert resumed.referee.snapshot() == env.referee.snapshot()


@pytest.mark.parametrize(
    "fault", ["cache", "history", "command", "clock", "failure", "bundle"]
)
def test_resigned_bad_snapshot_rejected_before_engine_commit(factory, fault):
    env, _ = factory[0]()
    env.reset(seed=71000)
    env.begin_control(zeros())
    env.advance_substep()
    s = env.snapshot()
    if fault == "cache":
        s["cached"]["A"][0] += 1
    elif fault == "history":
        s["history"]["B"]["samples"][0]["sampled_tick"] = 20
    elif fault == "command":
        s["decoder_record"]["actions"]["A"][0] = 0.5
    elif fault == "clock":
        s["decoder_record"]["sampled_tick"] = 20
    elif fault == "failure":
        s["failures"]["A"] = {"reason": "invented", "tick": 1}
    else:
        s["contract_hash"] = "x"
    s.pop("sha256")
    s["sha256"] = digest(s)
    before = env.core.state().tobytes()
    with pytest.raises(ValueError):
        env.restore(s)
    assert env.core.state().tobytes() == before


def test_opposed_native_yaw_constructor_is_explicit_and_unassisted(factory):
    env, actor = factory[0](placement="opposed", horizon=40)
    env.reset(seed=71000)
    env.step(zeros())
    assert len(actor.calls) == 2 and env.core.contract["yaw_rad"] == {
        "A": 0.0,
        "B": float(np.pi),
    }
    assert not np.any(env.core.data.xfrc_applied) and not np.any(
        env.core.data.qfrc_applied
    )
    assert env.info()["sporting_qualified"] is False
    assert env.info()["physics_total"] == 20


def test_opponent_limb_velocity_enters_actor_features_without_private_activation(
    factory,
):
    import mujoco

    from joxing.native_pair_motor_rl import RING
    from joxing.observations import FEATURE_SPEC, PublicState, observe

    env, _ = factory[0]()
    env.reset(seed=71000)
    env._scratch(full=True)
    own = env.bodies["A"].measure(env.scratch)
    other = env.bodies["B"].measure(env.scratch)
    public = PublicState(env.core.game_id, 0, 0.24, (0, 0), RING)
    original = observe(own, other, public, np.zeros(700, np.float32)).values
    # Explicit scratch measurement perturbation; evaluated live state stays exact.
    before = env.core.state().tobytes()
    env.scratch.qvel[env.bodies["B"].qvel_ids[6:]] += 0.2
    mujoco.mj_forward(env.core.model, env.scratch)
    altered = observe(
        env.bodies["A"].measure(env.scratch),
        env.bodies["B"].measure(env.scratch),
        public,
        np.zeros(700, np.float32),
    ).values
    row = next(
        x for x in FEATURE_SPEC if x["name"] == "opponent_body_linear_velocity_ego"
    )
    assert not np.array_equal(
        original[row["start"] : row["stop"]], altered[row["start"] : row["stop"]]
    )
    ownrow = next(
        x for x in FEATURE_SPEC if x["name"] == "own_intrinsic_joint_velocity"
    )
    assert np.array_equal(
        original[ownrow["start"] : ownrow["stop"]],
        altered[ownrow["start"] : ownrow["stop"]],
    )
    env.scratch.act[env.bodies["B"].activation_ids] = 0.9
    hidden = observe(
        env.bodies["A"].measure(env.scratch),
        env.bodies["B"].measure(env.scratch),
        public,
        np.zeros(700, np.float32),
    ).values
    assert (
        altered.tobytes() == hidden.tobytes() and before == env.core.state().tobytes()
    )


def test_actual_native_fixture_ppo_rollout_audited_update_and_tape(factory, tmp_path):
    import torch

    from joxing.native_pair_motor_train import AuditedPPO, NativeTransitions, actor_hash
    from joxing.native_pair_motor_vec import NativePairMotorVecEnv

    world, _ = factory[0](horizon=40)
    vec = NativePairMotorVecEnv(world, seed=71000)
    model = AuditedPPO(
        "MlpPolicy",
        vec,
        n_steps=8,
        batch_size=16,
        n_epochs=1,
        seed=1,
        device="cpu",
        policy_kwargs={"net_arch": [16, 16], "log_std_init": -3.0},
        verbose=0,
    )
    model.native_update_rows = []
    model.native_output = tmp_path
    before = actor_hash(model.policy)
    cb = NativeTransitions(vec, tmp_path, lambda n: None)
    model.learn(total_timesteps=16, callback=cb)
    assert (
        len(cb.rows) == 8 and world.physics_total == 160 and model.num_timesteps == 16
    )
    assert len(model.native_update_rows) == 1 and actor_hash(model.policy) != before
    assert cb.save_arrays()["arrays"]["commands"]["shape"] == [8, 2, 700]
    saved = torch.load(
        tmp_path / "update-000-continuation.pt", map_location="cpu", weights_only=False
    )
    assert saved["world"]["schema"] == "joxing-native-paired-motor-rl-v1" and saved[
        "last_obs"
    ].shape == (2, 7901)
