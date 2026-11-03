import copy

import numpy as np
import pytest
import test_native_pair_motor_rl as native_fixtures

from joxing.native_arm_guard import (
    GuardCounter,
    GuardMotorWorld,
    guard_good,
    guard_measure,
    guard_reward,
)
from joxing.native_pair_motor_rl import PairMotorConfig, yaw_frame
from joxing.native_reset_placement import (
    MODES,
    NativeResetPlacementSpec,
    PlacedNativePairMotorEnv,
    build_reset_scene,
)
from joxing.provenance import ROOT, digest
from joxing.shared_motor_inputs import FrozenPairTeacher
from joxing.shared_motor_task import NativeReferenceDriver
from joxing.upper_residual import UpperResidualSpec


@pytest.fixture(scope="module")
def native_factory():
    yield from native_fixtures.factory.__wrapped__()


@pytest.fixture(scope="module")
def guard_factory(native_factory):
    donor, groups = native_factory[1:3]
    initial = {p: NativeReferenceDriver(donor, 71000).initial for p in "AB"}
    scene = build_reset_scene(
        donor.model, initial, spec=NativeResetPlacementSpec(MODES[1]), seed=71000
    )
    spec = UpperResidualSpec.from_native_model(
        donor.model, groups, ROOT / "third_party/ms-human-700"
    )
    bundle = {"explicit-native-engine-synthetic-actor": "0" * 64}

    def make(*, horizon=300, baseline=False, run_id="guard-source-fixture"):
        actor = native_fixtures.FixtureActor()
        teacher = FrozenPairTeacher(actor, native_fixtures.FixtureNormalizer())
        if baseline:
            world = PlacedNativePairMotorEnv(
                scene,
                lambda seed: {p: NativeReferenceDriver(donor, seed) for p in "AB"},
                teacher,
                groups,
                bundle,
                config=PairMotorConfig(placement=MODES[1], horizon_ticks=horizon),
            )
        else:
            world = GuardMotorWorld(
                scene,
                donor,
                teacher,
                groups,
                bundle,
                spec,
                run_id=run_id,
                horizon_ticks=horizon,
            )
        return world, actor

    return make


def zeros():
    return {p: np.zeros(46, np.float32) for p in "AB"}


def nonzero():
    return {
        "A": np.linspace(-0.75, 0.75, 46, dtype=np.float32),
        "B": np.linspace(0.4, -0.4, 46, dtype=np.float32),
    }


def test_zero_arm_action_exact_native_walking_and_signed_zero(guard_factory):
    world, actor = guard_factory(horizon=60)
    base, baseline_actor = guard_factory(horizon=60, baseline=True)
    world.reset(seed=761)
    base.reset(seed=71000)
    assert world.core.state().tobytes() == base.core.state().tobytes()
    assert world.core.game_id != base.core.game_id
    for tick in range(60):
        if tick % 20 == 0:
            world.begin_control(zeros())
            base.begin_control(native_fixtures.zeros())
        a, b = world.advance_substep(), base.advance_substep()
        assert world.core.state().tobytes() == base.core.state().tobytes()
        assert native_fixtures.contact_hash(a) == native_fixtures.contact_hash(b)
    assert actor.calls == baseline_actor.calls and len(actor.calls) == 6
    assert np.signbit(world.core.commands["A"][0])
    assert world.core.model.nu == world.core.model.na == 1400
    assert world.physics_total == 60
    world.close()
    base.close()


@pytest.mark.parametrize("fault", ["missing", "shape", "nan", "bounds"])
def test_guard_bad_pair_atomic_before_inference(guard_factory, fault):
    world, actor = guard_factory()
    world.reset(seed=761)
    before = world.snapshot()
    a = zeros()
    if fault == "missing":
        a.pop("B")
    elif fault == "shape":
        a["B"] = np.zeros(138)
    elif fault == "nan":
        a["B"][0] = np.nan
    else:
        a["B"][0] = 1.1
    with pytest.raises(ValueError):
        world.begin_control(a)
    assert world.snapshot() == before and not actor.calls and world.physics_total == 0
    world.close()


def test_upper_commands_and_next_decision_exact_after_native_midhold(guard_factory):
    world, actor = guard_factory()
    world.reset(seed=761)
    for _ in range(10):
        world.step(nonzero())
    world.begin_control(nonzero())
    for _ in range(3):
        world.advance_substep()
    snap = world.snapshot()
    assert world.core.tick == 203 and world.core.remaining == 17
    for p in "AB":
        h = world.held[p]
        protected = list(world.spec.protected_ids)
        assert np.frombuffer(h.teacher_bytes, np.float32)[protected].tobytes() == (
            world.core.commands[p][protected].tobytes()
        )
        assert h.teacher_bytes != h.command_bytes
    count = len(actor.calls)
    expected = []
    for _ in range(17):
        contacts = world.advance_substep()
        expected.append(
            (world.core.state().tobytes(), native_fixtures.contact_hash(contacts))
        )
    assert len(actor.calls) == count
    expected_next = world.step(nonzero())
    next_state, next_snapshot = world.core.state().tobytes(), world.snapshot()
    resumed, resumed_actor = guard_factory()
    resumed.reset(seed=761)
    resumed.restore(snap)
    assert not resumed_actor.calls
    for state, contacts in expected:
        actual = resumed.advance_substep()
        assert resumed.core.state().tobytes() == state
        assert native_fixtures.contact_hash(actual) == contacts
    actual_next = resumed.step(nonzero())
    assert resumed.core.state().tobytes() == next_state
    assert resumed.snapshot() == next_snapshot
    for p in "AB":
        assert actual_next[0][p].tobytes() == expected_next[0][p].tobytes()
        assert actual_next[1][p] == expected_next[1][p]
    assert len(resumed_actor.calls) == 2
    assert resumed.physics_total == 37  # restore never rewinds physical charges.
    world.close()
    resumed.close()


def test_fixed_native_phase_has_unique_reset_ownership(guard_factory):
    world, _ = guard_factory()
    world.reset(seed=761)
    initial = world.core.state().tobytes()
    world.step(zeros())
    old = world.snapshot()
    old_id = world.core.game_id
    world.reset(seed=761)
    assert world.core.state().tobytes() == initial
    assert world.core.game_id != old_id
    assert world.native_seed == 71000 and world.episode_seed == 761
    before = world.snapshot()
    with pytest.raises(ValueError, match="epoch"):
        world.restore(old)
    assert world.snapshot() == before
    world.close()


@pytest.mark.parametrize(
    "fault", ["goal", "counter", "metric", "hold", "nonce", "anchor"]
)
def test_resigned_bad_guard_snapshot_atomic(guard_factory, fault):
    world, actor = guard_factory()
    world.reset(seed=761)
    world.begin_control(nonzero())
    world.advance_substep()
    before = world.snapshot()
    s = copy.deepcopy(before)
    if fault == "goal":
        s["guard_goals"]["B"][0] += 0.125
    elif fault == "counter":
        s["counters"]["A"]["current_hold"] = 2
    elif fault == "metric":
        s["last_metrics"]["A"]["face_cos"] += 0.1
    elif fault == "hold":
        s["held"] = None
    elif fault == "nonce":
        s["nonce"] = "0" * 64
    else:
        s["initial_head_z"]["B"] += 0.1
    s.pop("sha256")
    s["sha256"] = digest(s)
    count = len(actor.calls)
    with pytest.raises(ValueError):
        world.restore(s)
    assert world.snapshot() == before and len(actor.calls) == count
    world.close()


def test_guard_goal_matches_world_fk_and_causal_cache(guard_factory):
    world, _ = guard_factory(horizon=340)
    world.reset(seed=761)
    # Reach phase starts at300ms in the legacy task; this new task stays guard.
    for _ in range(16):
        world.step(zeros())
    obs = world.observations()
    measure = guard_measure(world.base, world.initial_head_z)
    for p in "AB":
        frame = yaw_frame(
            world.base.scratch.xmat[world.scene.model.body(p + "_pelvis").id].reshape(
                3, 3
            )
        )
        errors = obs[p][-9:-3].reshape(2, 3)
        assert np.allclose(
            np.linalg.norm(errors, axis=1), measure[p]["distances_m"], atol=2e-6
        )
        assert np.allclose(frame @ frame.T, np.eye(3))
        assert np.array_equal(obs[p][-3:], [1, 0, 0])
    world.begin_control(nonzero())
    for _ in range(3):
        world.advance_substep()
    assert world.core.tick == 323
    assert world.observations()["A"].tobytes() == obs["A"].tobytes()
    assert max(world.histories["A"]._samples) == 320
    assert not np.any(world.core.data.xfrc_applied)
    assert not np.any(world.core.data.qfrc_applied)
    world.close()


def metric():
    return {
        "distances_m": [0.1, 0.1],
        "face_cos": 0.75,
        "up_world_z": 1.0,
        "tilt_rad": 0.0,
        "head_drop_m": 0.0,
        "head_z_m": 1.7,
    }


def test_reward_linear_signal_units_effort_and_no_early_exit_incentive():
    m, a = metric(), np.zeros(46, np.float32)
    expected = 0.001 * (0.2 + 0.5 + 0.3 * (1 - 0.1 / 1.5) + 0.1 * 0.75)
    assert guard_reward(m, a, False) == pytest.approx(expected)
    far = m | {"distances_m": [0.8, 0.8]}
    farther = m | {"distances_m": [0.9, 0.9]}
    assert guard_reward(far, a, False) > guard_reward(farther, a, False) > 0
    assert guard_reward(m, np.ones(46, np.float32), False) == pytest.approx(
        expected - 0.00001
    )
    assert guard_reward(m, a, True) == pytest.approx(expected - 1)
    bowed = m | {"head_drop_m": 0.11}
    assert guard_reward(bowed, a, False) < guard_reward(m, a, False)
    assert not guard_good(bowed)


@pytest.mark.parametrize(
    "key,value",
    [
        ("distances_m", [0.16, 0.1]),
        ("face_cos", 0.49),
        ("tilt_rad", np.deg2rad(20.1)),
        ("head_drop_m", 0.1001),
    ],
)
def test_continuous_hold_resets_on_each_gate(key, value):
    counter = GuardCounter()
    for _ in range(39):
        counter.observe(metric())
    assert not counter.summary()["qualified_hold"]
    counter.observe(metric() | {key: value})
    assert counter.current_hold == 0
    for _ in range(40):
        counter.observe(metric())
    assert counter.summary()["qualified_hold"]
    assert counter.maximum_hold == 40 and counter.ticks == 80


def test_vec46_joint_terminal_capture_and_autoreset(guard_factory):
    from joxing.native_arm_guard_vec import GuardPairVec

    world, actor = guard_factory(horizon=20)
    vec = GuardPairVec(world, seed=761)
    initial = vec.reset()
    game = world.core.game_id
    before = world.core.state().tobytes()
    with pytest.raises(ValueError):
        vec.step_async(np.zeros((2, 138), np.float32))
    assert world.core.state().tobytes() == before and not actor.calls
    obs, rewards, done, infos = vec.step(np.zeros((2, 46), np.float32))
    transition = vec.last_transition
    assert done.tolist() == [True, True]
    assert world.core.tick == 0 and world.core.game_id != game
    assert transition["info"]["episode_id"] == game
    assert transition["episode_serial"] == 1 and world.episode_serial == 2
    assert transition["metrics_1ms"].shape == (20, 2, 7)
    assert transition["contact_sha256_1ms"].shape == (20, 32)
    assert transition["truncated"] and not transition["terminated"]
    assert np.array_equal(obs, initial)
    for i, p in enumerate("AB"):
        assert infos[i]["TimeLimit.truncated"]
        assert (
            infos[i]["terminal_observation"].tobytes()
            == transition["next_observations"][p].tobytes()
        )
        assert infos[i]["episode"]["r"] == rewards[i]
    assert len(actor.calls) == 2
    assert world.physics_total == 20
    vec.close()


def test_guard_trace_path_budget_and_typed46_actions(tmp_path):
    from joxing.native_arm_guard_evidence import (
        FIELDS,
        GuardTraceWriter,
        checked_arrays,
    )
    from joxing.native_pair_motor_evidence import EvidenceBudget

    budget = EvidenceBudget(
        tmp_path / "owned", raw_limit=2 * 1024**2, text_limit=1024**2
    )
    with pytest.raises(ValueError, match="outside"):
        GuardTraceWriter(tmp_path / "escaped", contract_hash="0" * 64, budget=budget)
    assert not (tmp_path / "escaped").exists()
    arrays = {
        k: np.zeros((1, *shape), dtype=dtype) for k, (dtype, shape) in FIELDS.items()
    }
    checked_arrays(arrays, 1)
    arrays["raw_policy_actions"][0, 1, 0] = 2.0
    with pytest.raises(ValueError, match="clipped46"):
        checked_arrays(arrays, 1)
    arrays["applied_actions"][0, 1, 0] = 1.0
    checked_arrays(arrays, 1)
    arrays["applied_actions"] = np.zeros((1, 2, 138), np.float32)
    with pytest.raises(ValueError, match="shape"):
        checked_arrays(arrays, 1)


@pytest.mark.parametrize("fault", ["negative", "nonce", "clock", "behavior"])
def test_guard_trace_metadata_rejects_resigned_semantic_fault(fault):
    from joxing.native_arm_guard_evidence import checked_metadata

    meta = {
        "episode_id": "motor-" + "1" * 32,
        "nonce": "2" * 64,
        "episode_serial": 1,
        "logical_seed": 761,
        "native_seed": 71000,
        "tick": 20,
        "behavior": {"boundary": 0, "policy_sha256": "3" * 64, "actor_hash": "4" * 64},
    }
    checked_metadata(meta)
    if fault == "negative":
        meta["episode_serial"] = 0
    elif fault == "nonce":
        meta["nonce"] = "motor-" + "2" * 32
    elif fault == "clock":
        meta["tick"] = 19
    else:
        meta["behavior"]["boundary"] = -1
    with pytest.raises(ValueError):
        checked_metadata(meta)


def test_guard_real_ppo_full_trace_replay_and_checkpoint_rng(guard_factory, tmp_path):
    import random

    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from joxing.native_arm_guard_evidence import GuardTraceWriter, read_guard_chunk
    from joxing.native_arm_guard_pilot import GuardTransitions, restore_guard_checkpoint
    from joxing.native_arm_guard_vec import GuardPairVec
    from joxing.native_pair_motor_evidence import EvidenceBudget
    from joxing.native_pair_motor_pilot import (
        CheckpointBook,
        PilotAuditedPPO,
        optimizer_steps,
    )
    from joxing.native_pair_motor_train import actor_hash

    world, actor = guard_factory(horizon=60)
    vec = GuardPairVec(world, seed=761)
    budget = EvidenceBudget(tmp_path, raw_limit=32 * 1024**2, text_limit=2 * 1024**2)
    writer = GuardTraceWriter(
        tmp_path / "chunks",
        contract_hash=world.contract_hash,
        budget=budget,
        chunk_rows=5,
    )
    model = PilotAuditedPPO(
        "MlpPolicy",
        vec,
        n_steps=8,
        batch_size=8,
        n_epochs=2,
        seed=761,
        device="cpu",
        policy_kwargs={"net_arch": [16, 16], "log_std_init": -2},
        verbose=0,
    )
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    before_hash = actor_hash(model.policy)
    model.pilot = CheckpointBook(tmp_path / "checkpoints", budget)

    class Golden(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []

        def _on_step(self):
            self.rows.append(copy.deepcopy(vec.last_transition))
            return True

    golden = Golden()
    model.learn(
        total_timesteps=32,
        callback=[GuardTransitions(vec, writer, lambda n: None), golden],
    )
    result = writer.close()
    book = model.pilot
    assert len(book.checkpoints) == 3 and len(book.updates) == 2
    assert optimizer_steps(model) == 8 and model.num_timesteps == 32
    assert actor_hash(model.policy) != before_hash
    assert world.physics_total == 320 and world.completed_episodes == 5
    assert model.action_space.shape == (46,)
    all_arrays, all_meta, chain, count = [], [], "0" * 64, 0
    for r in result["chunks"]:
        arrays, meta, chain = read_guard_chunk(
            writer.output / r["path"],
            r,
            contract_hash=world.contract_hash,
            first_row=count,
            previous_chain=chain,
        )
        all_arrays.append(arrays)
        all_meta.extend(meta)
        count += len(meta)
    assert count == 16 and chain == result["chain"]
    arrays = {k: np.concatenate([a[k] for a in all_arrays]) for k in all_arrays[0]}
    assert arrays["raw_policy_actions"].shape == (16, 2, 46)
    assert arrays["source_obs"].shape == (16, 2, 7901)
    assert arrays["metrics_1ms"].shape == (16, 20, 2, 7)
    error = []
    behavior = {}
    for i, meta in enumerate(all_meta):
        b = meta["behavior"]["boundary"]
        if b not in behavior:
            behavior[b] = PilotAuditedPPO.load(
                book.output / f"boundary-{b:03d}.zip", device="cpu"
            )
        with torch.no_grad():
            v, log, _ = behavior[b].policy.evaluate_actions(
                torch.as_tensor(arrays["source_obs"][i]),
                torch.as_tensor(arrays["raw_policy_actions"][i]),
            )
        error.append(
            max(
                float(np.max(abs(v.numpy() - arrays["values"][i]))),
                float(np.max(abs(log.numpy() - arrays["log_probs"][i]))),
            )
        )
    assert max(error) == 0.0
    calls = len(actor.calls)
    restore_guard_checkpoint(book.output, book.checkpoints[0], vec)
    assert len(actor.calls) == calls
    for i, original in enumerate(golden.rows):
        assert (
            vec.world.observations()["A"].tobytes()
            == arrays["source_obs"][i, 0].tobytes()
        )
        vec.step(arrays["applied_actions"][i])
        t = vec.last_transition
        for key in (
            "engine_before",
            "engine_after",
            "metrics_1ms",
            "contact_sha256_1ms",
        ):
            assert t[key].tobytes() == arrays[key][i].tobytes()
        for key in ("teacher_inputs", "teacher_commands", "commands"):
            assert (
                np.stack([t[key][p] for p in "AB"]).tobytes()
                == arrays[key][i].tobytes()
            )
        assert {k: v for k, v in t["info"].items() if k != "physics_total"} == {
            k: v
            for k, v in original["info"].items()
            if k not in ("physics_total", "behavior")
        }
        assert t["info"]["guard_metrics"] == original["info"]["guard_metrics"]
        assert t["rewards"] == original["rewards"]
        assert t["nonce"] == original["nonce"]
    for receipt in (book.checkpoints[0], book.checkpoints[-1]):
        first = restore_guard_checkpoint(book.output, receipt, vec)
        assert optimizer_steps(first) == (0 if receipt["number"] == 0 else 8)
        with torch.no_grad():
            action, value, logprob = first.policy(torch.as_tensor(first._last_obs))
        randoms = (
            torch.rand(3).numpy().tobytes(),
            np.random.rand(3).tobytes(),
            random.random(),
        )
        expected = vec.step(np.clip(action.numpy(), -1, 1))
        state = world.core.state().tobytes()
        second = restore_guard_checkpoint(book.output, receipt, vec)
        with torch.no_grad():
            action2, value2, logprob2 = second.policy(torch.as_tensor(second._last_obs))
        assert (
            action.numpy().tobytes(),
            value.numpy().tobytes(),
            logprob.numpy().tobytes(),
        ) == (
            action2.numpy().tobytes(),
            value2.numpy().tobytes(),
            logprob2.numpy().tobytes(),
        )
        assert randoms == (
            torch.rand(3).numpy().tobytes(),
            np.random.rand(3).tobytes(),
            random.random(),
        )
        actual = vec.step(np.clip(action2.numpy(), -1, 1))
        assert all(actual[i].tobytes() == expected[i].tobytes() for i in range(3))
        assert world.core.state().tobytes() == state
    assert world.physics_total == 720
    # Authentication rejects damaged checkpoints before touching the world.
    path = book.output / "boundary-000.pt"
    path.write_bytes(path.read_bytes() + b"corruption")
    snapshot, calls = world.snapshot(), len(actor.calls)
    with pytest.raises(ValueError, match="external identity"):
        restore_guard_checkpoint(book.output, book.checkpoints[0], vec)
    assert world.snapshot() == snapshot and len(actor.calls) == calls
    # All evidence survives; external receipts detect corrupt chunks, no retry.
    receipt = result["chunks"][0]
    path = writer.output / receipt["path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="external identity"):
        read_guard_chunk(
            path,
            receipt,
            contract_hash=world.contract_hash,
            first_row=0,
            previous_chain="0" * 64,
        )
    (tmp_path / "guard-source-receipt.json").write_text(
        __import__("json").dumps(
            {
                "scope": "Native engine/synthetic teacher source fixture; not released-controller skill",
                "joint_transitions": 16,
                "actor_samples": 32,
                "rollout_updates": 2,
                "optimizer_minibatches": 8,
                "episodes": 5,
                "physics": world.physics_total,
                "behavior_max_error": max(error),
                "all_native_replay_bytes_exact": True,
                "chunks": result,
                "checkpoints": book.checkpoints,
            },
            indent=2,
        )
        + "\n"
    )
    vec.close()
