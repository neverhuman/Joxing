import copy
import gc
import hashlib

import mujoco
import numpy as np
import pytest
import test_native_pair_motor_rl as native_fixtures
from test_native_pair_motor_rl import (
    FixtureActor,
    FixtureNormalizer,
    contact_hash,
    zeros,
)

from joxing.native_pair_motor_rl import PairMotorConfig
from joxing.native_reset_placement import (
    MODES,
    SEEDS,
    NativeResetPlacementSpec,
    PlacedNativePairMotorEnv,
    build_reset_scene,
    initial_hashes,
    reset_data,
    reset_geometry,
)
from joxing.provenance import digest
from joxing.shared_motor_inputs import FrozenPairTeacher
from joxing.shared_motor_task import NativeReferenceDriver


@pytest.fixture(scope="module")
def native_factory():
    yield from native_fixtures.factory.__wrapped__()


def numeric_model_receipt(model):
    # Full-byte identities, including signed zero. No eight-model cache or
    # exported NumPy view may retain its original native allocation.
    arrays = {}
    for name in dir(model):
        a = getattr(model, name)
        if isinstance(a, np.ndarray) and name != "body_pos":
            arrays[name] = (
                a.dtype.str,
                a.shape,
                hashlib.sha256(a.tobytes()).hexdigest(),
            )
    solver = {}
    for name in dir(model.opt):
        a = getattr(model.opt, name)
        if isinstance(a, np.ndarray):
            solver[name] = hashlib.sha256(a.tobytes()).hexdigest()
        elif isinstance(a, (int, float)):
            solver[name] = a
    return {
        "arrays": arrays,
        "solver": solver,
        "dimensions": (model.nq, model.nv, model.nu, model.na),
        "body_pos": model.body_pos.copy(),
        "root_ids": [model.body(p + "_pelvis").id for p in "AB"],
    }


@pytest.fixture(scope="module")
def placed_factory(native_factory):
    donor, groups = native_factory[1:3]
    # Keep every case, but only a pinned common replay scene and one rolling
    # scene. The target's8GiB limit caught the eager eight-model fixture.
    cache = {}
    receipts = {}
    initial = {}
    for seed in SEEDS:
        initial[seed] = {p: NativeReferenceDriver(donor, seed).initial for p in "AB"}

    def scene_for(mode, seed):
        key = (mode, seed)
        if key not in cache:
            for previous in list(cache):
                if previous != (MODES[0], SEEDS[0]):
                    del cache[previous]
            gc.collect()
            cache[key] = build_reset_scene(
                donor.model,
                initial[seed],
                spec=NativeResetPlacementSpec(mode),
                seed=seed,
            )
            receipts[key] = numeric_model_receipt(cache[key].model)
        assert len(cache) <= 2
        return cache[key]

    def make(mode=MODES[0], seed=71000, horizon=300):
        actor = FixtureActor()

        def drivers(requested):
            if requested != seed:
                raise ValueError("wrong fixture seed")
            return {p: NativeReferenceDriver(donor, seed) for p in "AB"}

        env = PlacedNativePairMotorEnv(
            scene_for(mode, seed),
            drivers,
            FrozenPairTeacher(actor, FixtureNormalizer()),
            groups,
            {"explicit-native-engine-synthetic-actor": "0" * 64},
            config=PairMotorConfig(placement=mode, horizon_ticks=horizon),
        )
        return env, actor

    return make, receipts, initial


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_all_phases_actual_positions_clearance_and_native_bits(
    placed_factory, mode, seed
):
    env, actor = placed_factory[0](mode, seed)
    env.reset(seed=seed)
    scene = env.scene
    initial = placed_factory[2][seed]
    record = scene.report["reset_placement"]
    geometry = record["initial_geometry"]
    assert env.physics_total == 0 and not actor.calls
    assert geometry["clearance_qualified"] and not geometry["penetrating_pair_count"]
    assert geometry["minimum_signed_clearance_lower_bound_m"] >= 0.02
    assert geometry["all_cross_player_pairs_queried"] == np.prod(
        list(geometry["physical_surface_counts"].values())
    )
    for p in "AB":
        actual = env.scratch.xpos[scene.model.body(p + "_pelvis").id]
        assert (
            np.max(abs(actual[:2] - NativeResetPlacementSpec(mode).targets()[p]))
            <= 1e-9
        )
        mapping = env.core.maps[p]
        for field, ids in (
            ("qpos", mapping.qpos_ids),
            ("qvel", mapping.qvel_ids),
            ("activation", mapping.activation_ids),
        ):
            live = (
                env.core.data.act
                if field == "activation"
                else getattr(env.core.data, field)
            )
            assert live[ids].tobytes() == initial[p][field].tobytes()
        assert np.any(env.core.data.qvel[mapping.qvel_ids])
    assert initial_hashes(initial) == record["native_reset_sha256"]


def test_all_other_compiled_numeric_arrays_and_solver_remain_bit_exact(
    placed_factory, native_factory
):
    original = numeric_model_receipt(native_factory[3]["opposed"].model)
    receipts = placed_factory[1]
    # Also works in isolation or a reordered source shard. The normal fullsuite
    # already has these receipts from its eight geometry cases, so no rebuild.
    for mode in MODES:
        for seed in SEEDS:
            if (mode, seed) not in receipts:
                env, _ = placed_factory[0](mode, seed)
                env.close()
    assert set(receipts) == {(mode, seed) for mode in MODES for seed in SEEDS}
    for receipt in receipts.values():
        assert receipt["dimensions"] == (170, 170, 1400, 1400)
        assert receipt["arrays"] == original["arrays"]
        assert receipt["solver"] == original["solver"]
        changed = np.any(original["body_pos"] != receipt["body_pos"], axis=1)
        assert set(np.flatnonzero(changed)) == set(receipt["root_ids"])
        assert (
            original["body_pos"][:, 2].tobytes() == receipt["body_pos"][:, 2].tobytes()
        )


def test_distance_oracle_positive_clipped_and_negative_control():
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="A_pelvis"><geom name="A_ball" type="sphere" size=".1"/></body><body name="B_pelvis"><geom name="B_ball" type="sphere" size=".1" pos=".5 0 0"/></body></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert reset_geometry(m, d)["minimum_signed_clearance_lower_bound_m"] == 0.05
    m.geom_pos[1, 0] = 0.21  # Scratch fixture, never an evaluated native body.
    mujoco.mj_forward(m, d)
    assert reset_geometry(m, d)[
        "minimum_signed_clearance_lower_bound_m"
    ] == pytest.approx(0.01)
    m.geom_pos[1, 0] = 0.15
    mujoco.mj_forward(m, d)
    bad = reset_geometry(m, d)
    assert bad["minimum_signed_clearance_lower_bound_m"] == pytest.approx(-0.05)
    assert bad["penetrating_pair_count"] == 1 and not bad["clearance_qualified"]
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    with pytest.raises(ValueError):
        reset_geometry(m, d)


def test_native_mesh_inclusive_distance_finds_original_overlap(native_factory):
    donor = native_factory[1]
    initial = {p: NativeReferenceDriver(donor, 71048).initial for p in "AB"}
    model = native_factory[3]["opposed"].model
    scratch, _ = reset_data(model, donor.model, initial)
    bad = reset_geometry(model, scratch)
    assert bad["minimum_signed_clearance_lower_bound_m"] < -0.10
    assert not bad["clearance_qualified"] and bad["penetrating_pair_count"] > 0


@pytest.mark.parametrize("bad", ["unknown", None, 1])
def test_unknown_modes_rejected(bad):
    with pytest.raises(ValueError):
        NativeResetPlacementSpec(bad)


@pytest.mark.parametrize("fault", ["owner", "shape", "nan", "dtype", "seed"])
def test_malformed_initial_state_rejected_before_model(
    placed_factory, native_factory, fault
):
    initial = copy.deepcopy(placed_factory[2][71000])
    seed = 71000
    if fault == "owner":
        initial.pop("B")
    elif fault == "shape":
        initial["B"]["qvel"] = np.zeros(84)
    elif fault == "nan":
        initial["A"]["qpos"][0] = np.nan
    elif fault == "dtype":
        initial["A"]["qvel"] = initial["A"]["qvel"].astype(np.float32)
    else:
        seed = True
    with pytest.raises(ValueError):
        build_reset_scene(
            native_factory[1].model,
            initial,
            spec=NativeResetPlacementSpec(MODES[0]),
            seed=seed,
        )


@pytest.mark.parametrize("mode", MODES)
def test_corrected_actual_midhold_and_next_decision_replay(placed_factory, mode):
    env, actor = placed_factory[0](mode)
    env.reset(seed=71000)
    for _ in range(10):
        env.step(zeros())
    env.begin_control(zeros())
    for _ in range(3):
        env.advance_substep()
    saved = env.snapshot()
    calls = len(actor.calls)
    expected = [
        (contact_hash(env.advance_substep()), env.core.state().tobytes())
        for _ in range(17)
    ]
    assert len(actor.calls) == calls
    env.step(zeros())
    state = env.core.state().tobytes()
    other, new_actor = placed_factory[0](mode)
    other.reset(seed=71000)
    other.restore(saved)
    for contacts, engine in expected:
        assert contact_hash(other.advance_substep()) == contacts
        assert other.core.state().tobytes() == engine
    assert not new_actor.calls
    other.step(zeros())
    assert len(new_actor.calls) == 2 and other.core.state().tobytes() == state
    assert env.physics_total + other.physics_total == 277
    assert other.referee.snapshot() == env.referee.snapshot()
    assert np.signbit(env.core.commands["A"][0])


@pytest.mark.parametrize(
    "fault", ["model", "spec", "report", "foreign_snapshot", "wrong_seed"]
)
def test_fixed_placement_drift_foreign_seed_and_restore_fail_atomically(
    placed_factory, fault
):
    env, actor = placed_factory[0]()
    env.reset(seed=71000)
    snapshot = env.snapshot()
    before = env.core.state().tobytes()
    root = env.scene.model.body("A_pelvis").id
    original_pos = env.scene.model.body_pos[root].copy()
    original_spec = env.scene.spec.frame("A_spawn").pos.copy()
    original_report = copy.deepcopy(env.scene.report["reset_placement"])
    try:
        if fault == "model":
            env.scene.model.body_pos[root, 0] += 0.001
        elif fault == "spec":
            env.scene.spec.frame("A_spawn").pos[0] += 0.001
        elif fault == "report":
            env.scene.report["reset_placement"]["seed"] += 1
        elif fault == "foreign_snapshot":
            snapshot["contract_hash"] = "0" * 64
            snapshot.pop("sha256")
            snapshot["sha256"] = digest(snapshot)
        with pytest.raises(ValueError):
            if fault == "wrong_seed":
                env.reset(seed=71016)
            else:
                env.restore(snapshot)
        assert (
            before == env.core.state().tobytes()
            and not actor.calls
            and env.physics_total == 0
        )
    finally:
        env.scene.model.body_pos[root] = original_pos
        env.scene.spec.frame("A_spawn").pos[:] = original_spec
        env.scene.report["reset_placement"] = original_report


def test_corrected_source_trace_replays_exactly(placed_factory, tmp_path):
    from joxing.native_pair_motor_evidence import EvidenceBudget
    from joxing.native_stance_probe import audit_stance, evaluate_stance

    env, _ = placed_factory[0](horizon=60)
    b = EvidenceBudget(tmp_path, raw_limit=8 * 1024**2, text_limit=1024**2)
    result = evaluate_stance(
        env, seed=71000, output=tmp_path / "trial", budget=b, guard=lambda n: None
    )
    replay = audit_stance(
        env, folder=tmp_path / "trial", expected=result, guard=lambda n: None
    )
    assert replay["new_replay_physics"] == 60 and env.physics_total == 120
    assert result["metrics"]["first_interactor_contact"] is None
