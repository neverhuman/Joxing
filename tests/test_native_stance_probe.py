import copy

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

from joxing.native_pair_motor_rl import NativePairMotorEnv, PairMotorConfig
from joxing.native_stance_reference import PROFILE, WalkToHoldReferenceDriver
from joxing.provenance import digest
from joxing.scene import build_scene
from joxing.shared_motor_inputs import FrozenPairTeacher
from joxing.shared_motor_task import NativeReferenceDriver

LANES = "opposed-separated-lanes-v1"


@pytest.fixture(scope="module")
def native_factory():
    yield from native_fixtures.factory.__wrapped__()


@pytest.fixture(scope="module")
def stance_factory(native_factory):
    donor, groups, scenes = native_factory[1:4]
    variants = {"opposed": scenes["opposed"], LANES: build_scene(spawn_variant=LANES)}

    def make(placement=LANES, profile=PROFILE):
        actor = FixtureActor()
        driver = (
            WalkToHoldReferenceDriver if profile == PROFILE else NativeReferenceDriver
        )
        env = NativePairMotorEnv(
            variants[placement],
            lambda seed: {p: driver(donor, seed) for p in "AB"},
            FrozenPairTeacher(actor, FixtureNormalizer()),
            groups,
            {"explicit-native-engine-synthetic-actor": "0" * 64},
            config=PairMotorConfig(placement=placement, horizon_ticks=300),
            reference_profile=profile,
        )
        return env, actor

    return make, variants


def test_lanes_preserve_all_native_physics_except_fixed_spawn(stance_factory):
    baseline, lanes = [stance_factory[1][p].model for p in ("opposed", LANES)]
    assert (lanes.nq, lanes.nv, lanes.nu, lanes.na) == (170, 170, 1400, 1400)
    # Compile-then-translate preserves even derived rest-length/inertia caches.
    # There is no arithmetic tolerance, including signed zero, in this gate.
    for name in dir(baseline):
        a, b = getattr(baseline, name), getattr(lanes, name)
        if not isinstance(a, np.ndarray) or name == "body_pos":
            continue
        assert a.tobytes() == b.tobytes(), name
    delta = lanes.body_pos - baseline.body_pos
    expected = np.zeros_like(delta)
    for p, y in (("A", -1.25), ("B", 1.25)):
        expected[lanes.body(p + "_pelvis").id, 1] = y
    assert delta.tobytes() == expected.tobytes()
    assert baseline.opt.timestep == lanes.opt.timestep == 0.001
    assert lanes.opt.integrator == baseline.opt.integrator
    assert lanes.opt.gravity.tobytes() == baseline.opt.gravity.tobytes()


def test_lanes_rigid_translation_preserves_native_state_and_command_mapping(
    stance_factory,
):
    base, _ = stance_factory[0]("opposed", None)
    lanes, _ = stance_factory[0](LANES, None)
    base.reset(seed=71000)
    lanes.reset(seed=71000)
    assert base.core.state().tobytes() == lanes.core.state().tobytes()
    for p in "AB":
        for name in (
            "qpos_ids",
            "qvel_ids",
            "actuator_ids",
            "activation_ids",
            "native_to_world",
        ):
            assert (
                getattr(base.core.maps[p], name).tobytes()
                == getattr(lanes.core.maps[p], name).tobytes()
            )
        # World translation cancellation can differ below f32 rounding at zero.
        assert (
            np.max(abs(base.core.inputs[p].values - lanes.core.inputs[p].values)) < 1e-6
        )
    base.begin_control(zeros())
    lanes.begin_control(zeros())
    for p in "AB":
        assert base.core.commands[p].tobytes() == lanes.core.commands[p].tobytes()
        assert np.signbit(lanes.core.commands[p][0])
    assert "reference_profile" not in base.contract
    assert "reference_profile" not in base.core.drivers["A"].snapshot()


@pytest.mark.parametrize(
    "placement,profile", [(LANES, None), (LANES, PROFILE), ("opposed", PROFILE)]
)
def test_new_variants_actual_203_plus_held17_and_next_decision(
    stance_factory, placement, profile
):
    env, actor = stance_factory[0](placement, profile)
    env.reset(seed=71000)
    for _ in range(10):
        env.step(zeros())
    env.begin_control(zeros())
    for _ in range(3):
        env.advance_substep()
    snapshot = env.snapshot()
    assert snapshot["core"]["tick"] == 203 and snapshot["core"]["remaining"] == 17
    calls = len(actor.calls)
    expected = [
        (contact_hash(env.advance_substep()), env.core.state().tobytes())
        for _ in range(17)
    ]
    assert len(actor.calls) == calls
    env.step(zeros())
    final_state, final_obs = env.core.state().tobytes(), env.observations()
    resumed, other = stance_factory[0](placement, profile)
    resumed.reset(seed=71000)
    resumed.restore(snapshot)
    assert not other.calls
    for contacts, state in expected:
        assert contact_hash(resumed.advance_substep()) == contacts
        assert resumed.core.state().tobytes() == state
    assert not other.calls
    resumed.step(zeros())
    assert len(other.calls) == 2 and resumed.core.state().tobytes() == final_state
    assert all(
        final_obs[p].tobytes() == resumed.observations()[p].tobytes() for p in "AB"
    )
    assert resumed.referee.snapshot() == env.referee.snapshot()
    assert not np.any(env.core.data.xfrc_applied) and not np.any(
        env.core.data.qfrc_applied
    )
    assert env.physics_total + resumed.physics_total == 277


@pytest.mark.parametrize("fault", ["mode", "T", "axis", "phase", "sha"])
def test_resigned_reference_corruption_rejected_atomically(stance_factory, fault):
    env, actor = stance_factory[0]()
    env.reset(seed=71000)
    s = copy.deepcopy(env.snapshot())
    ref = s["core"]["references"]["A"]
    if fault == "phase":
        ref["sample_time"] = 0.02
    elif fault == "sha":
        ref["reference_profile_sha256"] = "0" * 64
    else:
        field = {"mode": "profile", "T": "stop_time_s", "axis": "native_axes_sha256"}[
            fault
        ]
        ref["reference_profile"][field] = 0.5 if fault == "T" else "corrupt"
        ref["reference_profile_sha256"] = digest(ref["reference_profile"])
    s["core"].pop("sha256")
    s["core"]["sha256"] = digest(s["core"])
    s.pop("sha256")
    s["sha256"] = digest(s)
    before = env.core.state().tobytes()
    with pytest.raises(ValueError):
        env.restore(s)
    assert before == env.core.state().tobytes() and not actor.calls


def test_unknown_variants_and_mismatched_factory_rejected(stance_factory):
    with pytest.raises(ValueError):
        build_scene(spawn_variant="unnamed-lanes")
    with pytest.raises(ValueError):
        PairMotorConfig(placement="unnamed-lanes")
    env, _ = stance_factory[0](profile=None)
    env.reference_profile = PROFILE
    with pytest.raises(ValueError):
        env.reset(seed=71000)
    assert env.physics_total == 0


def test_evaluated_reference_sampling_never_writes_actual_engine(stance_factory):
    env, _ = stance_factory[0]()
    env.reset(seed=71000)
    state = env.core.state().tobytes()
    for driver in env.core.drivers.values():
        driver.sample(400, 0.4)
        driver.sample(3000, 3.0)
    assert state == env.core.state().tobytes()
    assert env.physics_total == 0
    assert not np.any(env.core.model.body_gravcomp)
    assert env.core.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


@pytest.mark.parametrize(
    "tick,failed,drift,speed,good",
    [
        (3000, False, 0.30, 0.15, True),
        (3000, False, 0.3000001, 0.15, False),
        (3000, False, 0.30, 0.1500001, False),
        (2980, False, 0.0, None, False),
        (3000, True, 0.0, 0.0, False),
    ],
)
def test_stance_requires_whole_horizon_maximum_excursion_and_last_full_second(
    tick, failed, drift, speed, good
):
    from joxing.native_stance_probe import stance_gate

    assert (
        stance_gate(
            final_tick=tick,
            failed=failed,
            maximum_drift_m=drift,
            last_second_mean_speed_mps=speed,
        )
        is good
    )


def test_stance_denominator_requires_all_four_seeds_and_both_corners():
    from joxing.native_stance_probe import SEEDS, condition_conclusion

    rows = [
        {
            "seed": s,
            "metrics": {
                "actors": {
                    p: {"stance_candidate": i < (3 if p == "A" else 2)} for p in "AB"
                }
            },
        }
        for i, s in enumerate(SEEDS)
    ]
    assert condition_conclusion(rows)["qualifying_starts"] == {"A": 3, "B": 2}
    assert not condition_conclusion(rows)["stance_candidate_only"]
    with pytest.raises(ValueError):
        condition_conclusion(rows[:3])
    with pytest.raises(ValueError):
        condition_conclusion([rows[0]] * 4)


def test_stance_fixture_full_trace_exact_replay_and_corruption(
    stance_factory, tmp_path
):
    import json

    from joxing.native_pair_motor_evidence import EvidenceBudget
    from joxing.native_stance_probe import METRIC_COLUMNS, audit_stance, evaluate_stance
    from joxing.native_tape_storage import load

    env, _ = stance_factory[0]()
    budget = EvidenceBudget(tmp_path, raw_limit=16 * 1024**2, text_limit=2 * 1024**2)
    result = evaluate_stance(
        env, seed=71000, output=tmp_path / "trial", budget=budget, guard=lambda n: None
    )
    trace = load(tmp_path / "trial/stance-trace.npz", result["trace_tape"])
    assert trace["metrics_1ms"].shape == (result["final_tick"], 2, len(METRIC_COLUMNS))
    assert result["optimizer_updates"] == 0 and not result["sporting_qualified"]
    assert result["metrics"]["first_interactor_contact"] is None
    audit = audit_stance(
        env, folder=tmp_path / "trial", expected=result, guard=lambda n: None
    )
    assert audit["new_replay_physics"] == result["final_tick"]
    assert env.physics_total == 2 * result["final_tick"]
    # Both complete source passes count; this is synthetic teacher evidence only.
    (tmp_path / "fixture-physics.json").write_text(
        json.dumps(
            {
                "physics": env.physics_total,
                "scope": "native engine/synthetic teacher source fixture",
            }
        )
        + "\n"
    )
    bad = copy.deepcopy(result)
    bad["metrics"]["actors"]["A"]["maximum_root_xy_excursion_m"] += 1
    before = env.physics_total
    with pytest.raises(ValueError):
        audit_stance(env, folder=tmp_path / "trial", expected=bad, guard=lambda n: None)
    assert env.physics_total == before
