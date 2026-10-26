import json

import mujoco
import numpy as np
import pytest

from joxing.balance import BalanceConfig, BalanceModel
from joxing.physics import Contact
from joxing.stance_env import StanceEnv
from joxing.support import SupportMonitor, floor_loads


@pytest.fixture(scope="module")
def env():
    return StanceEnv()


def test_native_gravity_model_preserves_anatomy_and_equipment(env):
    audit = env.rig.audit()
    assert (audit["nq"], audit["nv"], audit["nu"], audit["na"], audit["neq"]) == (
        85,
        85,
        700,
        700,
        42,
    )
    assert audit["preserved_muscle_parameters"]
    assert audit["preserved_segment_mass_inertia"]
    assert audit["glove_mass_increment_kg"] == pytest.approx(0.566, abs=1e-12)
    assert audit["gravity"] == [0, 0, -9.81]
    assert all("weld" not in x for x in audit["root_joint_names"])


def test_reset_projection_solves_actual_native_keyframe_error(env):
    env.reset(options={"scenario_id": 30001})
    audit = env.reset_audit
    assert audit["source_equality_error"] > 0.01
    assert audit["projected_equality_error"] < 1e-8
    assert audit["bilateral_100ms"]
    assert audit["max_penetration_m"] < 0.002
    assert env.data.time == 0
    assert audit["assistance_during_evaluation"] == []


def test_group_partition_covers_every_muscle_and_bounds_controls(env):
    ids = np.concatenate(env.groups.ids)
    assert sorted(ids.tolist()) == list(range(700))
    assert len(env.groups.names) == 138
    ctrl = env.groups.expand(np.linspace(-1, 1, 138))
    assert ctrl.min() == pytest.approx(1 / (1 + np.exp(7.5)))
    assert ctrl.max() == pytest.approx(1 / (1 + np.exp(-2.5)))
    for bad in (np.zeros(700), np.full(138, np.nan), np.full(138, 1.1)):
        with pytest.raises(ValueError):
            env.groups.expand(bad)


def test_observations_are_real_and_control_cadence_has_no_pose_overwrite(env):
    obs, _ = env.reset(options={"scenario_id": 30002})
    assert obs.shape == (1023,)
    a, b = env.obs_slices["activation"]
    assert obs[a:b] == pytest.approx(env.data.act)
    initial = env.data.qpos.copy()
    obs, reward, term, trunc, _ = env.step(np.zeros(138))
    assert env.tick == 10 and env.data.time == pytest.approx(0.01)
    assert not np.array_equal(initial, env.data.qpos)
    assert env.data.ctrl == pytest.approx(env.groups.expand(np.zeros(138)))
    assert np.isfinite(reward) and not term and not trunc


def test_snapshot_replays_at_control_boundary_and_rng(env):
    env.reset(seed=81, options={"scenario_id": 30000})
    for _ in range(8):
        env.step(np.zeros(138))
    snapshot = json.loads(json.dumps(env.snapshot(), sort_keys=True))
    reference = env.step(np.full(138, -0.3))
    reference_state = env.snapshot()
    env.restore(snapshot)
    restored = env.step(np.full(138, -0.3))
    assert restored[0] == pytest.approx(reference[0], abs=1e-9)
    assert restored[1:] == reference[1:]
    assert env.snapshot()["state"] == pytest.approx(reference_state["state"], abs=1e-9)
    snapshot["action_hash"] = "invalid"
    before = env.data.qpos.copy()
    with pytest.raises(ValueError):
        env.restore(snapshot)
    assert np.array_equal(before, env.data.qpos)


def test_external_assistance_and_nonfinite_observation_fail(env):
    env.reset(options={"scenario_id": 30000})
    env.data.xfrc_applied[env.rig.pelvis, 2] = 1
    with pytest.raises(RuntimeError, match="assistance"):
        env.step(np.zeros(138))
    env.reset(options={"scenario_id": 30000})
    env.data.act[0] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        env.observe()


@pytest.mark.parametrize("dt", [0.002, 0.001, 0.0005])
def test_support_requires_time_persistence_not_one_brush(dt):
    monitor = SupportMonitor()
    assert (
        monitor.update({"right": 300, "left": 300, "nonfoot": 100}, 600, 0, 0, dt)
        is None
    )
    monitor.update({"right": 300, "left": 300, "nonfoot": 0}, 600, 0, 0, dt)
    for _ in range(round(0.05 / dt) - 1):
        assert (
            monitor.update({"right": 0, "left": 0, "nonfoot": 100}, 600, 0, 0, dt)
            is None
        )
    assert (
        monitor.update({"right": 0, "left": 0, "nonfoot": 100}, 600, 0, 0, dt)
        == "nonfoot_support"
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_support_force_sign_with_reversed_geom_order(reverse):
    z = np.zeros(3)
    c = Contact(
        2 if reverse else 0,
        0 if reverse else 2,
        z,
        z,
        np.array([0, 0, -50 if reverse else 50]),
        50,
        0,
        0,
        z,
        z,
    )
    assert floor_loads([c], 0, {2: "right"}) == {"right": 50, "left": 0, "nonfoot": 0}
    with pytest.raises(KeyError):
        floor_loads([c], 0, {})


def test_real_floor_load_balances_momentum_and_weight():
    xml = '<mujoco><option timestep=".001" integrator="implicitfast"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body pos="0 0 .099"><freejoint/><geom name="foot" type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    from joxing.physics import step_contacts

    impulse = 0.0
    for _ in range(200):
        contacts = step_contacts(m, d)
        impulse += (
            floor_loads(contacts, m.geom("floor").id, {m.geom("foot").id: "right"})[
                "right"
            ]
            * 0.001
        )
    assert impulse - 2 * 9.81 * 0.2 == pytest.approx(2 * d.qvel[2], abs=1e-9)


def test_truncation_is_not_physical_failure():
    e = StanceEnv(BalanceConfig(episode_s=0.02))
    e.step(np.zeros(138))
    _, _, terminated, truncated, _ = e.step(np.zeros(138))
    assert not terminated and truncated
    with pytest.raises(RuntimeError):
        e.step(np.zeros(138))


def test_reset_variants_and_invalid_config_fail():
    with pytest.raises(ValueError):
        BalanceConfig(dt_s=0.003)
    with pytest.raises(ValueError):
        BalanceConfig(episode_s=float("nan"))
    rig = BalanceModel()
    a, _ = rig.prepare(30000)
    b, _ = rig.prepare(30001)
    assert not np.array_equal(a.qpos, b.qpos)


def test_push_scheduler_delivers_exact_impulse_and_restores_during_push(env):
    env.reset(
        options={"scenario_id": 30000, "push_Ns": 2.0, "push_direction_rad": np.pi / 2}
    )
    # Scheduler fixture starts an accepted upright state at the scheduled time;
    # this is not evidence that a policy has survived the preceding five seconds.
    env.tick = 5000
    env.data.time = 5.0
    env.step(np.full(138, -0.3))
    snapshot = json.loads(json.dumps(env.snapshot()))
    for _ in range(5):
        env.step(np.full(138, -0.3))
    expected = env.snapshot()
    assert env.push_delivered == pytest.approx([0, 2, 0], abs=1e-12)
    assert not np.any(env.data.xfrc_applied)
    env.restore(snapshot)
    for _ in range(5):
        env.step(np.full(138, -0.3))
    assert env.snapshot()["state"] == pytest.approx(expected["state"], abs=1e-9)
    assert env.push_delivered == pytest.approx([0, 2, 0], abs=1e-12)


def test_fault_recorder_preserves_last_valid_state_and_invalid_attempt(env, tmp_path):
    from joxing.stance_train import FaultRecorder

    env.reset(options={"scenario_id": 30000})
    wrapper = FaultRecorder(env, tmp_path)
    with pytest.raises(ValueError):
        wrapper.step(np.full(138, np.nan))
    row = json.loads((tmp_path / "fault-step-1.json").read_text())
    assert row["invalid"] and row["attempted_transition"] == 1
    assert row["last_valid"]["application"]["tick"] == 0


def test_invalid_and_incomplete_evaluations_never_enter_summary():
    from joxing.stance_train import paired_summary

    with pytest.raises(ValueError):
        paired_summary(
            [{"scenario_id": 1, "invalid": True, "status": "invalid"}],
            [{"scenario_id": 1, "invalid": False, "status": "not_run"}],
        )


def test_json_key_order_cannot_change_observation_semantics(env):
    env.reset(options={"scenario_id": 30000})
    env.step(np.zeros(138))
    before = env.observe().copy()
    snapshot = json.loads(json.dumps(env.snapshot(), sort_keys=True))
    env.restore(snapshot)
    assert env.observe() == pytest.approx(before, abs=1e-9)


@pytest.mark.parametrize("corruption", ["nonfinite", "shape", "unknown_field", "clock"])
def test_malformed_snapshot_is_rejected_before_mutating_environment(env, corruption):
    env.reset(options={"scenario_id": 30000})
    snapshot = json.loads(json.dumps(env.snapshot()))
    before = env.data.qpos.copy()
    if corruption == "nonfinite":
        snapshot["state"][0] = float("nan")
    elif corruption == "shape":
        snapshot["application"]["initial_hands"] = [0]
    elif corruption == "unknown_field":
        snapshot["application"]["rig"] = "bad"
    else:
        snapshot["application"]["tick"] = 12
    with pytest.raises(ValueError):
        env.restore(snapshot)
    assert np.array_equal(before, env.data.qpos)
