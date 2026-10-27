import copy
import json

import numpy as np
import pytest

from joxing.boxing_evidence import diagnostic_command
from joxing.env import BoxingEnv
from joxing.provenance import digest
from joxing.replay import ReplaySnapshot
from joxing.scene import build_scene
from joxing.sensed_env import SensedBoxingEnv


@pytest.fixture(scope="module")
def scene():
    return build_scene()


def controller(frame, rng):
    assert frame.available
    return diagnostic_command(frame.latest_observation, frame.history, rng)


def normalized(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    return value


def test_zero_delay_variant_matches_default_engine_actions_history_and_base_application(
    scene,
):
    base = BoxingEnv(scene, horizon_ticks=120)
    sensed = SensedBoxingEnv(scene, horizon_ticks=120)
    base.reset(seed=90101)
    sensed.reset(seed=90101)
    for _ in range(6):
        plain = base.decision_frame()
        timed = sensed.decision_frame()
        commands = {}
        for a in ("A", "B"):
            np.testing.assert_array_equal(
                plain[a][0].values, timed[a].latest_observation.values
            )
            np.testing.assert_array_equal(plain[a][1].values, timed[a].history.values)
            assert timed[a].sample_age_ticks == timed[a].receipt_age_ticks == 0
            assert timed[a].latest_observation.game_id == "actor-visible"
            assert not hasattr(timed[a], "queues") and not hasattr(timed[a], "game_id")
            commands[a] = diagnostic_command(*plain[a], base.rngs[a])
            np.testing.assert_array_equal(
                commands[a], controller(timed[a], sensed.rngs[a])
            )
        base.step(commands, delays={"A": 1, "B": 2})
        sensed.step(commands, delays={"A": 1, "B": 2})
        np.testing.assert_array_equal(base.state(), sensed.state())
        assert (
            ReplaySnapshot.capture(sensed)["application"]
            == base.snapshot()["application"]
        )
    # More than100ticks exercises numeric->JSON key canonicalization.
    snapshot = json.loads(json.dumps(sensed.snapshot()))
    sensed.restore(snapshot)
    assert sensed.snapshot() == snapshot


def test_real_delayed_state_and_velocity_not_just_changed_time_labels(scene):
    e = SensedBoxingEnv(scene, horizon_ticks=60, sensor_schedule={"A": [0], "B": [20]})
    e.reset(seed=90102)
    truth = {0: {a: o.values.copy() for a, o in e.cached_observations.items()}}
    b_rng = copy.deepcopy(e.rngs["B"].bit_generator.state)
    seen = []

    def b(frame, rng):
        seen.append(frame)
        np.testing.assert_array_equal(
            frame.latest_observation.values, truth[e.tick - 20]["B"]
        )
        assert frame.sample_age_ticks == 20 and frame.receipt_age_ticks == 0
        return controller(frame, rng)

    e.act({"A": controller, "B": b})
    assert not seen and e.rngs["B"].bit_generator.state == b_rng
    assert np.any(e.cached_observations["B"].values[79:158] != 0)
    truth[20] = {a: o.values.copy() for a, o in e.cached_observations.items()}
    e.act({"A": controller, "B": b})
    truth[40] = {a: o.values.copy() for a, o in e.cached_observations.items()}
    e.act({"A": controller, "B": b})
    assert len(seen) == 2 and e.done


def test_actual_one_two_tick_receipts_and_no_mid_control_decision(scene):
    e = SensedBoxingEnv(scene, horizon_ticks=20, sensor_schedule={"A": [1], "B": [2]})
    e.reset(seed=90103)
    assert not any(f.available for f in e.decision_frame().values())
    e.begin_control({a: np.full(700, 0.1, np.float32) for a in ("A", "B")})
    e.advance_substep()
    frames = e.observations()
    assert frames["A"].available and not frames["B"].available
    assert e.sensors["A"].records[0]["received_tick"] == 1
    with pytest.raises(RuntimeError):
        e.decision_frame()
    e.advance_substep()
    assert e.sensors["B"].records[0]["received_tick"] == 2
    assert e.observations()["B"].latest_observation.sampled_tick == 0


def test_mid_control_pending_sensors_and_full_future_replay_exact(scene):
    e = SensedBoxingEnv(
        scene, horizon_ticks=80, sensor_schedule={"A": [40], "B": [20, 0]}
    )
    e.reset(seed=90104)
    e.act({"A": controller, "B": controller})
    frame = e.decision_frame()
    commands = {
        a: controller(frame[a], e.rngs[a])
        if frame[a].available
        else e.commands[a].copy()
        for a in ("A", "B")
    }
    e.begin_control(commands, delays={"A": 1, "B": 2})
    for _ in range(3):
        e.advance_substep()
    saved = json.loads(json.dumps(e.snapshot()))
    assert len(e.sensors["A"].pending) == 2
    physics_offset = len(e.evidence)
    sensor_offset = len(e.sensor_events)
    e.resume_control()
    while not e.done:
        e.act({"A": controller, "B": controller})
    final = e.snapshot()
    physical = normalized(e.evidence[physics_offset:])
    sensor = copy.deepcopy(e.sensor_events[sensor_offset:])
    count = e.physics_total
    e.restore(saved)
    assert e.physics_total == count
    e.resume_control()
    while not e.done:
        e.act({"A": controller, "B": controller})
    assert e.snapshot() == final
    assert normalized(e.evidence) == physical and e.sensor_events == sensor
    assert e.physics_total == count + 57


def test_drop_reorder_and_reset_do_not_expose_old_or_future_match_inputs(scene):
    e = SensedBoxingEnv(
        scene, horizon_ticks=60, sensor_schedule={"A": [40, 0, 0], "B": [None, 0]}
    )
    e.reset(seed=90101)
    old = e.sensors["A"].pending[0]
    e.act({"A": controller, "B": controller})
    assert e.decision_frame()["A"].latest_observation.sampled_tick == 20
    e.act({"A": controller, "B": controller})
    f = e.decision_frame()["A"]
    assert (
        f.latest_observation.sampled_tick == 40
        and e.sensors["A"].records[0]["disposition"] == "late_history"
    )
    assert e.sensors["B"].records[0]["disposition"] == "dropped"
    e.reset(seed=90101)
    with pytest.raises(ValueError, match="Foreign"):
        e.sensors["A"].enqueue(old, now_tick=0)


@pytest.mark.parametrize(
    "kind",
    [
        "proof",
        "commands",
        "pending",
        "future_receipt",
        "schedule",
        "missing_proof",
        "seed_metadata",
    ],
)
def test_sensor_snapshot_corruption_rejects_before_any_live_mutation(scene, kind):
    e = SensedBoxingEnv(scene, horizon_ticks=60, sensor_schedule={"A": [40], "B": [0]})
    e.reset(seed=90101)
    e.act({"A": controller, "B": controller})
    good = e.snapshot()
    bad = copy.deepcopy(good)
    if kind == "proof":
        bad["sample_proofs"]["0"]["engine"][1] += 0.1
    if kind == "commands":
        bad["sample_proofs"]["0"]["commands"]["A"][0] += 0.1
    if kind == "pending":
        bad["sensors"]["A"]["pending"][0]["values"][0] += 0.1
    if kind == "future_receipt":
        bad["sensors"]["B"]["records"][0]["received_tick"] = 21
    if kind == "schedule":
        bad["sensors"]["A"]["records"][0]["delivery_tick"] += 1
    if kind == "missing_proof":
        bad["sample_proofs"].pop("0")
    if kind == "seed_metadata":
        bad["base"]["application"]["game_id"] = "seed-stolen"
        bad["base"]["sha256"] = digest(
            {k: v for k, v in bad["base"].items() if k != "sha256"}
        )
    bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError):
        e.restore(bad)
    assert e.snapshot() == good


def test_external_cut_uses_delivered_final_input_and_preserves_ending_semantics(scene):
    e = SensedBoxingEnv(scene, horizon_ticks=40, sensor_schedule={"A": [1], "B": [20]})
    e.reset(seed=90103)
    e.begin_control({a: np.zeros(700, np.float32) for a in ("A", "B")})
    for _ in range(3):
        e.advance_substep()
    result = e.truncate("test resource cut")
    assert result["bootstrap_mask"] == {"A": 1.0, "B": 1.0} and result[
        "trace_mask"
    ] == {"A": 0.0, "B": 0.0}
    assert result["observations"]["A"].latest_observation.sampled_tick == 0
    assert not result["observations"]["B"].available
    snapshot = json.loads(json.dumps(e.snapshot()))
    e.restore(snapshot)
    assert e.snapshot() == snapshot


def test_same_tick_actions_cannot_change_other_delivered_input_rng_or_command(scene):
    captured = []
    for value in [-0.8, 0.8]:
        e = SensedBoxingEnv(scene, horizon_ticks=20)
        e.reset(seed=90101)

        def b(frame, rng):
            command = controller(frame, rng)
            captured.append(
                (
                    frame.latest_observation.values.copy(),
                    frame.history.values.copy(),
                    command,
                    rng.bit_generator.state,
                )
            )
            return command

        e.act(
            {
                "A": lambda frame, rng, chosen=value: np.full(700, chosen, np.float32),
                "B": b,
            }
        )
    for a, b in zip(captured[0][:3], captured[1][:3], strict=True):
        np.testing.assert_array_equal(a, b)
    assert captured[0][3] == captured[1][3]


def test_pair_validation_atomic_and_schedule_drift_stops_before_physics(scene):
    e = SensedBoxingEnv(scene, horizon_ticks=20)
    e.reset(seed=90101)
    saved = e.snapshot()
    with pytest.raises(ValueError):
        e.step({"A": np.zeros(700, np.float32), "B": np.full(700, np.nan, np.float32)})
    assert e.snapshot() == saved and e.physics_total == 0
    e.sensor_schedule["A"] = (40,)
    with pytest.raises(ValueError, match="drift"):
        e.act({"A": controller, "B": controller})
    assert e.physics_total == 0
    with pytest.raises(ValueError, match="drift"):
        e.snapshot()
