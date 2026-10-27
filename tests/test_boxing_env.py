import copy

import numpy as np
import pytest

from joxing.env import BoxingEnv, excitation
from joxing.provenance import digest
from joxing.scene import build_scene


@pytest.fixture(scope="module")
def scene():
    return build_scene()


def commands(a=0.2, b=-0.2):
    return {"A": np.full(700, a, np.float32), "B": np.full(700, b, np.float32)}


def normalized(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def test_commands_atomic_and_same_tick_opponent_observation_is_causal(scene):
    e = BoxingEnv(scene, horizon_ticks=20)
    e.reset(seed=90101)
    before = e.snapshot()
    with pytest.raises(ValueError):
        e.begin_control(commands(b=float("nan")))
    assert e.snapshot() == before
    captured = []

    def b(o, h, rng):
        captured.append((o.values.copy(), h.values.copy(), float(rng.random())))
        return np.full(700, -0.2, np.float32)

    for value in [0.8, -0.8]:
        world = BoxingEnv(scene, horizon_ticks=20)
        world.reset(seed=90101)
        result = world.act(
            {
                "A": lambda o, h, rng, selected=value: np.full(
                    700, selected, np.float32
                ),
                "B": b,
            }
        )
        assert all(result["terminated"].values())
        assert result["bootstrap_mask"] == {"A": 0.0, "B": 0.0}
        assert not result["info"]["competitive_ready"]
    np.testing.assert_array_equal(captured[0][0], captured[1][0])
    np.testing.assert_array_equal(captured[0][1], captured[1][1])
    assert captured[0][2] == captured[1][2]


def test_delay_and_mid_control_shared_engine_contact_history_replay_exact(scene):
    e = BoxingEnv(scene, horizon_ticks=40)
    e.reset(seed=90102)
    e.begin_control(commands(), delays={"A": 2, "B": 0})
    for _ in range(2):
        e.advance_substep()
        np.testing.assert_array_equal(
            e.data.ctrl[e.bodies["A"].actuator_ids],
            excitation(np.zeros(700, np.float32)),
        )
    e.advance_substep()
    s = e.snapshot()
    np.testing.assert_array_equal(e.commands["A"], commands()["A"])
    with pytest.raises(RuntimeError):
        e.begin_control(commands())
    e.resume_control()
    expected = e.snapshot()
    tail = normalized(e.evidence[3:])
    original_counter = e.physics_total
    e.restore(s)
    assert e.physics_total == original_counter  # Audit counter never rewinds.
    e.resume_control()
    assert e.snapshot() == expected
    assert digest(normalized(e.evidence)) == digest(tail)
    assert e.physics_total == original_counter + 17
    boundary = e.collector_cut()
    assert not e.done
    e.step(commands(-0.1, 0.3))
    final = e.snapshot()
    e.restore(boundary)
    e.step(commands(-0.1, 0.3))
    assert e.snapshot() == final


def test_replay_rejects_semantic_corruption_before_live_mutation(scene):
    e = BoxingEnv(scene, horizon_ticks=40)
    e.reset(seed=90103)
    e.begin_control(commands())
    e.advance_substep()
    good = e.snapshot()
    for kind in ["control", "observation", "clock", "contract", "history"]:
        bad = copy.deepcopy(good)
        if kind == "control":
            bad["application"]["commands"]["A"][0] = 0.9
        if kind == "observation":
            bad["application"]["cached_observations"]["A"][0] += 1
        if kind == "clock":
            bad["application"]["tick"] += 1
        if kind == "contract":
            bad["contract_hash"] = "0" * 64
        if kind == "history":
            bad["application"]["histories"]["A"]["samples"][-1]["values"][0] += 1
        bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
        with pytest.raises(ValueError):
            e.restore(bad)
        assert e.snapshot() == good


def test_bell_has_no_extra_physics_and_resource_cut_bootstraps(scene):
    e = BoxingEnv(scene, horizon_ticks=3)
    e.reset(seed=90104)
    result = e.step(commands())
    assert e.tick == e.physics_total == 3
    assert e.referee.ended and all(result["terminated"].values())
    with pytest.raises(RuntimeError):
        e.step(commands())
    saved = e.snapshot()
    e.restore(saved)
    assert e.snapshot() == saved
    e = BoxingEnv(scene, horizon_ticks=40)
    e.reset(seed=90104)
    e.begin_control(commands())
    e.advance_substep()
    cut = e.collector_cut()
    assert not e.done
    result = e.truncate("unit resource cap")
    assert all(result["truncated"].values()) and not any(result["terminated"].values())
    assert result["bootstrap_mask"] == {"A": 1.0, "B": 1.0} and result[
        "trace_mask"
    ] == {"A": 0.0, "B": 0.0}
    e.restore(cut)
    assert not e.done and e.remaining_control == 19


def test_numerical_fault_quarantines_whole_world_and_preserves_attempt(
    scene, monkeypatch
):
    e = BoxingEnv(scene, horizon_ticks=20)
    e.reset(seed=90104)
    e.begin_control(commands())

    def fault(model, data):
        data.qvel[0] = np.nan
        raise FloatingPointError("injected unit fixture")

    monkeypatch.setattr("joxing.env.step_contacts", fault)
    with pytest.raises(FloatingPointError):
        e.advance_substep()
    assert e.physics_total == 1 and np.isnan(e.fault_state).any() and e.invalid_reason
    with pytest.raises(RuntimeError):
        e.transition()
    with pytest.raises(ValueError):
        e.snapshot()


def test_native_opponent_glove_force_reaches_referee_and_mid_contact_replays(scene):
    e = BoxingEnv(scene, horizon_ticks=20, scenario="close-overlap-collision-probe-v2")
    e.reset(seed=90104)
    e.begin_control(commands())
    e.advance_substep()
    row = e.evidence[-1]
    positive = [
        t
        for interval in row["gloves"].values()
        for t in interval["touches"]
        if t["normal_force_N"] > 0
    ]
    assert positive, (
        "Named native collision stress reset must actually transmit glove force"
    )
    assert e.referee.next_tick == 1
    saved = e.snapshot()
    e.resume_control()
    final = e.snapshot()
    tail = normalized(e.evidence[1:])
    e.restore(saved)
    e.resume_control()
    assert e.snapshot() == final and normalized(e.evidence) == tail
    assert not e.competitive_ready


def test_reset_clears_old_game_packets_and_live_contract_edits_reject(scene):
    e = BoxingEnv(scene, horizon_ticks=20)
    e.reset(seed=90101)
    e.begin_control(commands(), delays={"A": 2, "B": 0})
    old = e.queues["A"].pending[0]
    e.reset(seed=90101)
    with pytest.raises(ValueError, match="Foreign"):
        e.queues["A"].enqueue(old, now_tick=0)
    assert not e.queues["A"].pending and e.physics_total == 0
    e.contract = dict(e.contract, control_ticks=10)
    with pytest.raises(ValueError, match="contract drift"):
        e.begin_control(commands())
