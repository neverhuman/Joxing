import copy
import json

import numpy as np
import pytest

from joxing.motor_storage import write_json
from joxing.scene import build_scene
from joxing.sensed_env import SensedBoxingEnv
from joxing.sensor_audit import (
    audit_sensor_transport,
    audit_transport,
    compare_sensor_replay,
    compare_zero_delay_oracle,
    read_arrays,
)
from joxing.sensor_evidence import SensorTape

DELAYS = {"A": 1, "B": 2}
SCHEDULE = {"A": [40, 0], "B": [None, 20, 0]}


def collect(env, tape, checkpoint=None):
    saved = None
    while not env.done:
        if not env.remaining_control:
            env.begin_control(tape.choose(env), delays=DELAYS)
        env.advance_substep()
        tape.capture(env)
        if env.tick == checkpoint:
            saved = env.snapshot()
    return saved


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sensor-tape")
    scene = build_scene()
    env = SensedBoxingEnv(scene, horizon_ticks=200, sensor_schedule=SCHEDULE)
    env.reset(seed=90202)
    tape = SensorTape(env, maximum_ticks=200)
    checkpoint = collect(env, tape, checkpoint=83)
    source = folder / "source"
    tape.save(env, source)
    write_json(source / "checkpoint-83.json", checkpoint)
    write_json(source / "contract.json", env.contract)
    final = env.snapshot()
    env.restore(json.loads(json.dumps(checkpoint)))
    replay = SensorTape(env, maximum_ticks=117)
    collect(env, replay)
    assert env.snapshot() == final
    mid = folder / "mid"
    replay.save(env, mid)
    write_json(mid / "contract.json", env.contract)
    return scene, source, mid


def load(folder):
    def j(name):
        return json.loads((folder / name).read_text())

    tape = read_arrays(folder / "tape.npz")
    s = read_arrays(folder / "sensor-tape.npz")
    for name in ["history_sample_ages", "history_receipt_ages"]:
        tape[name] = s[name]
    return [
        j("sensor-initial.json"),
        j("sensor-final.json"),
        j("trace.json"),
        j("sensor-trace.json"),
        tape,
        dict(zip(s["sample_ticks"].tolist(), s["sample_values"], strict=True)),
        SCHEDULE,
        DELAYS,
    ]


def test_saved_native_inputs_audit_without_new_physics(evidence):
    scene, source, mid = evidence
    result = audit_sensor_transport(source, scene, expected_physics=200, delays=DELAYS)
    assert result["status"] == "transport_passed"
    assert result["audit_new_physics"] == 0
    assert result["sampled_engines_checked"] == 11
    assert result["actor_calls"] == {"A": 9, "B": 8}
    assert not result["physical_referee_audited"] and not result["formal_sensor_proof"]
    assert (
        audit_sensor_transport(mid, scene, expected_physics=117, delays=DELAYS)[
            "status"
        ]
        == "transport_passed"
    )


def test_actual_pending_sensor_mid_hold_tape_matches_source_tail(evidence):
    _, source, mid = evidence
    a, b = load(source), load(mid)
    assert b[0]["base"]["application"]["tick"] == 83
    assert b[0]["base"]["application"]["remaining_control"] == 17
    assert b[0]["sensors"]["A"]["pending"]
    for k in ["applied", "excitation", "packets"]:
        np.testing.assert_array_equal(a[4][k][83:], b[4][k])
    assert a[2]["state_sha256"][83:] == b[2]["state_sha256"]
    for k in [
        "observations",
        "histories",
        "history_ticks",
        "history_received",
        "history_valid",
        "chosen",
        "decision_ticks",
    ]:
        np.testing.assert_array_equal(a[4][k][5:], b[4][k])
    assert a[1] == b[1]
    assert compare_sensor_replay(
        source, mid, source_checkpoint=source / "checkpoint-83.json"
    )["sensor_exact"]


@pytest.mark.parametrize(
    "kind",
    [
        "arrival_tick",
        "missing_delivery",
        "missing_drop",
        "fake_available",
        "old_velocity",
        "future_tick",
        "private_seed",
        "extra_rng_before",
        "extra_rng_after",
        "missing_decision",
        "held_command",
        "final_pending",
        "history_order",
        "extra_frame_field",
        "sample_age",
        "receipt_age",
        "availability_type",
    ],
)
def test_transport_semantic_corruptions_reject_without_checksum_gate(evidence, kind):
    _, source, _ = evidence
    args = copy.deepcopy(load(source))
    initial, final, trace, sensed, tape, _, _, _ = args
    if kind == "arrival_tick":
        next(r for r in sensed["events"] if r["event"] == "received")[
            "received_tick"
        ] += 1
    elif kind == "missing_delivery":
        sensed["events"].remove(
            next(r for r in sensed["events"] if r["event"] == "received")
        )
    elif kind == "missing_drop":
        sensed["events"].remove(
            next(r for r in sensed["events"] if r["disposition"] == "dropped")
        )
    elif kind == "fake_available":
        sensed["frames"][0]["A"]["available"] = True
    elif kind == "old_velocity":
        tape["histories"][5, 0, -2, 79] += 0.25
    elif kind == "future_tick":
        tape["history_ticks"][5, 0, -1] = 120
    elif kind == "private_seed":
        sensed["frames"][1]["A"]["observation_owner"]["game_id"] = initial["base"][
            "application"
        ]["game_id"]
    elif kind == "extra_rng_before":
        trace["rngs_before"][1]["A"]["state"]["state"] += 1
    elif kind == "extra_rng_after":
        sensed["rngs_after"][0]["B"]["state"]["state"] += 1
    elif kind == "missing_decision":
        tape["decision_ticks"] = tape["decision_ticks"][1:]
    elif kind == "held_command":
        tape["applied"][4, 1, 0] += 0.2
        from joxing.env import excitation

        tape["excitation"][4, 1] = excitation(tape["applied"][4, 1])
    elif kind == "final_pending":
        final["sensors"]["A"]["pending"].append(
            copy.deepcopy(initial["sensors"]["A"]["pending"][0])
        )
    elif kind == "history_order":
        tape["histories"][5, 0, [-1, -2]] = tape["histories"][5, 0, [-2, -1]]
    elif kind == "extra_frame_field":
        sensed["frames"][1]["A"]["opponent_rng"] = {"seed": 90202}
    elif kind == "sample_age":
        tape["history_sample_ages"][5, 0, -2] += 1
    elif kind == "receipt_age":
        tape["history_receipt_ages"][5, 0, -2] += 1
    elif kind == "availability_type":
        sensed["frames"][1]["A"]["available"] = 1
    with pytest.raises(ValueError):
        audit_transport(*args)


def test_both_dropped_actual_sample_is_retained_even_without_live_proofs(
    tmp_path, evidence
):
    scene, _, _ = evidence
    env = SensedBoxingEnv(
        scene, horizon_ticks=20, sensor_schedule={"A": [None], "B": [None]}
    )
    env.reset(seed=90203)
    assert env.sample_proofs == {}
    tape = SensorTape(env, maximum_ticks=20)
    collect(env, tape)
    tape.save(env, tmp_path)
    write_json(tmp_path / "contract.json", env.contract)
    result = audit_sensor_transport(tmp_path, scene, expected_physics=20, delays=DELAYS)
    assert result["actor_calls"] == {"A": 0, "B": 0}
    assert result["sampled_engines_checked"] == 2


def test_sensor_allocation_bound_rejects_before_collection(evidence):
    scene, _, _ = evidence
    env = SensedBoxingEnv(scene)
    env.reset(seed=90203)
    for maximum in [0, 1001, True]:
        with pytest.raises(ValueError):
            SensorTape(env, maximum_ticks=maximum)


def test_zero_delay_comparator_matches_new_development_oracle_and_rejects_drift(
    tmp_path, evidence
):
    from joxing.boxing_evidence import BoxingTape
    from joxing.env import BoxingEnv
    from joxing.motor_storage import save_npz

    scene, _, _ = evidence
    plain = BoxingEnv(scene, horizon_ticks=40)
    sensed = SensedBoxingEnv(scene, horizon_ticks=40)
    plain.reset(seed=90101)
    sensed.reset(seed=90101)
    old, new = BoxingTape(plain, maximum_ticks=40), SensorTape(sensed, maximum_ticks=40)
    collect(plain, old)
    collect(sensed, new)
    a, b = tmp_path / "plain", tmp_path / "sensed"
    old.save(plain, a)
    new.save(sensed, b)
    write_json(a / "contract.json", plain.contract)
    write_json(b / "contract.json", sensed.contract)
    assert compare_zero_delay_oracle(a, b)["exact_saved_oracle"]
    arrays = read_arrays(b / "tape.npz")
    arrays["histories"][1, 0, -2, 79] += 0.1
    save_npz(b / "tape.npz", **arrays)
    with pytest.raises(ValueError, match="Oracle arrays"):
        compare_zero_delay_oracle(a, b)


@pytest.mark.parametrize(
    "kind", ["receipt", "frame", "sample", "physical_contact", "checkpoint"]
)
def test_replay_comparator_rejects_tail_and_checkpoint_corruption(
    tmp_path, evidence, kind
):
    import shutil

    from joxing.motor_storage import save_npz

    _, source, mid = evidence
    replay = tmp_path / "replay"
    shutil.copytree(mid, replay)
    checkpoint = source / "checkpoint-83.json"
    if kind in {"receipt", "frame"}:
        name = replay / "sensor-trace.json"
        v = json.loads(name.read_text())
        if kind == "receipt":
            v["events"][0]["sampled_tick"] += 1
        else:
            v["frames"][0]["A"]["sample_age_ticks"] += 1
        write_json(name, v)
    elif kind == "sample":
        arrays = read_arrays(replay / "sensor-tape.npz")
        arrays["sample_values"][0, 0, 79] += 0.1
        save_npz(replay / "sensor-tape.npz", **arrays)
    elif kind == "physical_contact":
        name = replay / "trace.json"
        v = json.loads(name.read_text())
        v["intervals"][0]["gloves"].pop(next(iter(v["intervals"][0]["gloves"])))
        write_json(name, v)
    else:
        checkpoint = tmp_path / "wrong-checkpoint.json"
        v = json.loads((source / "checkpoint-83.json").read_text())
        v["last_sensor_sample"] += 20
        write_json(checkpoint, v)
    with pytest.raises(ValueError):
        compare_sensor_replay(source, replay, source_checkpoint=checkpoint)


@pytest.mark.parametrize("kind", ["engine", "commands", "features"])
def test_actual_sample_corruption_rejects_after_array_and_state_hash_updates(
    tmp_path, evidence, kind
):
    import shutil

    from joxing.boxing_evidence import array_sha
    from joxing.motor_storage import save_npz

    scene, source, _ = evidence
    folder = tmp_path / "altered"
    shutil.copytree(source, folder)
    arrays = read_arrays(folder / "sensor-tape.npz")
    if kind == "engine":
        arrays["sample_engines"][1, 1] += 0.01
        trace = json.loads((folder / "trace.json").read_text())
        trace["state_sha256"][19] = array_sha(arrays["sample_engines"][1])
        write_json(folder / "trace.json", trace)
    elif kind == "commands":
        arrays["sample_commands"][1, 0, 0] += 0.1
    else:
        arrays["sample_values"][1, 0, 79] += 0.1
    save_npz(folder / "sensor-tape.npz", **arrays)
    with pytest.raises(ValueError):
        audit_sensor_transport(folder, scene, expected_physics=200, delays=DELAYS)
