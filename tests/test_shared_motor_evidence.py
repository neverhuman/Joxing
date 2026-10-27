import copy
import json
from pathlib import Path

import numpy as np
import pytest

from joxing.evidence_storage import read_arrays, read_json
from joxing.motor_storage import RawCredit
from joxing.provenance import digest
from joxing.shared_motor_audit import SemanticInputAudit
from joxing.shared_motor_evidence import MotorTape
from joxing.shared_motor_proof import (
    exact_tail,
    planned_rows,
    run_proof,
    validate_packet,
    validate_test_summary,
)
from joxing.shared_motor_saved_audit import audit_measurements, audit_saved_tape
from joxing.shared_motor_task import (
    NativeReferenceDriver,
    SharedMotorTask,
    build_motor_scene,
)
from joxing.upstream_motor import NativeTask


class FixtureNormalizer:
    def __init__(self):
        self.fingerprint = digest(self.export())

    def check(self):
        return None

    def apply(self, values):
        n = self.export()
        return np.clip(
            (values - np.asarray(n["mean"]))
            / np.sqrt(np.asarray(n["var"]) + n["epsilon"]),
            -10,
            10,
        ).astype(np.float32)

    def export(self):
        return {
            "mean": [0.0] * 3601,
            "var": [10000.0] * 3601,
            "epsilon": 1e-8,
            "clip": 10.0,
            "training": False,
            "norm_reward": False,
            "norm_obs": True,
        }


class FixtureTeacher:
    """No trained weights; source-regression physical fixtures only."""

    def __init__(self):
        self.calls = []
        self.normalizer = FixtureNormalizer()

    def infer_pair(self, pair, *, game_id, tick):
        self.calls.append(tick)
        return {
            p: (-0.2 - 0.01 * np.tanh(v.values[85:170])[np.arange(700) % 85]).astype(
                np.float32
            )
            for p, v in pair.items()
        }


@pytest.fixture(scope="module")
def proof(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "experiments/v0.0.6/protocol.json").read_text())[
        "environment"
    ]
    drivers = {
        p: NativeReferenceDriver(NativeTask(config), 71000 + i)
        for i, p in enumerate(("A", "B"))
    }
    task = SharedMotorTask(
        build_motor_scene(),
        drivers,
        FixtureTeacher(),
        horizon_ticks=60,
        bundle_hash="fixture-only",
    )
    folder = tmp_path_factory.mktemp("shared-motor-evidence")
    credit = RawCredit(folder, 8 * 1024**2)
    tape = MotorTape(task, recording=True)
    mid = prefix = None
    while not task.done:
        if not task.remaining:
            tape.collect_decision(task)
        contacts = task.advance_substep(lambda _: None)
        tape.capture_substep(task, contacts)
        if task.tick == 23:
            mid, prefix = task.snapshot(), tape.support_snapshot(task)
            tape.mid = {"task": mid, "support": prefix}
    tape.save(task, folder / "source")
    source_final = task.snapshot()
    calls = len(task.teacher.calls)
    task.restore(mid)
    replay = MotorTape(task, continuation=prefix)
    while not task.done:
        if not task.remaining:
            replay.collect_decision(task)
        replay.capture_substep(task, task.advance_substep(lambda _: None))
        if task.tick == 40:
            assert (
                len(task.teacher.calls) == calls
            )  # 17 held physical steps, no decision
    replay.save(task, folder / "replay")
    assert task.snapshot() == source_final
    yield task, folder, tape, replay, credit
    for driver in drivers.values():
        driver.task.close()


def test_lossless_motor_tape_and_saved_independent_measurements(proof):
    task, folder, tape, replay, credit = proof
    result = audit_saved_tape(task, folder / "source")
    assert result["passed"] and result["measured_intervals"] == 60
    assert result["decisions"] == 3 and result["video_frames"] == 3
    assert result["physics_steps"] == 0
    assert not result["trajectory_continuation_audited"]
    assert not result["policy_inference_audited"]
    assert audit_saved_tape(task, folder / "replay")["passed"]
    source_arrays = read_arrays(folder / "source/tape.npz")
    replay_arrays = read_arrays(folder / "replay/tape.npz")
    for key in ("ticks", "end_qpos", "posture", "loads"):
        assert np.array_equal(source_arrays[key][23:], replay_arrays[key])
    assert tape.hashes[23:] == replay.hashes
    ncontacts = tape.offsets[23]
    for key in ("contacts", "contact_geoms"):
        assert np.array_equal(source_arrays[key][ncontacts:], replay_arrays[key])
    for key in ("raw", "normalized", "commands", "excitation"):
        assert np.array_equal(source_arrays[key][2:], replay_arrays[key])
    assert task.physics_total == 97 and all(
        d.task.physics_total == 0 for d in task.drivers.values()
    )
    assert credit.used() < credit.maximum - credit.reserve


@pytest.mark.parametrize(
    "fault",
    [
        "engine",
        "raw",
        "normalized_dtype",
        "normalized_value",
        "normalizer",
        "mid_support",
        "final_engine",
        "cache_force",
        "cache_phase",
        "owner",
        "reference",
        "action",
        "excitation",
        "contact_sign",
        "contact_approach",
        "load",
        "posture",
        "latch",
        "clock",
        "video",
        "missing_array",
    ],
)
def test_saved_semantic_physical_corruption_rejected_even_after_container_rehash(
    proof, fault
):
    task, folder, _, _, _ = proof
    trace = read_json(folder / "source/trace.json")
    arrays = read_arrays(folder / "source/tape.npz")
    assert audit_measurements(task, trace, arrays)["passed"]
    if fault == "engine":
        arrays["decision_engines"][0, 1] += 0.01
    elif fault == "raw":
        arrays["raw"][0, 0, 0] += 0.01
    elif fault == "normalized_dtype":
        arrays["normalized"] = arrays["normalized"].astype(np.float64)
    elif fault == "normalized_value":
        arrays["normalized"][0, 0, 0] += 0.01
    elif fault == "normalizer":
        trace["normalizer"]["mean"][0] += 0.01
    elif fault == "mid_support":
        trace["mid"]["support"]["monitors"]["A"]["tilt_s"] += 0.001
    elif fault == "final_engine":
        trace["last_valid"]["engine"][1] += 0.01
        trace["last_valid"]["sha256"] = digest(
            {k: v for k, v in trace["last_valid"].items() if k != "sha256"}
        )
    elif fault == "cache_force":
        arrays["witness_actuator_force"][0, 0] += 1000
    elif fault == "cache_phase":
        trace["decisions"][0]["tick"] = 20
    elif fault == "owner":
        arrays["raw"][:, [0, 1]] = arrays["raw"][:, [1, 0]]
    elif fault == "reference":
        trace["decisions"][0]["references"]["B"]["init_time"] += 0.02
    elif fault == "action":
        arrays["commands"][0, 0, 0] = 1.1
    elif fault == "excitation":
        arrays["excitation"][0, 0, 0] += 0.01
    elif fault == "contact_sign":
        assert len(arrays["contacts"]) > 0
        arrays["contacts"][0, 6:9] *= -1
    elif fault == "contact_approach":
        arrays["contacts"][0, 16] += 0.01
    elif fault == "load":
        arrays["loads"][0, 0, 0] += 1
    elif fault == "posture":
        arrays["posture"][0, 0, 0] += 0.01
    elif fault == "latch":
        trace["physical_failure"]["A"] = {"tick": 1, "reason": "pelvis_drop"}
    elif fault == "clock":
        arrays["ticks"][0] = 2
    elif fault == "video":
        arrays["video_states"][-1, 1] += 0.01
    else:
        del arrays["witness_qacc"]
    with pytest.raises(ValueError):
        audit_measurements(task, trace, arrays)


def test_motor_sampling_is_read_only_and_wrong_yaw_rejected(proof):
    task, _, tape, _, _ = proof
    initial = copy.deepcopy(tape.initial)
    task.restore(initial)
    before = task.snapshot()
    audit = SemanticInputAudit(task.model, task.drivers["A"].task.model)
    witness = audit.witness(task.data, tick=0)
    assert audit.verify(
        witness,
        task.state(),
        task.inputs,
        {p: d.reference for p, d in task.drivers.items()},
        game_id=task.game_id,
        tick=0,
    )["passed"]
    assert task.snapshot() == before
    with pytest.raises(ValueError, match="clock"):
        audit.witness(task.data, tick=20)
    b = task.model.body("B_pelvis").id
    old = task.model.body_quat[b].copy()
    try:
        task.model.body_quat[b] = [0.0, 0.0, 0.0, 1.0]
        with pytest.raises(ValueError, match="yaw"):
            SemanticInputAudit(task.model, task.drivers["A"].task.model)
    finally:
        task.model.body_quat[b] = old


def test_no_replay_without_actual_prefix_and_no_duplicate_capture(proof):
    task, _, tape, _, _ = proof
    task.restore(tape.mid["task"])
    with pytest.raises(ValueError, match="prefix"):
        MotorTape(task)
    continuation = MotorTape(task, continuation=tape.mid["support"])
    contacts = task.advance_substep(lambda _: None)
    continuation.capture_substep(task, contacts)
    with pytest.raises(ValueError, match="duplicate"):
        continuation.capture_substep(task, contacts)


@pytest.mark.parametrize(
    "fault", [None, "held_cache", "contact", "state", "support", "reference"]
)
def test_saved_exact_continuation_includes_solver_cache_contacts_and_support(
    proof, fault
):
    _, folder, _, _, _ = proof
    source_trace = read_json(folder / "source/trace.json")
    replay_trace = read_json(folder / "replay/trace.json")
    source = read_arrays(folder / "source/tape.npz")
    replay = read_arrays(folder / "replay/tape.npz")
    assert exact_tail(source_trace, source, replay_trace, replay)["passed"]
    if fault is None:
        return
    if fault == "held_cache":
        replay_trace["initial"]["inputs"]["A"]["values"][170] += 0.01
    elif fault == "contact":
        replay["contacts"][0, 6] += 0.01
    elif fault == "state":
        replay_trace["state_sha256"][0] = "0" * 64
    elif fault == "support":
        replay_trace["support_prefix"]["monitors"]["A"]["tilt_s"] += 0.001
    else:
        replay_trace["decisions"][0]["references"]["A"]["init_time"] += 0.02
    with pytest.raises(ValueError):
        exact_tail(source_trace, source, replay_trace, replay)


def packet_fixture():
    return {
        "schema": "joxing-shared-motor-physical-v1",
        "rows": planned_rows(),
        "planned_physics": 17397,
        "physics_cap": 60000,
        "runner_wall_s": 180,
        "post_gate_wall_s": 300,
        "raw_bytes": 32 * 1024**2,
        "formal_simulation_workers": 1,
        "threads_per_process": 1,
        "prior_v11_physics": 5390,
        "prior_v11_raw_bytes": 202302252,
        "prior_v11_post_gate_upper_s": 600,
        "decoder_amplitude": 0.075,
        "source_host": "xbabe1",
    }


@pytest.mark.parametrize(
    "fault",
    [None, "seed", "rows", "physics", "wall", "raw", "workers", "prior", "decoder"],
)
def test_physical_packet_never_changes_denominator_caps_or_decoder(fault):
    p = packet_fixture()
    validate_packet(p)
    if fault is None:
        assert sum(r["planned_physics"] for r in p["rows"]) == 17397
        return
    if fault == "seed":
        p["rows"][0]["seed"] += 1
    elif fault == "rows":
        p["rows"].pop()
    elif fault == "physics":
        p["physics_cap"] += 1
    elif fault == "wall":
        p["runner_wall_s"] += 1
    elif fault == "raw":
        p["raw_bytes"] += 1
    elif fault == "workers":
        p["formal_simulation_workers"] = 2
    elif fault == "prior":
        p["prior_v11_raw_bytes"] -= 1
    else:
        p["decoder_amplitude"] = None
    with pytest.raises(ValueError):
        validate_packet(p)


def test_model_setup_failure_retains_every_not_run_slot_and_zero_physics(
    tmp_path, monkeypatch
):
    import joxing.shared_motor_proof as module

    monkeypatch.setattr(module, "verify_frozen", lambda *args: None)
    monkeypatch.setattr(module, "manifest", lambda: {"scope": "fixture"})

    def model_failure(*args):
        raise ValueError("fixture model loader failed")

    monkeypatch.setattr(module.FrozenPairTeacher, "from_inputs", model_failure)
    p = packet_fixture() | {"inputs_root": "fixture-no-assets"}
    result = run_proof(
        p, tmp_path, tmp_path / "proof", source_sha="fixture", protocol_sha="fixture"
    )
    assert result["status"] == "halted" and result["physics_attempted"] == 0
    assert len(result["rows"]) == 5
    assert all(
        r["status"] == "not_run" and r["physics_attempted"] == 0 for r in result["rows"]
    )
    assert (
        json.loads((tmp_path / "proof/fault.json").read_text())["physics_attempted"]
        == 0
    )


@pytest.mark.parametrize(
    "fault", [None, "count", "extra_xfail", "wrong_xfail", "no_xfail"]
)
def test_linux_gate_requires_exact_pass_count_and_only_the_known_existing_failure(
    fault,
):
    known = (
        "grokbot/tests/test_referee_fuzz.py::test_restore_does_not_alias_the_snapshot"
    )
    p = {"expected_tests": 123, "expected_xfail_nodes": [known]}
    text = f"XFAIL {known} - existing strict snapshot alias defect\n123 passed, 1 xfailed in 10s\n"
    validate_test_summary(p, text)
    if fault is None:
        return
    if fault == "count":
        text = text.replace("123 passed", "122 passed")
    elif fault == "extra_xfail":
        text += "XFAIL tests/test_motor.py::broken - unexpected\n"
    elif fault == "wrong_xfail":
        p["expected_xfail_nodes"] = ["tests/test_motor.py::broken"]
    else:
        text = "123 passed in 10s\n"
    with pytest.raises(ValueError):
        validate_test_summary(p, text)
