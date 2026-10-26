"""Physical invariants and semantic reference mapping, without large checkpoints."""

import copy

import numpy as np
import pytest

from joxing.native_transfer import (
    ReferenceBodyMap,
    TransferTask,
    feature_oracle,
    native_invariants,
    protocol,
    validate_protocol,
)
from joxing.provenance import digest

pytest.importorskip("stable_baselines3")


def test_protocol_and_denominator_are_frozen():
    p = protocol()
    assert validate_protocol(p) == p
    assert len(p["seeds"]) * len(p["variants"]) == 80
    for key, value in [
        ("seeds", list(range(72001, 72021))),
        ("decoder_amplitude", 0.1),
    ]:
        q = copy.deepcopy(p)
        q[key] = value
        with pytest.raises(ValueError):
            validate_protocol(q)


def test_reference_indices_survive_insertions_and_reject_missing_duplicate_names():
    m = ReferenceBodyMap.from_names(
        ["world", "pelvis", "hand", "head"],
        ["world", "pelvis", "hand", "glove", "head"],
    )
    assert m.indices(["head", "pelvis"]) == ([3, 1], [4, 1])
    with pytest.raises(ValueError, match="Missing"):
        ReferenceBodyMap.from_names(["head"], ["hand"])
    with pytest.raises(ValueError, match="Duplicate"):
        ReferenceBodyMap.from_names(["head"], ["head", "head"])


@pytest.fixture(params=range(4))
def task(request):
    p = protocol()
    t = TransferTask(p["environment"], p["variants"][request.param])
    t.reset(73091)
    yield t
    t.close()


def test_native_anatomy_equipment_phase_and_reference_features(task):
    assert (task.model.nq, task.model.nv, task.model.nu, task.model.na) == (
        85,
        85,
        700,
        700,
    )
    assert task.env.dt == 0.02
    assert task.env.frame_skip == (10 if task.variant["physics_dt_s"] == 0.002 else 20)
    assert task.env.cycles == 18
    reference, actual = task.mapping.indices(task.env.key_body_names)
    pelvis_ref, pelvis_actual = task.mapping.indices(["pelvis"])
    obs, blocks = task.env._get_obs()
    assert len(obs) == 3601
    expected = task.env.xpos_ref[reference] - task.env.xpos_ref[pelvis_ref[0]]
    assert np.array_equal(blocks["key_xpos_ref"], expected.ravel())
    assert np.array_equal(
        blocks["key_xpos"],
        (task.data.xpos[actual] - task.data.xpos[pelvis_actual[0]]).ravel(),
    )
    if task.variant["gloves"]:
        assert actual != reference  # The silent upstream indexing defect is exercised.
        assert task.invariants["mass_increment_kg"] == pytest.approx(0.566, abs=1e-12)
    else:
        assert task.invariants["mass_increment_kg"] == 0
    assert len(task.env.xpos_ref) == len(task.mapping.names)
    before = task.data.act.copy()
    task.env.set_state = lambda *a: pytest.fail("Per-step kinematic assistance")
    task.step(np.full(700, -0.6, dtype=np.float32))
    assert (
        task.physics_total == task.env.frame_skip
        and task.data.time == pytest.approx(0.02)
    )
    assert np.any(task.data.act != before)


def test_all_four_variants_have_exact_paired_reset_fields():
    p = protocol()
    fields = []
    for variant in p["variants"]:
        t = TransferTask(p["environment"], variant)
        try:
            t.reset(73092)
            fields.append(digest(t.paired_fields()))
        finally:
            t.close()
    assert len(set(fields)) == 1


def test_native_feature_reward_and_state_oracle():
    result = feature_oracle(protocol()["environment"])
    assert (
        result["passed"]
        and result["max_feature_error"] == 0
        and result["physics_steps"] > 0
    )


def test_cached_inputs_and_application_state_replay_without_pose_writes(task):
    def action():
        return np.tanh(np.resize(task.last_obs, 700) * 0.001)

    task.step(action())
    snapshot = task.snapshot()
    actions = []
    for _ in range(5):
        if task.done:
            break
        a = action()
        actions.append(a)
        task.step(a)
    expected = task.snapshot()
    for closed in [False, True]:
        task.restore(copy.deepcopy(snapshot))
        for a in actions:
            task.step(action() if closed else a)
        assert (
            np.max(abs(np.asarray(task.snapshot()["state"]) - expected["state"])) < 1e-9
        )
        assert np.array_equal(task.last_obs, expected["last_raw_observation"])
    bad = copy.deepcopy(snapshot)
    bad["contract_hash"] = "other"
    before = task.data.qpos.copy()
    with pytest.raises(ValueError):
        task.restore(bad)
    assert np.array_equal(before, task.data.qpos)
    task.model.body_gravcomp[1] = 1
    if task.done:
        task.restore(snapshot)
    with pytest.raises(RuntimeError, match="assistance"):
        task.step(np.zeros(700))


def test_native_inertia_mutation_is_rejected(task):
    import mujoco

    from joxing.upstream_motor import MODEL_ROOT

    canonical = mujoco.MjModel.from_xml_path(str(MODEL_ROOT / "MS-Human-700.xml"))
    task.model.body_mass[task.model.body("pelvis").id] += 0.01
    with pytest.raises(ValueError, match="body_mass"):
        native_invariants(canonical, task.model, task.mapping, task.variant)


def test_all_eighty_outcomes_and_faults_remain_visible():
    from joxing.native_transfer import planned_rows, summarize

    rows = planned_rows(protocol())
    s = summarize(rows)
    assert s["attempts"] == 80 and all(
        v["not_run"] == 20 for v in s["variants"].values()
    )
    rows[0].update(status="invalid", reason="deliberate fault")
    assert summarize(rows)["invalid"] == 1
    rows[0] = dict(rows[1])
    with pytest.raises(ValueError, match="denominator"):
        summarize(rows)


def test_precondition_failure_preserves_all_unattempted_rows(tmp_path, monkeypatch):
    import json

    from joxing.native_transfer import run_transfer
    from joxing.provenance import ROOT

    monkeypatch.setattr(
        "joxing.wave.check_resources",
        lambda *a: {"disk_free_bytes": 10**12, "available_memory_bytes": 10**12},
    )
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"wrong archive")
    out = tmp_path / "failed"
    result = run_transfer(ROOT / "experiments/v0.0.7/protocol.json", archive, out)
    assert not result["all_attempts_completed"] and result["physics_steps"] == 0
    assert len(json.loads((out / "evaluations.json").read_text())) == 80
    assert (out / "fault.json").exists() and json.loads(
        (out / "process.json").read_text()
    )["status"] == "finished"
    with pytest.raises(FileExistsError):
        run_transfer(ROOT / "experiments/v0.0.7/protocol.json", archive, out)


def test_worker_failure_writes_only_its_own_receipts(tmp_path):
    import json
    import time

    from joxing.native_transfer import variant_worker

    p = protocol()
    out = tmp_path / "native_2ms"
    result = variant_worker(
        str(tmp_path / "missing-inputs"),
        p,
        p["variants"][0],
        str(out),
        time.perf_counter() + 10,
        100,
    )
    assert result["status"] == "halted" and result["physics_steps"] == 0
    assert (out / "fault.json").exists()
    assert not (
        tmp_path / "stop.json"
    ).exists()  # Parent is the sole shared stop writer.
    rows = json.loads((out / "evaluations.json").read_text())
    assert len(rows) == 20 and all(r["status"] == "not_run" for r in rows)
