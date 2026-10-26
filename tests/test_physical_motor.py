"""Substep timing/contact/replay invariants; no checkpoint or training claims."""

import copy
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from joxing.native_transfer import planned_rows
from joxing.physical_motor import (
    PhysicalEnvironmentMixin,
    PhysicalMotorConfig,
    PhysicalMotorTask,
    equivalence_oracle,
    protocol,
    signed_floor_evidence,
    summarize,
    validate_protocol,
)
from joxing.provenance import digest
from joxing.support import SupportMonitor


@pytest.fixture(params=range(2))
def task(request):
    p = protocol()
    t = PhysicalMotorTask(p["environment"], p["variants"][request.param])
    t.reset(75092)
    yield t
    t.close()


def feedback(task):
    return np.tanh(np.resize(task.last_obs, 700) * 0.001)


def test_protocol_freezes_physics_history_and_denominator():
    p = protocol()
    assert validate_protocol(p) == p
    assert len(planned_rows(p)) == 40
    altered = copy.deepcopy(p)
    altered["physical"]["nonfoot_persistence_s"] = 0.01
    with pytest.raises(ValueError):
        validate_protocol(altered)
    with pytest.raises(ValueError):
        PhysicalMotorConfig(horizon_s=5)
    rows = planned_rows(p)
    s = summarize(rows)
    assert s["attempts"] == 40 and not s["physical_completion_threshold_met"]
    rows[0]["status"] = "invalid"
    assert summarize(rows)["invalid"] == 1
    rows[0] = dict(rows[1])
    with pytest.raises(ValueError, match="denominator"):
        summarize(rows)


def test_explicit_substeps_leave_native_state_cached_features_and_reward_exact():
    p = protocol()
    result = equivalence_oracle(p["environment"], p["variants"])
    assert result["passed"] and result["physics_steps"] > 0
    assert all(
        r["max_state_error"] == r["max_feature_error"] == r["max_reward_error"] == 0
        for r in result["variants"]
    )
    assert all(r["old_task_done"] for r in result["variants"])


def test_scratch_end_posture_is_current_and_does_not_refresh_actor_cache(task):
    assert (task.model.nq, task.model.nv, task.model.nu, task.model.na) == (
        85,
        85,
        700,
        700,
    )
    task.env.set_state = lambda *_: pytest.fail("Per-step pose write")
    old = task.data.act.copy()
    task.step(feedback(task))
    assert np.any(task.data.act != old)
    assert task.physics_total == task.env.frame_skip
    assert len(task.evidence) == task.env.frame_skip
    row = task.evidence[-1]
    reference = mujoco.MjData(task.model)
    reference.qpos[:] = task.data.qpos
    mujoco.mj_forward(task.model, reference)
    up = reference.xmat[task.pelvis].reshape(3, 3)[:, 1]
    assert row.tilt_rad == float(np.arccos(np.clip(up[2], -1, 1)))
    assert row.drop_m == task.initial_height - float(reference.xpos[task.pelvis, 2])
    before = copy.deepcopy(task.snapshot())
    task.telemetry()
    task.telemetry()
    assert digest(before) == digest(task.snapshot())
    assert not np.array_equal(task.data.xpos[task.pelvis], reference.xpos[task.pelvis])


@pytest.mark.parametrize("mid", [False, True])
def test_boundary_and_mid_command_replay_is_exact_without_new_decision(task, mid):
    task.step(feedback(task))
    if mid:
        task.begin_control(feedback(task))
        task.advance_substeps(3)
        assert not np.array_equal(task.last_obs[:85], task.data.qpos.astype(np.float32))
    snapshot = json.loads(json.dumps(task.snapshot()))
    if mid:
        with pytest.raises(RuntimeError, match="pending"):
            task.step(feedback(task))
        task.resume_control()
    actions = []
    for _ in range(4):
        if task.done:
            break
        a = feedback(task)
        actions.append(a)
        task.step(a)
    expected = task.snapshot()
    expected_telemetry = task.telemetry()
    for closed in (False, True, True):
        total_before = task.physics_total
        task.restore(snapshot)
        if mid:
            task.resume_control()  # No policy evaluation at this partial interval.
        for a in actions:
            task.step(feedback(task) if closed else a)
        assert task.physics_total > total_before
        actual = task.snapshot()
        assert np.max(abs(np.asarray(actual["state"]) - expected["state"])) < 1e-9
        assert np.array_equal(
            actual["last_raw_observation"], expected["last_raw_observation"]
        )
        assert actual["physical_control"] == expected["physical_control"]
        assert actual["monitor"] == expected["monitor"]
        assert task.telemetry() == expected_telemetry


def test_malformed_partial_restore_rejects_before_mutation(task):
    task.step(feedback(task))
    task.begin_control(feedback(task))
    task.advance_substeps(3)
    snapshot = task.snapshot()
    changes = [
        lambda s: s.update(contract_hash="foreign"),
        lambda s: s["physical_control"].update(substep=4),
        lambda s: s["physical_control"].update(action=[0] * 700),
        lambda s: s["physical_control"].update(completed_controls=-1),
        lambda s: s["physical_control"].update(interval_evidence=[]),
        lambda s: s["physical_control"].update(physical_failure="excess_tilt"),
        lambda s: s["physical_control"].update(control_start_s=0),
        lambda s: s["monitor"].update(tilt_s=-1),
        lambda s: s["application"].update(qpos_ref=[0]),
        lambda s: s["application"].update(unknown=0),
        lambda s: s["rng"].update(bit_generator="UnknownGenerator"),
        lambda s: s.update(done=True),
    ]
    for change in changes:
        bad = copy.deepcopy(snapshot)
        change(bad)
        before = digest(task.snapshot())
        with pytest.raises((ValueError, TypeError, KeyError)):
            task.restore(bad)
        assert digest(task.snapshot()) == before


def test_monitor_latch_is_substep_time_and_finishes_held_interval(task, monkeypatch):
    # Controlled diagnostic force stream, not a physical performance outcome.
    # Preserve actual engine stepping and test the crossing inside a control.
    loads = (0.0, 0.0, task.weight * 0.2)
    monkeypatch.setattr(
        "joxing.physical_motor.signed_floor_evidence", lambda *_: (loads, ())
    )
    task.monitor.nonfoot_s = 0.049 - task.model.opt.timestep
    task.step(feedback(task))
    assert task.physical_failure == "nonfoot_support"
    expected = 2 * task.model.opt.timestep
    assert task.physical_failure_s == pytest.approx(expected)
    assert task.data.time == pytest.approx(0.02)
    assert task.physical_failure_s < task.data.time
    assert len(task.interval_evidence) == task.env.frame_skip
    with pytest.raises(RuntimeError, match="Reset"):
        task.step(feedback(task))


@pytest.mark.parametrize("dt", [0.001, 0.002])
def test_support_duration_crosses_controls_and_clears_after_brush(dt):
    m = SupportMonitor()
    loaded = {"right": 300.0, "left": 300.0, "nonfoot": 60.0}
    clear = {**loaded, "nonfoot": 0.0}
    for _ in range(round(0.04 / dt)):
        assert m.update(loaded, 600, 0, 0, dt) is None
    assert m.update(clear, 600, 0, 0, dt) is None
    assert m.nonfoot_s == 0
    for _ in range(round(0.05 / dt) - 1):
        assert m.update(loaded, 600, 0, 0, dt) is None
    assert m.update(loaded, 600, 0, 0, dt) == "nonfoot_support"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.7])
def test_contact_rotation_sign_manifolds_and_unknown_roles(reverse, angle, monkeypatch):
    c, s = np.cos(angle), np.sin(angle)
    frame = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    # Synthetic solved cache permits both engine geometry orders and a signed
    # downward tangential resultant; raw evidence must retain negative force.
    contact = SimpleNamespace(
        geom1=4 if reverse else 0,
        geom2=0 if reverse else 4,
        frame=frame.ravel(),
        pos=np.array([1.0, 2.0, 3.0]),
        efc_address=0,
    )
    data = SimpleNamespace(contact=[contact, copy.deepcopy(contact)])
    wrench = np.array([3.0, 0.0, -7.0, 0.0, 0.0, 0.0])

    def force(_m, _d, _slot, out):
        out[:] = wrench

    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    loads, contacts = signed_floor_evidence(None, data, 0, {4: "nonfoot"})
    expected = frame.T @ wrench[:3] * (-1 if reverse else 1)
    assert np.array_equal(contacts[0][3:6], expected)
    assert len(contacts) == 2
    assert loads == (0.0, 0.0, 2 * max(0.0, expected[2]))
    with pytest.raises(KeyError):
        signed_floor_evidence(None, data, 0, {})


def test_real_mixed_floor_manifolds_reconcile_each_body_momentum():
    bodies = "".join(
        f'<body name="b{i}" pos="{i * 0.3} 0 .049"><freejoint/><geom name="g{i}" type="box" size=".05 .05 .05" mass="{i + 2}"/></body>'
        for i in range(3)
    )
    xml = f'<mujoco><option timestep=".001" integrator="Euler"/><worldbody><geom name="floor" type="plane" size="2 2 .1"/>{bodies}</worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    roles = {
        model.geom(f"g{i}").id: role
        for i, role in enumerate(("right", "left", "nonfoot"))
    }
    impulse = np.zeros(3)
    max_contacts = 0
    for _ in range(200):
        mujoco.mj_step(model, data)
        loads, contacts = signed_floor_evidence(model, data, floor, roles)
        max_contacts = max(max_contacts, len(contacts))
        impulse += np.asarray(loads) * 0.001
    assert max_contacts >= 12
    for i in range(3):
        mass = i + 2
        assert impulse[i] - mass * 9.81 * 0.2 == pytest.approx(
            mass * data.qvel[6 * i + 2], abs=1e-9
        )


def test_compact_dump_keeps_every_substep_and_manifold(task, tmp_path):
    task.step(feedback(task))
    path = tmp_path / "substeps.npz"
    task.dump_substeps(path)
    with np.load(path) as z:
        assert len(z["intervals_s"]) == task.env.frame_skip
        assert len(z["contact_offsets"]) == task.env.frame_skip + 1
        assert z["contact_offsets"][-1] == len(z["contact_geom_ids"])
        assert np.array_equal(z["loads_N"], [r.loads_N for r in task.evidence])


def test_fresh_task_restore_does_not_need_an_unregistered_reset(task):
    task.begin_control(feedback(task))
    task.advance_substeps(3)
    snapshot = task.snapshot()
    restored = PhysicalMotorTask(protocol()["environment"], task.variant)
    try:
        restored.restore(snapshot)
        restored.resume_control()
        task.resume_control()
        assert digest(restored.snapshot()) == digest(task.snapshot())
    finally:
        restored.close()


def test_physical_failure_at_horizon_is_not_a_completion():
    class Base:
        def step(self, action):
            return [], 0.0, True, False, {}

    class Boundary(PhysicalEnvironmentMixin, Base):
        data = SimpleNamespace(time=20.0)
        _physical_task = SimpleNamespace(physical=PhysicalMotorConfig())

    assert Boundary().step(None)[2:4] == (True, False)


def test_restore_before_and_after_persistence_latch(task, monkeypatch):
    for _ in range(3):
        task.step(feedback(task))
    other = next(
        k for k, role in task.roles.items() if role == "nonfoot" and k != task.floor
    )
    force = 0.2 * task.weight
    loads = (0.0, 0.0, force)
    contacts = ((task.floor, other, 2, 0.0, 0.0, force, 0.0, 0.0, 0.0),)
    # Explicit measurement fixture, never a physical performance result.
    monkeypatch.setattr(
        "joxing.physical_motor.signed_floor_evidence", lambda *_: (loads, contacts)
    )
    task.monitor.nonfoot_s = 0.048
    task.begin_control(feedback(task))
    before = task.snapshot()
    task.advance_substeps(round(0.002 / task.model.opt.timestep))
    assert task.physical_failure == "nonfoot_support" and not task.done
    after = task.snapshot()
    task.resume_control()
    expected = task.snapshot()
    for snapshot in (before, after):
        task.restore(snapshot)
        task.resume_control()
        assert digest(task.snapshot()) == digest(expected)


def test_failed_prerequisite_keeps_all_forty_rows_and_final_progress(
    tmp_path, monkeypatch
):
    from joxing.physical_motor import run_physical
    from joxing.provenance import ROOT

    monkeypatch.setattr(
        "joxing.wave.check_resources",
        lambda *a: {"disk_free_bytes": 10**12, "available_memory_bytes": 10**12},
    )
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"wrong archive")
    out = tmp_path / "failed"
    result = run_physical(ROOT / "experiments/v0.0.8/protocol.json", archive, out)
    assert result["attempts"] == 40 and result["physics_steps"] == 0
    assert not result["all_attempts_completed"] and (out / "fault.json").exists()
    progress = json.loads((out / "progress.json").read_text())
    assert (
        progress["status"] == "finished"
        and progress["completed"] == 0
        and progress["planned"] == 40
    )
    with pytest.raises(FileExistsError):
        run_physical(ROOT / "experiments/v0.0.8/protocol.json", archive, out)


def test_worker_retains_real_physical_outcome_and_raw_evidence(tmp_path, monkeypatch):
    import time

    from joxing.native_transfer import variant_worker

    # Tiny feedback stub exercises file/attempt bookkeeping, not policy quality.
    normalizer = SimpleNamespace(fingerprint="development", check=lambda: None)
    monkeypatch.setattr(
        "joxing.native_transfer.load_controller", lambda _: (None, normalizer)
    )
    monkeypatch.setattr(
        "joxing.native_transfer.predict",
        lambda _m, _n, obs: (np.tanh(np.resize(obs, 700) * 0.001), 0.0),
    )
    p = protocol()
    p["seeds"] = [75092]  # Development-only worker test; run_physical rejects it.
    out = tmp_path / "native_2ms"
    result = variant_worker(
        str(tmp_path), p, p["variants"][0], str(out), time.perf_counter() + 30, 12000
    )
    assert result["status"] == "completed"
    row = json.loads((out / "evaluations.json").read_text())[0]
    assert row["status"] == "completed" and row["physical_failure"] is not None
    assert "native_terminated" not in row and row["physical_terminated"]
    assert (out / "substeps-75092.npz").exists()
    assert not (tmp_path / "stop.json").exists()
    import runpy

    from joxing.provenance import ROOT

    audit = runpy.run_path(str(ROOT / "scripts/verify_physical_motor.py"))[
        "audit_substeps"
    ]
    contract = json.loads((out / "specification.json").read_text()) | {
        "floor_geom_id": 0
    }
    initial = json.loads((out / "starts/75092.json").read_text())
    audited = audit(out / "substeps-75092.npz", row, contract, initial)
    assert audited["substeps"] == row["physics_steps"]
    with np.load(out / "substeps-75092.npz") as z:
        tampered = {k: z[k].copy() for k in z.files}
    tampered["loads_N"][0, 0] += 1
    np.savez_compressed(out / "tampered.npz", **tampered)
    with pytest.raises(AssertionError):
        audit(out / "tampered.npz", row, contract, initial)
