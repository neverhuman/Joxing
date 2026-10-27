"""Candidate collector preserves force ownership, cached physics and held replay."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from joxing.physical_fast import AllocationReducedTask, floor_evidence
from joxing.physical_motor import PhysicalMotorTask, protocol, signed_floor_evidence
from joxing.provenance import digest


def fake_contacts(monkeypatch, reverse=False):
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    contact = SimpleNamespace(
        geom1=4 if reverse else 0,
        geom2=0 if reverse else 4,
        efc_address=0,
        frame=rotation.ravel(),
        pos=np.array([1.0, 2.0, 3.0]),
    )

    def force(model, data, slot, wrench):
        assert np.array_equal(wrench, np.zeros(6))
        wrench[:3] = [7.0, 0.0, 2.0]  # Sparse fake writes test explicit zeroing.

    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    return SimpleNamespace(contact=[contact]), {4: "right"}


@pytest.mark.parametrize("reverse", [False, True])
def test_floor_evidence_rotation_geom_sign_and_owned_native_float_values(
    monkeypatch, reverse
):
    data, roles = fake_contacts(monkeypatch, reverse)
    expected = signed_floor_evidence(None, data, 0, roles)
    buffer = np.full(6, 50.0)
    actual = floor_evidence(None, data, 0, roles, buffer)
    assert actual == expected
    assert all(type(v) is float for v in actual[1][0][3:])
    before = np.asarray(actual[1]).copy()
    data.contact[0].pos[:] = -100
    buffer[:] = 200
    assert np.array_equal(actual[1], before)


@pytest.mark.parametrize("invalid", ["force", "position"])
def test_floor_evidence_rejects_nonfinite_fields(monkeypatch, invalid):
    data, roles = fake_contacts(monkeypatch)
    if invalid == "force":
        monkeypatch.setattr(
            mujoco, "mj_contactForce", lambda m, d, s, w: w.fill(np.nan)
        )
    else:
        data.contact[0].pos[2] = np.nan
    for collector in (
        lambda: signed_floor_evidence(None, data, 0, roles),
        lambda: floor_evidence(None, data, 0, roles, np.empty(6)),
    ):
        with pytest.raises(RuntimeError, match="Nonfinite"):
            collector()


def test_floor_evidence_keeps_missing_and_mutated_role_faults(monkeypatch):
    data, _roles = fake_contacts(monkeypatch)
    for bad, error in [({}, KeyError), ({4: "unknown"}, ValueError)]:
        for collector in (
            lambda bad=bad: signed_floor_evidence(None, data, 0, bad),
            lambda bad=bad: floor_evidence(None, data, 0, bad, np.empty(6)),
        ):
            with pytest.raises(error):
                collector()
    data.contact[0].efc_address = -1
    assert floor_evidence(None, data, 0, {}, np.empty(6)) == ((0.0, 0.0, 0.0), ())


@pytest.mark.parametrize("variant_index", [0, 1])
def test_engine_every_cached_feature_contact_and_reward_matches_reference(
    variant_index,
):
    p = protocol()
    baseline = PhysicalMotorTask(p["environment"], p["variants"][variant_index])
    candidate = AllocationReducedTask(p["environment"], p["variants"][variant_index])
    try:
        baseline.reset(78091)
        candidate.reset(78091)
        assert (
            baseline.contract == candidate.contract
            and candidate.model.nu == candidate.model.na == 700
        )
        for _ in range(4):
            action = np.tanh(np.resize(baseline.last_obs, 700) * 0.001)
            a, b = baseline.step(action), candidate.step(action)
            assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]
            assert digest(baseline.snapshot()) == digest(candidate.snapshot())
            assert baseline.evidence == candidate.evidence
            if baseline.done:
                break
        assert baseline.physics_total == candidate.physics_total
    finally:
        baseline.close()
        candidate.close()


def test_cross_implementation_mid_control_restore_preserves_held_action():
    p = protocol()
    baseline = PhysicalMotorTask(p["environment"], p["variants"][0])
    candidate = AllocationReducedTask(p["environment"], p["variants"][0])
    try:
        baseline.reset(78092)
        baseline.begin_control(np.tanh(np.resize(baseline.last_obs, 700) * 0.001))
        baseline.advance_substeps(3)
        candidate.restore(baseline.snapshot())
        assert candidate.pending and candidate.substep == 3
        assert candidate._contact_wrench.shape == (6,)
        baseline.resume_control()
        candidate.resume_control()
        assert digest(baseline.snapshot()) == digest(candidate.snapshot())
        assert candidate.physics_total == 7 and baseline.physics_total == 10
    finally:
        baseline.close()
        candidate.close()


def test_candidate_still_checks_assistance_before_every_attempted_substep():
    p = protocol()
    candidate = AllocationReducedTask(p["environment"], p["variants"][0])
    try:
        candidate.reset(78093)
        candidate.begin_control(np.zeros(700, dtype=np.float32))
        candidate.data.xfrc_applied[1, 0] = 1.0
        with pytest.raises(RuntimeError, match="assistance"):
            candidate.advance_substeps(1)
        assert candidate.physics_total == 0
    finally:
        candidate.close()
