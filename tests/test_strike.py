import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from joxing.scene import SOURCE
from joxing.strike import StrikeConfig, StrikeRig, initial_strike_policy


@pytest.fixture(scope="module")
def rig():
    return StrikeRig(StrikeConfig())


def test_rig_keeps_native_muscles_and_explicit_assistance(rig):
    assert rig.model.nu == rig.model.na == 700
    assert rig.model.nq == 79
    assert rig.model.opt.gravity == pytest.approx([0, 0, 0])
    assert rig.model.body("strike_glove").mass[0] == pytest.approx(0.283)
    groups = np.concatenate(rig.groups)
    assert len(groups) == len(set(groups))


def test_added_glove_preserves_native_muscle_parameters_and_segment_inertia(rig):
    source = mujoco.MjModel.from_xml_path(str(SOURCE))
    ids = [rig.model.actuator(source.actuator(i).name).id for i in range(source.nu)]
    for field in (
        "actuator_dynprm",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
    ):
        assert np.array_equal(getattr(rig.model, field)[ids], getattr(source, field))
    bodies = [rig.model.body(source.body(i).name).id for i in range(source.nbody)]
    assert rig.model.body_mass[bodies] == pytest.approx(source.body_mass, abs=1e-12)
    assert rig.model.body_inertia[bodies] == pytest.approx(
        source.body_inertia, abs=1e-12
    )


def test_geometry_evidence_uses_hand_frame_and_engine_distance(rig):
    rig.reset(4)
    a = rig.adapter.measure(rig.data, [], rig.data.qvel.copy())
    independent = (
        np.linalg.norm(rig.data.geom_xpos[rig.glove] - rig.data.geom_xpos[rig.target])
        - 0.142
    )
    assert a.clearance_m == pytest.approx(independent)
    assert a.touches == []
    assert np.isfinite(a.extension_m)


def test_full_engine_and_application_restore_matches_continuation(rig):
    rig.reset(10000, np.full((4, 6), -1.5))
    for _ in range(75):
        rig.advance()
    state = json.loads(json.dumps(rig.snapshot()))
    for _ in range(30):
        rig.advance()
    reference = rig.snapshot()
    rig.restore(state)
    for _ in range(30):
        rig.advance()
    restored = rig.snapshot()
    assert restored["state"] == pytest.approx(reference["state"], abs=1e-10)
    assert restored["tracker"] == reference["tracker"]
    assert restored["stats"] == pytest.approx(reference["stats"])


def test_reset_and_pulses_do_not_write_other_muscle_controls(rig):
    rig.reset(2)
    rig.advance()
    controlled = set(np.concatenate(rig.groups))
    other = [i for i in range(rig.model.nu) if i not in controlled]
    assert np.all(rig.data.ctrl[other] == 0.005)
    with pytest.raises(ValueError):
        rig.reset(1, np.zeros((2, 6)))


def test_invalid_protocol_and_incompatible_snapshot_fail(rig):
    with pytest.raises(ValueError):
        replace(StrikeConfig(), dt_s=0)
    with pytest.raises(ValueError):
        replace(StrikeConfig(), target_zone="unknown")
    rig.reset(9)
    snapshot = rig.snapshot()
    snapshot["model_hash"] = "other-model"
    with pytest.raises(ValueError):
        rig.restore(snapshot)


def test_native_muscle_contact_scores_once_despite_contact_beyond_window(rig):
    episode = rig.rollout(initial_strike_policy(), 10000)
    assert episode["contact_steps"] * rig.config.dt_s > rig.rules.impact_window_s
    assert episode["legal_hits"] == episode["points"] == 1
    assert episode["normal_impulse_Ns"] > episode["events"][0]["window_impulse_Ns"]
    assert not episode["strike_withdrawal_success"]
