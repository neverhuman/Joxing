import json
from collections import Counter
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from joxing import native_face_audit as observer
from joxing.referee import MatchReferee
from joxing.scene import build_scene


@pytest.fixture(scope="module")
def native():
    def no_steps(*args, **kwargs):
        pytest.fail("A kinematic observer attempted a physics step")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            patch.setattr(mujoco, name, no_steps)
        scene = build_scene()
        result = observer.audit(scene)
        yield scene, result


def test_all_native_hands_front_back_and_both_zones_have_fixed_denominator(native):
    scene, result = native
    assert scene.model.nu == scene.model.na == 1400
    assert result["row_count"] == 192
    assert Counter(r["front"] for r in result["rows"]) == {True: 96, False: 96}
    assert {r["glove"] for r in result["rows"]} == {
        "A_glove_l",
        "A_glove_r",
        "B_glove_l",
        "B_glove_r",
    }
    assert {r["zone"] for r in result["rows"]} == {"head", "torso"}
    assert all(r["staged"]["striking_face"] == r["front"] for r in result["rows"])


def test_independent_material_point_fd_and_nonzero_rotational_witness(native):
    _, result = native
    assert result["maximum_velocity_error_mps"] < 2e-7
    assert result["maximum_point_rotation_witness_mps"] > 1e-4
    passive = [r for r in result["rows"] if r["profile"] == "opponent-only"]
    assert all(abs(r["staged"]["active_speed_mps"]) < 2e-7 for r in passive)
    assert any(abs(r["staged"]["approach_mps"]) > 1e-4 for r in passive)
    assert len(result["native_yaw_checks"]) == 4


def test_reversed_contact_order_preserves_all_semantic_measurements(native):
    _, result = native
    seen = {}
    for r in result["rows"]:
        key = tuple(r[k] for k in ("pose", "profile", "glove", "zone", "front"))
        value = {k: v for k, v in r["staged"].items() if k != "sample_id"}
        if key in seen:
            assert value == seen[key]
        seen[key] = value
    assert len(seen) == 96


def test_no_synthetic_case_claims_a_real_punch_motor_or_anatomical_calibration(native):
    _, result = native
    assert result["new_physics"] == result["new_optimizer_updates"] == 0
    assert result["referee_commits"] == 0
    assert result["native_boxing_proof_passed"] is False
    assert len(result["model_binary_sha256"]) == 64
    assert any("anatomical knuckle" in s for s in result["limitations"])
    assert any(
        "no collision/force measurement" in r["classification"] for r in result["rows"]
    )


def test_complete_diagnostic_and_journal_fit_four_mib_with_fault_reserve(native):
    scene, result = native
    report_bytes = len(json.dumps(result, indent=2, sort_keys=True).encode()) + 1
    journal_bytes = sum(len(json.dumps(row).encode()) + 1 for row in result["rows"])
    scene_bytes = len(json.dumps(scene.report, indent=2).encode()) + 1
    # Registered reserves: 2 MiB fault, 256 KiB supervision, 128 KiB events/results.
    assert (
        report_bytes
        + journal_bytes
        + scene_bytes
        + 2 * 1024**2
        + 256 * 1024
        + 128 * 1024
        < 4 * 1024**2
    )


def test_center_velocity_substitution_is_detected(native, monkeypatch):
    scene, _ = native
    original = observer.point_velocity
    monkeypatch.setattr(
        observer, "point_velocity", lambda m, d, b, p: original(m, d, b, d.xpos[b])
    )
    with pytest.raises(ValueError, match="independent FK derivative"):
        observer.audit(scene)


def test_face_sign_corruption_is_detected(native, monkeypatch):
    scene, _ = native
    original = MatchReferee.stage

    def wrong_face(self, *args, **kwargs):
        intervals = original(self, *args, **kwargs)
        for name, row in intervals.items():
            if row.touches:
                intervals[name] = replace(
                    row,
                    touches=(
                        replace(row.touches[0], face_cos=-row.touches[0].face_cos),
                    ),
                )
        return intervals

    monkeypatch.setattr(MatchReferee, "stage", wrong_face)
    with pytest.raises(ValueError, match="staged synthetic face/velocity"):
        observer.audit(scene)


def test_asymmetric_native_wrist_pose_is_detected(native, monkeypatch):
    scene, _ = native
    pose = np.array(scene.report["initial_qpos"])
    hand = scene.model.body("B_hand_l").id
    j = int(np.flatnonzero(scene.model.jnt_bodyid == hand)[0])
    pose[scene.model.jnt_qposadr[j]] += 0.02
    monkeypatch.setitem(scene.report, "initial_qpos", pose.tolist())
    with pytest.raises(ValueError, match="yaw"):
        observer.audit(scene)


@pytest.mark.parametrize(
    "arguments",
    [{"epsilon": 0}, {"velocity_tolerance": 1e-3}, {"geometry_tolerance": -1}],
)
def test_invalid_calibration_tolerances_rejected(native, arguments):
    with pytest.raises(ValueError, match="tolerances"):
        observer.audit(native[0], **arguments)
