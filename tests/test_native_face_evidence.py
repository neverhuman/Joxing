import copy
import json

import mujoco
import pytest

from joxing.native_face_audit import audit
from joxing.native_face_evidence import audit_saved
from joxing.scene import build_scene


@pytest.fixture(scope="module")
def saved():
    def no_steps(*args, **kwargs):
        pytest.fail("Independent kinematic read-audit attempted a physics step")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            patch.setattr(mujoco, name, no_steps)
        original = build_scene()
        report = json.loads(json.dumps(audit(original), allow_nan=False))
        reconstructed = build_scene()
        yield reconstructed, report


def test_separate_native_reconstruction_and_saved_audit_do_not_use_production_adapter(
    saved, monkeypatch
):
    from joxing import native_face_audit, physics
    from joxing.referee import MatchReferee

    def forbidden(*args, **kwargs):
        pytest.fail("Independent auditor reused production adapter or candidate audit")

    monkeypatch.setattr(MatchReferee, "stage", forbidden)
    monkeypatch.setattr(physics, "point_velocity", forbidden)
    monkeypatch.setattr(native_face_audit, "audit", forbidden)
    result = audit_saved(*saved)
    assert result["status"] == "passed"
    assert result["rows"] == 192 and result["prescribed_states"] == 12
    assert result["new_physics"] == 0
    assert result["production_referee_or_point_velocity_called"] is False
    assert result["native_boxing_proof_passed"] is False


@pytest.mark.parametrize(
    "corruption",
    [
        "point",
        "active",
        "state",
        "duplicate",
        "tolerance",
        "model",
        "nan_velocity",
        "classification",
    ],
)
def test_saved_native_calibration_corruptions_fail(saved, corruption):
    scene, original = saved
    report = copy.deepcopy(original)
    if corruption == "point":
        report["rows"][0]["point_world"][0] += 0.01
    elif corruption == "active":
        report["rows"][0]["staged"]["active_speed_mps"] += 0.01
    elif corruption == "state":
        next(iter(report["kinematic_states"].values()))["qvel"][0] += 0.01
    elif corruption == "duplicate":
        report["rows"][-1] = copy.deepcopy(report["rows"][0])
    elif corruption == "tolerance":
        report["velocity_tolerance_mps"] = 0.1
    elif corruption == "model":
        report["model_binary_sha256"] = "0" * 64
    elif corruption == "nan_velocity":
        report["rows"][0]["velocity_glove_fd"][0] = float("nan")
    else:
        report["rows"][0]["classification"] = "native boxing proof"
    with pytest.raises(ValueError, match="Saved native face"):
        audit_saved(scene, report)
