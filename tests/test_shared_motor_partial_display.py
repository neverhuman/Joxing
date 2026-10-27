import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "partial_motor_display", ROOT / "scripts/render_shared_motor_partial.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "fault",
    [None, "source", "incomplete", "rows", "audit", "promotion", "replay", "frames"],
)
def test_partial_display_keeps_failed_qualification_and_first_source_identity(fault):
    result = {
        "status": "halted",
        "rows": [
            {
                "id": "source-71000",
                "seed": 71000,
                "status": "completed",
                "physics_attempted": 3520,
            },
            {},
            {},
            {},
            {},
        ],
    }
    audit = {
        "passed": False,
        "short_motor_gate_passed": False,
        "new_audit_physics": 0,
        "exact_replay": {"passed": False},
        "findings": [
            {
                "id": "source-71000",
                "passed": True,
                "policy_inference_audited": True,
                "measured_intervals": 3520,
                "video_frames": 89,
            }
        ],
    }
    module.validate_partial_display(result, audit)
    if fault is None:
        return
    if fault == "source":
        result["rows"][0]["id"] = "source-71001"
    elif fault == "incomplete":
        result["rows"][0]["status"] = "incomplete"
    elif fault == "rows":
        result["rows"].pop()
    elif fault == "audit":
        audit["findings"][0]["passed"] = False
    elif fault == "promotion":
        audit["short_motor_gate_passed"] = True
    elif fault == "replay":
        audit["exact_replay"]["passed"] = True
    else:
        audit["findings"][0]["video_frames"] = 88
    with pytest.raises(ValueError):
        module.validate_partial_display(result, audit)
