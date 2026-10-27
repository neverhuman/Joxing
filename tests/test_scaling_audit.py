"""Stopped scaling stays visible; lost cases or trajectory mutations fail audit."""

import copy
import json
import runpy

import pytest

from joxing.provenance import ROOT

helpers = runpy.run_path(str(ROOT / "scripts/audit_physical_scaling.py"))
folder = ROOT / "artifacts/v0.0.9/H2/attempt-01"


def test_actual_capacity_stop_retains_all_seventy_eight_cases():
    result = helpers["audit"](folder)
    assert result["passed"] and not result["all_scaling_cases_completed"]
    assert result["case_counts"] == {"completed": 31, "not_run": 47}
    assert result["wave_counts"] == {"completed": 4, "invalid": 1, "not_run": 5}
    assert result["physics_steps"] == 37200 and result["physics_accounting_exact"]
    assert result["parallel_completed_cases_matching_serial"] == 15


@pytest.mark.parametrize("fault", ["missing_case", "changed_step", "false_physics"])
def test_missing_or_mutated_evidence_cannot_pass(fault):
    report = copy.deepcopy(json.loads((folder / "result.json").read_text()))
    protocol = json.loads((folder / "protocol.json").read_text())
    if fault == "missing_case":
        report["waves"][-1]["cases"].pop()
    elif fault == "changed_step":
        report["waves"][0]["cases"][0]["step_hashes"][0] = "0" * 64
    else:
        report["physics_steps"] -= 1
    with pytest.raises(AssertionError):
        helpers["check_registry"](report, protocol)
