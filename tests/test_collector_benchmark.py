"""All collector attempts remain registered; startup faults have zero physics."""

import json
import runpy
import time

import numpy as np

from joxing.physical_motor import PhysicalMotorTask, protocol
from joxing.provenance import ROOT, digest

helpers = runpy.run_path(str(ROOT / "scripts/benchmark_collector.py"))


def test_protocol_denominator_counterbalance_and_remaining_mission_budget():
    p = json.loads((ROOT / "experiments/v0.0.9/collector-protocol.json").read_text())
    assert digest(p) == helpers["PROTOCOL_SHA"]
    waves = helpers["schedule"](p)
    assert [w["arm"] for w in waves] == [
        "baseline",
        "allocation_reduced",
        "allocation_reduced",
        "baseline",
        "baseline",
        "allocation_reduced",
    ]
    cases = [row for wave in waves for row in wave["cases"]]
    assert len(cases) == len({r["case_id"] for r in cases}) == 24
    assert (
        24 * 300 * 10 + 10394
        == p["planned_physics_steps"]
        <= p["caps"]["physics_steps"]
    )
    assert p["criteria"]["minimum_steady_speedup"] == 1.2
    assert (
        p["mission_accounting"]["previous_physics_steps"] + p["caps"]["physics_steps"]
        < 500000
    )
    assert p["mission_accounting"]["previous_formal_wall_s"] + p["caps"]["wall_s"] < 900


def test_unknown_implementation_preserves_failure_receipt_without_model_load(tmp_path):
    row = {"case_id": "unknown", "arm": "unknown", "seed": 78000, "status": "not_run"}
    result = helpers["worker"](
        tmp_path,
        {},
        row,
        tmp_path / "worker",
        tmp_path / "stop.json",
        time.monotonic() + 10,
    )
    assert result["phase"] == "failed" and result["case"]["status"] == "invalid"
    assert (
        result["case"]["physics_steps"] == 0 and "Unknown collector" in result["error"]
    )
    assert (tmp_path / "worker/progress.json").exists()


def test_nonfinite_fault_state_is_retained_without_json_masking(tmp_path):
    p = protocol()
    task = PhysicalMotorTask(p["environment"], p["variants"][0])
    try:
        task.reset(78094)
        snapshot = task.snapshot()
        task.data.qvel[0] = np.nan  # No physics; deliberate fault-evidence fixture.
        helpers["save_fault"](tmp_path, task, RuntimeError("fixture"), snapshot)
        info = json.loads((tmp_path / "fault.json").read_text())
        assert info["last_valid_snapshot"] == snapshot and info["reason"] == "fixture"
        with np.load(tmp_path / "fault-arrays.npz", allow_pickle=False) as raw:
            assert np.isnan(raw["state"]).any()
        assert task.physics_total == 0
    finally:
        task.close()
