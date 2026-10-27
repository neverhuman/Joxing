"""Reject corrupted role/load/tape evidence without stepping a simulator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "collector_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_collector.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def evidence():
    return {
        "intervals_s": np.array([[0.0, 0.002]]),
        "loads_N": np.array([[0.0, 3.0, 0.0]]),
        "tilt_drop": np.zeros((1, 2)),
        "contact_offsets": np.array([0, 2]),
        "contact_geom_ids": np.array([[0, 7], [7, 0]], dtype=np.int32),
        "contact_role": np.array([1, 1], dtype=np.int8),
        "contact_force_N": np.array([[1.0, 2.0, 3.0], [0.0, 0.0, -4.0]]),
        "contact_position_m": np.zeros((2, 3)),
    }


def test_signed_load_and_semantic_role_corruption():
    data = evidence()
    assert audit.check_contacts(data, 1, 0, {"7": "left"}) == 2
    with pytest.raises(AssertionError):
        audit.check_contacts(data, 1, 0, {"7": "right"})
    data["loads_N"][0, 1] = 7.0  # Absolute-force accumulation is incorrect.
    with pytest.raises(AssertionError):
        audit.check_contacts(data, 1, 0, {"7": "left"})


def test_raw_nonfinite_and_changed_dtype_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, actions=np.array([np.nan]))
    with pytest.raises(AssertionError):
        audit.arrays(path)
    with pytest.raises(AssertionError):
        audit.same_arrays(
            {"x": np.array([1.0])}, {"x": np.array([1.0], dtype=np.float32)}
        )
    with pytest.raises(AssertionError):
        audit.same_arrays({"x": np.array([1.0])}, {"x": np.array([2.0])})
