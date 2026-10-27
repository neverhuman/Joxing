"""Offline normalization/reference audit catches concrete input inconsistencies."""

import runpy

import numpy as np
import pytest

from joxing.provenance import ROOT

audit = runpy.run_path(str(ROOT / "scripts/audit_motor_inputs.py"))


def test_normalization_uses_frozen_variance_and_partitions_every_coordinate():
    normalizer = {"mean": [1.0, 0.0], "var": [4.0, 1.0], "epsilon": 1e-8, "clip": 2.0}
    stats = audit["normalized_statistics"](
        np.array([[1.0, -3.0], [5.0, 0.0]], dtype=np.float32),
        normalizer,
        {"a": [0, 1], "b": [1, 2]},
    )
    assert stats["a"]["fraction_clipped"] == 0
    assert stats["b"]["clipped_count_by_control"] == [1, 0]
    with pytest.raises(ValueError, match="partition"):
        audit["normalized_statistics"](np.ones((2, 2)), normalizer, {"missing": [0, 1]})
    with pytest.raises(ValueError, match="Malformed"):
        audit["normalized_statistics"](
            np.full((2, 2), np.nan), normalizer, {"all": [0, 2]}
        )


def synthetic_motion(path, offset):
    # Four frames with a translating root and one limb coordinate.
    q = np.zeros((4, 85), dtype=np.float32)
    q[:, 0] = np.array([0.0, 0.1, 0.2, 0.3]) + offset
    np.savez(
        path,
        qpos_traj=q,
        xpos_traj=np.zeros((4, 2, 3)),
        framerate=np.array(100.0),
        velocity=1.0,
        period=0.04,
        stride=0.3,
    )


def test_seam_witness_compares_unwrapped_positions_not_wrapped_difference(tmp_path):
    motion = tmp_path / "motion.npz"
    synthetic_motion(motion, 0)
    query = audit["load_query"](
        ROOT / "third_party/msgym/msgym/envs/imitation_trajectory.py", motion
    )
    witness = audit["seam_witness"](query)
    assert witness["max_error_coordinate"] == 0
    assert witness["reported_qvel"][0] == pytest.approx(-30)
    assert witness["unwrapped_forward_difference_qvel"][0] == 0
    assert witness["max_absolute_error"] == pytest.approx(30)
    # Returning the matching unwrapped velocity removes the witness.
    query.trajectories[0]["qvel_traj"][-1, 0] = 0
    assert audit["seam_witness"](query)["max_absolute_error"] == 0
