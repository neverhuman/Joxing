"""Seam correction owns its data and changes only the declared velocity row."""

import runpy

import numpy as np

from joxing.provenance import ROOT
from joxing.reference_seam import UnwrappedSeamTrajectory

helpers = runpy.run_path(str(ROOT / "tests/test_motor_inputs.py"))
audit = runpy.run_path(str(ROOT / "scripts/audit_motor_inputs.py"))


def test_variant_changes_seam_root_only_and_preserves_legacy_arrays(tmp_path):
    motion = tmp_path / "motion.npz"
    helpers["synthetic_motion"](motion, 0)
    legacy = audit["load_query"](
        ROOT / "third_party/msgym/msgym/envs/imitation_trajectory.py", motion
    )
    before = legacy.trajectories[0]["qvel_traj"].copy()
    fixed = UnwrappedSeamTrajectory(legacy)
    assert np.array_equal(legacy.trajectories[0]["qvel_traj"], before)
    assert not np.shares_memory(before, fixed.trajectories[0]["qvel_traj"])
    assert audit["seam_witness"](legacy)["max_absolute_error"] > 29
    assert audit["seam_witness"](fixed)["max_absolute_error"] == 0
    times = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.07])
    old = legacy.query_batch(times, 0)
    new = fixed.query_batch(times, 0)
    assert np.array_equal(old[0], new[0]) and np.array_equal(old[1], new[1])
    changed = np.argwhere(old[2] != new[2])
    assert changed.tolist() == [[3, 0], [5, 0]]
    for time in times:
        old, new = legacy.query(float(time), 0), fixed.query(float(time), 0)
        assert np.array_equal(old[0], new[0]) and np.array_equal(old[1], new[1])
        assert np.array_equal(new[2][1:], old[2][1:])


def test_new_reference_is_independent_from_mutating_legacy_after_creation(tmp_path):
    motion = tmp_path / "motion.npz"
    helpers["synthetic_motion"](motion, 0)
    legacy = audit["load_query"](
        ROOT / "third_party/msgym/msgym/envs/imitation_trajectory.py", motion
    )
    fixed = UnwrappedSeamTrajectory(legacy)
    expected = fixed.query(0.01, 0)[0]
    legacy.trajectories[0]["qpos_traj"][:] = 999
    assert np.array_equal(fixed.query(0.01, 0)[0], expected)
