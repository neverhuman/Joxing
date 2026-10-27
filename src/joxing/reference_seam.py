"""Explicit reference variant: correct only translated seam root velocities.

This does not repair motor balance or change native geometry, actor inputs,
reference positions, timing, or the default PhysicalMotorTask.
"""

import copy

VARIANT = "unwrapped_seam_root_velocity_v1"


class UnwrappedSeamTrajectory:
    def __init__(self, legacy):
        self._trajectory = copy.deepcopy(legacy)
        self.corrections = []
        for index, trajectory in enumerate(self._trajectory.trajectories):
            if trajectory["qpos_traj"].shape[1] != 85:
                raise ValueError("Seam variant requires native85 scalar coordinates")
            duration = float(trajectory["terminate_time"])
            dt = float(trajectory["dt"])
            following = legacy.query(duration, index)[0]
            last = trajectory["qpos_traj"][-1]
            velocity = trajectory["qvel_traj"]
            before = velocity[-1].copy()
            # These are precisely the two coordinates translated by the pinned
            # native query; leave every other legacy component bit-identical.
            for coordinate in (0, 2):
                velocity[-1, coordinate] = (
                    float(following[coordinate]) - float(last[coordinate])
                ) / dt
            self.corrections.append(
                {
                    "trajectory_index": index,
                    "before": before.tolist(),
                    "after": velocity[-1].tolist(),
                }
            )

    @property
    def trajectories(self):
        return self._trajectory.trajectories

    @property
    def num_trajectories(self):
        return self._trajectory.num_trajectories

    def get_trajectory_properties(self, index):
        return self._trajectory.get_trajectory_properties(index)

    def query(self, time, index):
        return self._trajectory.query(time, index)

    def query_batch(self, times, index):
        return self._trajectory.query_batch(times, index)
