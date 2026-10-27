"""Semantic readout of each intact native body from one shared MjData.

These are measurements, not an actor observation: do not hand an opponent's
private activation/joint bundle to a policy. Future observations select visible
kinematics and transform them into the observer's frame.
"""

import mujoco
import numpy as np

from .base import BodyMeasurements, owned_readonly

KEY_BODIES = ("pelvis", "head_neck", "hand_r", "hand_l", "calcn_r", "calcn_l")


class NativeBodyAdapter:
    def __init__(self, scene, actor):
        if actor not in {"A", "B"}:
            raise ValueError("Unknown corner")
        self.model, self.actor = scene.model, actor
        self.actuator_ids = owned_readonly(scene.actuator_ids[actor])
        if len(self.actuator_ids) != 700 or len(set(self.actuator_ids)) != 700:
            raise ValueError("Complete 700-muscle ownership required")
        prefix = actor + "_"
        joints = [
            j
            for j in range(self.model.njnt)
            if self.model.joint(j).name.startswith(prefix)
        ]
        if len(joints) != 85 or any(
            self.model.jnt_type[j]
            not in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)
            for j in joints
        ):
            raise ValueError("Unexpected native scalar-joint contract")
        self.qpos_ids = self.model.jnt_qposadr[joints].copy()
        self.qvel_ids = self.model.jnt_dofadr[joints].copy()
        self.activation_ids = self.model.actuator_actadr[self.actuator_ids].copy()
        if np.any(self.activation_ids < 0) or np.any(
            self.model.actuator_actnum[self.actuator_ids] != 1
        ):
            raise ValueError("Native muscle activation state missing")
        self.body_ids = tuple(self.model.body(prefix + name).id for name in KEY_BODIES)

    def measure(self, data):
        if not np.isfinite(data.time):
            raise ValueError("Nonfinite shared clock")
        velocities = []
        for body in self.body_ids:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros_like(jacp)
            mujoco.mj_jacBody(self.model, data, jacp, jacr, body)
            velocities.append(jacp @ data.qvel)
        return BodyMeasurements(
            actor=self.actor,
            time_s=float(data.time),
            qpos=owned_readonly(data.qpos[self.qpos_ids]),
            qvel=owned_readonly(data.qvel[self.qvel_ids]),
            activation=owned_readonly(data.act[self.activation_ids]),
            body_names=KEY_BODIES,
            world_position_m=owned_readonly(data.xpos[list(self.body_ids)]),
            world_linear_velocity_mps=owned_readonly(velocities),
        )
