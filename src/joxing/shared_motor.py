"""Read-only first stage of the pretrained shared-world motor bridge.

This builds the native teacher's 3601 features, not a trained boxing controller.
It never steps, calls mj_forward, changes physics, or normalizes an observation.
Reference targets are supplied explicitly in the original native task frame.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

KEYPOINTS = (
    "sternum",
    "head_neck",
    "toes_r",
    "toes_l",
    "proximal_row_r",
    "proximal_row_l",
)


def frozen(value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("Malformed native motor feature/reference")
    array = array.copy()
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class MotorReferenceState:
    qpos: np.ndarray
    future_qpos: np.ndarray
    keypoints_relative_pelvis: np.ndarray

    def __post_init__(self):
        for key, shape in [
            ("qpos", (85,)),
            ("future_qpos", (5, 85)),
            ("keypoints_relative_pelvis", (6, 3)),
        ]:
            object.__setattr__(self, key, frozen(getattr(self, key), shape))


class NativeMotorMap:
    """Semantic native order; explicit, validated spawn yaw, never guessed IDs."""

    def __init__(self, shared, canonical, actor, *, spawn_yaw_rad):
        if actor not in ("A", "B") or not np.isfinite(spawn_yaw_rad):
            raise ValueError("Motor corner/yaw")
        if (canonical.nq, canonical.nv, canonical.nu, canonical.na) != (
            85,
            85,
            700,
            700,
        ):
            raise ValueError("Native teacher anatomy")
        prefix = actor + "_"
        joints = [
            shared.joint(prefix + canonical.joint(i).name).id
            for i in range(canonical.njnt)
        ]
        if len(joints) != 85 or any(
            shared.jnt_type[j]
            not in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)
            for j in joints
        ):
            raise ValueError("Native scalar joint map")
        self.qpos_ids = shared.jnt_qposadr[joints].copy()
        self.qvel_ids = shared.jnt_dofadr[joints].copy()
        self.actuator_ids = np.array(
            [
                shared.actuator(prefix + canonical.actuator(i).name).id
                for i in range(700)
            ]
        )
        self.activation_ids = shared.actuator_actadr[self.actuator_ids].copy()
        if (
            len(set(self.actuator_ids)) != 700
            or np.any(self.activation_ids < 0)
            or np.any(shared.actuator_actnum[self.actuator_ids] != 1)
        ):
            raise ValueError("Native muscle map")
        for field in (
            "actuator_dynprm",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_ctrlrange",
        ):
            if not np.array_equal(
                getattr(shared, field)[self.actuator_ids], getattr(canonical, field)
            ):
                raise ValueError("Native muscle parameters changed")
        self.keypoint_ids = [shared.body(prefix + name).id for name in KEYPOINTS]
        self.pelvis_id = shared.body(prefix + "pelvis").id
        yaw_quat = np.array(
            [np.cos(spawn_yaw_rad / 2), 0.0, 0.0, np.sin(spawn_yaw_rad / 2)]
        )
        expected = np.empty(4)
        mujoco.mju_mulQuat(expected, yaw_quat, canonical.body("pelvis").quat)
        actual = shared.body(prefix + "pelvis").quat
        if min(np.max(abs(actual - expected)), np.max(abs(actual + expected))) > 1e-12:
            raise ValueError("Spawn yaw does not match compiled native pelvis frame")
        c, s = np.cos(spawn_yaw_rad), np.sin(spawn_yaw_rad)
        self.native_to_world = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for field in (
            "qpos_ids",
            "qvel_ids",
            "actuator_ids",
            "activation_ids",
            "native_to_world",
        ):
            getattr(self, field).flags.writeable = False


def native_teacher_observation(data, mapping, reference):
    """Capture actual dynamics without recomputing the solver's cached fields.

    Caller must freeze both corners at one actual motor sample boundary. This
    does not supply temporal ownership, reference phase or replay integration;
    those remain the next bridge stage. Future self-targets are not opponent data.
    """
    keypoints = (
        data.xpos[mapping.keypoint_ids] - data.xpos[mapping.pelvis_id]
    ) @ mapping.native_to_world
    blocks = {
        "qpos": data.qpos[mapping.qpos_ids],
        "qvel": data.qvel[mapping.qvel_ids],
        "qacc": data.qacc[mapping.qvel_ids],
        "act": data.act[mapping.activation_ids],
        "actuator_forces": (data.actuator_force[mapping.actuator_ids] / 1000).clip(
            -100, 100
        ),
        "actuator_length": data.actuator_length[mapping.actuator_ids],
        "actuator_velocity": data.actuator_velocity[mapping.actuator_ids].clip(
            -100, 100
        ),
        "key_xpos": keypoints.ravel(),
        "qpos_ref": reference.qpos,
        "qpos_ref_future": reference.future_qpos.ravel(),
        "key_xpos_ref": reference.keypoints_relative_pelvis.ravel(),
    }
    blocks = {key: np.array(value, copy=True) for key, value in blocks.items()}
    values = np.concatenate(list(blocks.values())).astype(np.float32)
    if values.shape != (3601,) or not np.isfinite(values).all():
        raise ValueError("Native teacher observation shape/finite contract")
    values.flags.writeable = False
    for block in blocks.values():
        block.flags.writeable = False
    return values, blocks
