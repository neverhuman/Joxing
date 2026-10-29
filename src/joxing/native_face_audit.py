"""Scratch native FK and synthetic contact calibration; never integrates physics.

The mechanical face is the hemisphere pointing from the native hand origin
towards the added glove center. This is an equipment convention, not an
independently measured anatomical knuckle or force calibration.
"""

import hashlib
from dataclasses import asdict

import mujoco
import numpy as np

from .physics import Contact, point_velocity
from .referee import MatchReferee


def sha_array(a):
    return hashlib.sha256(np.asarray(a).tobytes(order="C")).hexdigest()


def fingerprint(data):
    return {
        k: sha_array(getattr(data, k))
        for k in ("qpos", "qvel", "act", "ctrl", "qfrc_applied", "xfrc_applied")
    } | {"time": float(data.time)}


def velocity_pair(model, data, epsilon):
    """Differentiate native FK at q±epsilon*v; these are scratch data only."""
    plus, minus = mujoco.MjData(model), mujoco.MjData(model)
    for target, sign in ((plus, 1), (minus, -1)):
        target.qpos[:] = data.qpos
        mujoco.mj_integratePos(model, target.qpos, data.qvel, sign * epsilon)
        mujoco.mj_kinematics(model, target)
    return plus, minus


def finite_point_velocity(data, plus, minus, body, point, epsilon):
    local = data.xmat[body].reshape(3, 3).T @ (point - data.xpos[body])
    a = plus.xpos[body] + plus.xmat[body].reshape(3, 3) @ local
    b = minus.xpos[body] + minus.xmat[body].reshape(3, 3) @ local
    return (a - b) / (2 * epsilon)


def synthetic_contact(glove, target, point, normal, vg, vt, reverse):
    approach = float(np.dot(vg - vt, normal))
    if reverse:
        return Contact(
            target, glove, point, -normal, -10.0 * normal, 10.0, approach, 0.0, vt, vg
        )
    return Contact(
        glove, target, point, normal, 10.0 * normal, 10.0, approach, 0.0, vg, vt
    )


def audit(
    scene,
    *,
    epsilon=1e-6,
    velocity_tolerance=2e-7,
    geometry_tolerance=1e-10,
    on_row=None,
    on_event=None,
):
    model = scene.model
    if (
        model.nu != 1400
        or model.na != 1400
        or any(len(scene.actuator_ids[a]) != 700 for a in "AB")
        or len(set(np.concatenate(list(scene.actuator_ids.values())).tolist())) != 1400
        or scene.report.get("motor_variant") is not None
        or any(
            error
            for row in scene.report["preserved_actuator_max_error"].values()
            for error in row.values()
        )
    ):
        raise ValueError("Intact default native embodiment required")
    if (
        not 0 < epsilon <= 1e-4
        or not 0 < velocity_tolerance <= 1e-5
        or not 0 < geometry_tolerance <= 1e-7
    ):
        raise ValueError("Native face calibration tolerances")
    referee = MatchReferee(scene, "synthetic-native-face-calibration")
    initial_referee = referee.snapshot()
    hand_joints = {}
    for name in sorted(referee.gloves):
        hand = model.body(name[:2] + "hand_" + name[-1]).id
        joints = np.flatnonzero(model.jnt_bodyid == hand).tolist()
        if not joints or any(
            model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints
        ):
            raise ValueError("Native wrist hinge mapping unavailable")
        hand_joints[name] = joints
    dof_actors = [model.body(int(b)).name[0] for b in model.dof_bodyid]
    rows, seen_orders, mirror_rows, states = [], {}, [], {}
    largest_velocity_error, largest_rotation_witness = 0.0, 0.0
    for pose in ("native-key", "scratch-wrist-offset"):
        qpos = np.array(scene.report["initial_qpos"], np.float64)
        if pose == "scratch-wrist-offset":
            for name, joints in hand_joints.items():
                for j in joints:
                    adr = int(model.jnt_qposadr[j])
                    value = qpos[adr] + (0.05 if name[-1] == "r" else -0.05)
                    qpos[adr] = (
                        np.clip(value, *model.jnt_range[j])
                        if model.jnt_limited[j]
                        else value
                    )
        reference = mujoco.MjData(model)
        reference.qpos[:] = qpos
        mujoco.mj_forward(model, reference)
        rotation = np.diag([-1.0, -1.0, 1.0])
        for hand in "lr":
            ga, gb = [model.geom(a + "_glove_" + hand).id for a in "AB"]
            expected = np.array([0.8, 0, 0]) + rotation @ (
                reference.geom_xpos[ga] - [-0.8, 0, 0]
            )
            if np.max(np.abs(reference.geom_xpos[gb] - expected)) > geometry_tolerance:
                raise ValueError("Native A/B spawn yaw position equivariance")
            ba, bb = [int(model.geom_bodyid[g]) for g in (ga, gb)]
            if (
                np.max(
                    np.abs(
                        reference.xmat[bb].reshape(3, 3)
                        - rotation @ reference.xmat[ba].reshape(3, 3)
                    )
                )
                > geometry_tolerance
            ):
                raise ValueError("Native A/B spawn yaw orientation equivariance")
            mirror_rows.append(
                {"pose": pose, "hand": hand, "A_B_yaw_equivariant": True}
            )
        for actor in "AB":
            for profile in ("rest", "mixed", "opponent-only"):
                data = mujoco.MjData(model)
                data.qpos[:] = qpos
                data.qvel[:] = (
                    0
                    if profile == "rest"
                    else 0.2 * np.sin(0.37 * (1 + np.arange(model.nv)))
                )
                if profile == "opponent-only":
                    data.qvel[:] *= np.array([a != actor for a in dof_actors])
                mujoco.mj_forward(model, data)
                before = fingerprint(data)
                state_key = f"{pose}/{actor}/{profile}"
                states[state_key] = {
                    "qpos": data.qpos.tolist(),
                    "qvel": data.qvel.tolist(),
                    "qpos_sha256": sha_array(data.qpos),
                    "qvel_sha256": sha_array(data.qvel),
                    "engine_unchanged_fingerprint": before,
                }
                plus, minus = velocity_pair(model, data, epsilon)
                for hand in "lr":
                    name = actor + "_glove_" + hand
                    glove = model.geom(name).id
                    body = int(model.geom_bodyid[glove])
                    parent = model.body(actor + "_hand_" + hand).id
                    if int(model.body_parentid[body]) != parent:
                        raise ValueError("Glove parent differs from native hand")
                    displacement = data.geom_xpos[glove] - data.xpos[parent]
                    if abs(np.linalg.norm(displacement) - 0.045) > geometry_tolerance:
                        raise ValueError("Native equipment distal offset")
                    axis = displacement / np.linalg.norm(displacement)
                    declared_axis = data.xmat[body].reshape(3, 3) @ [0, -1.0, 0]
                    if np.max(np.abs(axis - declared_axis)) > geometry_tolerance:
                        raise ValueError(
                            "Face convention differs from equipment displacement"
                        )
                    radius = float(model.geom_size[glove, 0])
                    if (
                        model.geom_type[glove] != mujoco.mjtGeom.mjGEOM_SPHERE
                        or abs(radius - 0.062) > geometry_tolerance
                    ):
                        raise ValueError("Native glove geometry differs")
                    trunk = model.body(actor + "_thoracic12").id
                    opponent = "B" if actor == "A" else "A"
                    for zone, shell in (
                        ("head", "shell_head"),
                        ("torso", "shell_chest"),
                    ):
                        target = model.geom(opponent + "_" + shell).id
                        target_body = int(model.geom_bodyid[target])
                        for front in (True, False):
                            case_key = (pose, profile, name, zone, front)
                            if on_event is not None:
                                on_event(
                                    "entered", "/".join(map(str, (*case_key, False)))
                                )
                            normal = axis if front else -axis
                            point = data.geom_xpos[glove] + radius * normal
                            fd = [
                                finite_point_velocity(
                                    data, plus, minus, b, point, epsilon
                                )
                                for b in (body, target_body, trunk)
                            ]
                            jac = [
                                point_velocity(model, data, b, point)
                                for b in (body, target_body, trunk)
                            ]
                            error = max(
                                float(np.max(np.abs(v - w)))
                                for v, w in zip(fd, jac, strict=True)
                            )
                            largest_velocity_error = max(largest_velocity_error, error)
                            if error > velocity_tolerance:
                                raise ValueError(
                                    "Native material-point Jacobian differs from independent FK derivative"
                                )
                            center_velocity = finite_point_velocity(
                                data, plus, minus, body, data.geom_xpos[glove], epsilon
                            )
                            rotational_witness = float(
                                np.linalg.norm(fd[0] - center_velocity)
                            )
                            largest_rotation_witness = max(
                                largest_rotation_witness, rotational_witness
                            )
                            expected_active = float(np.dot(fd[0] - fd[2], normal))
                            for reverse in (False, True):
                                if reverse and on_event is not None:
                                    on_event(
                                        "entered", "/".join(map(str, (*case_key, True)))
                                    )
                                contact = synthetic_contact(
                                    glove, target, point, normal, fd[0], fd[1], reverse
                                )
                                interval = referee.stage(
                                    data, [contact], data.qvel.copy()
                                )[name]
                                if len(interval.touches) != 1:
                                    raise ValueError(
                                        "Native synthetic sample ownership"
                                    )
                                touch = asdict(interval.touches[0])
                                if (
                                    touch["zone"] != zone
                                    or abs(touch["face_cos"] - (1 if front else -1))
                                    > geometry_tolerance
                                    or touch["striking_face"] is not front
                                    or abs(touch["active_speed_mps"] - expected_active)
                                    > velocity_tolerance
                                    or abs(
                                        touch["approach_mps"]
                                        - np.dot(fd[0] - fd[1], normal)
                                    )
                                    > velocity_tolerance
                                    or touch["normal_force_N"] != 10.0
                                    or (
                                        profile == "opponent-only"
                                        and abs(touch["active_speed_mps"])
                                        > velocity_tolerance
                                    )
                                ):
                                    raise ValueError(
                                        "Native staged synthetic face/velocity predicate differs"
                                    )
                                key = (pose, profile, name, zone, front)
                                values = {
                                    k: v for k, v in touch.items() if k != "sample_id"
                                }
                                if reverse and values != seen_orders[key]:
                                    raise ValueError(
                                        "Native synthetic geom-order sign differs"
                                    )
                                seen_orders[key] = values
                                row = {
                                    "id": "/".join(map(str, (*key, reverse))),
                                    "pose": pose,
                                    "profile": profile,
                                    "glove": name,
                                    "zone": zone,
                                    "front": front,
                                    "reverse_geom_order": reverse,
                                    "point_world": point.tolist(),
                                    "glove_position_world": data.geom_xpos[
                                        glove
                                    ].tolist(),
                                    "hand_position_world": data.xpos[parent].tolist(),
                                    "glove_radius_m": radius,
                                    "normal_from_glove_world": normal.tolist(),
                                    "equipment_axis_world": axis.tolist(),
                                    "velocity_glove_fd": fd[0].tolist(),
                                    "velocity_target_fd": fd[1].tolist(),
                                    "velocity_trunk_fd": fd[2].tolist(),
                                    "staged": touch,
                                    "maximum_velocity_error_mps": error,
                                    "point_rotation_witness_mps": rotational_witness,
                                    "qpos_sha256": sha_array(data.qpos),
                                    "qvel_sha256": sha_array(data.qvel),
                                    "state_key": state_key,
                                    "classification": "synthetic contact on scratch native FK; no collision/force measurement",
                                    "status": "passed",
                                }
                                if on_row is not None:
                                    on_row(row)
                                rows.append(row)
                                if on_event is not None:
                                    on_event("completed", row["id"])
                if before != fingerprint(data):
                    raise ValueError(
                        "Read-only stage changed native scratch engine state"
                    )
        del reference
    if referee.snapshot() != initial_referee or largest_rotation_witness <= 1e-4:
        raise ValueError("Referee unchanged or nonzero rotation witness missing")
    if len(rows) != 192:
        raise ValueError("Native synthetic calibration denominator")
    buffer = np.empty(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return {
        "schema": "joxing-native-face-kinematic-audit-v1",
        "status": "passed",
        "rows": rows,
        "kinematic_states": states,
        "row_count": len(rows),
        "native_yaw_checks": mirror_rows,
        "model_binary_sha256": sha_array(buffer),
        "model_binary_bytes": len(buffer),
        "maximum_velocity_error_mps": largest_velocity_error,
        "maximum_point_rotation_witness_mps": largest_rotation_witness,
        "epsilon_s": epsilon,
        "velocity_tolerance_mps": velocity_tolerance,
        "geometry_tolerance": geometry_tolerance,
        "new_physics": 0,
        "new_optimizer_updates": 0,
        "referee_commits": 0,
        "native_boxing_proof_passed": False,
        "limitations": [
            "Synthetic contacts do not lie on actual opponent surfaces and impose no measured force",
            "Distal hand-to-glove hemisphere is a mechanical convention, not anatomical knuckle validation",
            "No missing old signed contacts, physical punch, motor stability or sporting outcome is recovered",
        ],
    }
