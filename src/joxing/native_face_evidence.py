"""Independent saved calibration read-audit; no production referee/velocity call."""

import hashlib
from itertools import product

import mujoco
import numpy as np


def bits(a):
    return hashlib.sha256(np.asarray(a).tobytes(order="C")).hexdigest()


def audit_saved(
    scene, report, *, epsilon=1e-6, velocity_tolerance=2e-7, geometry_tolerance=1e-10
):
    if (
        report["schema"] != "joxing-native-face-kinematic-audit-v1"
        or report["status"] != "passed"
        or report["row_count"] != 192
        or report["epsilon_s"] != epsilon
        or report["velocity_tolerance_mps"] != velocity_tolerance
        or report["geometry_tolerance"] != geometry_tolerance
        or report["new_physics"]
        or report["referee_commits"]
        or report["new_optimizer_updates"]
        or report["native_boxing_proof_passed"]
    ):
        raise ValueError("Saved native face scope/parameters")
    model = scene.model
    buffer = np.empty(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    if (
        bits(buffer) != report["model_binary_sha256"]
        or len(buffer) != report["model_binary_bytes"]
    ):
        raise ValueError("Saved native face compiled model identity")
    if model.nu != 1400 or model.na != 1400:
        raise ValueError("Saved native face all muscles")
    states = {}
    for pose, actor, profile in product(
        ("native-key", "scratch-wrist-offset"), "AB", ("rest", "mixed", "opponent-only")
    ):
        key = f"{pose}/{actor}/{profile}"
        saved = report["kinematic_states"][key]
        qpos = np.array(scene.report["initial_qpos"], np.float64)
        if pose == "scratch-wrist-offset":
            for a, hand in product("AB", "lr"):
                body = model.body(a + "_hand_" + hand).id
                for j in np.flatnonzero(model.jnt_bodyid == body):
                    adr = int(model.jnt_qposadr[j])
                    delta = 0.05 if hand == "r" else -0.05
                    value = qpos[adr] + delta
                    if model.jnt_limited[j]:
                        value = min(
                            max(value, model.jnt_range[j, 0]), model.jnt_range[j, 1]
                        )
                    qpos[adr] = value
        qvel = (
            np.zeros(model.nv)
            if profile == "rest"
            else 0.2 * np.sin(0.37 * (1 + np.arange(model.nv)))
        )
        if profile == "opponent-only":
            qvel *= [
                not model.body(int(b)).name.startswith(actor + "_")
                for b in model.dof_bodyid
            ]
        if (
            bits(qpos) != saved["qpos_sha256"]
            or bits(qvel) != saved["qvel_sha256"]
            or bits(np.array(saved["qpos"], np.float64)) != bits(qpos)
            or bits(np.array(saved["qvel"], np.float64)) != bits(qvel)
        ):
            raise ValueError("Saved native face prescribed pose/velocity")
        data, plus, minus = [mujoco.MjData(model) for _ in range(3)]
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)
        fingerprint = {
            k: bits(getattr(data, k))
            for k in ("qpos", "qvel", "act", "ctrl", "qfrc_applied", "xfrc_applied")
        } | {"time": float(data.time)}
        if fingerprint != saved["engine_unchanged_fingerprint"]:
            raise ValueError("Saved native face unassisted scratch state")
        for scratch, dt in ((plus, epsilon), (minus, -epsilon)):
            scratch.qpos[:] = qpos
            mujoco.mj_integratePos(model, scratch.qpos, qvel, dt)
            mujoco.mj_kinematics(model, scratch)
        states[key] = data, plus, minus
    if len(report["kinematic_states"]) != len(states):
        raise ValueError("Saved native face state denominator")
    expected_keys = set(
        product(
            ("native-key", "scratch-wrist-offset"),
            ("rest", "mixed", "opponent-only"),
            ("A_glove_l", "A_glove_r", "B_glove_l", "B_glove_r"),
            ("head", "torso"),
            (True, False),
            (False, True),
        )
    )
    seen, orders = set(), {}
    maximum_error, rotation_witness = 0.0, 0.0
    for row in report["rows"]:
        key = tuple(
            row[k]
            for k in ("pose", "profile", "glove", "zone", "front", "reverse_geom_order")
        )
        if (
            key not in expected_keys
            or key in seen
            or type(row["front"]) is not bool
            or type(row["reverse_geom_order"]) is not bool
        ):
            raise ValueError("Saved native face attempted-row denominator")
        seen.add(key)
        if (
            row["id"] != "/".join(map(str, key))
            or row["classification"]
            != "synthetic contact on scratch native FK; no collision/force measurement"
        ):
            raise ValueError("Saved native face case identity/classification")
        state_key = "/".join((row["pose"], row["glove"][0], row["profile"]))
        if row["state_key"] != state_key:
            raise ValueError("Saved native face state ownership")
        data, plus, minus = states[state_key]
        if row["qpos_sha256"] != bits(data.qpos) or row["qvel_sha256"] != bits(
            data.qvel
        ):
            raise ValueError("Saved native face case state identity")
        glove = model.geom(row["glove"]).id
        gb = int(model.geom_bodyid[glove])
        hand = model.body(row["glove"][:2] + "hand_" + row["glove"][-1]).id
        axis = data.geom_xpos[glove] - data.xpos[hand]
        if abs(np.linalg.norm(axis) - 0.045) > geometry_tolerance:
            raise ValueError("Saved native face equipment offset")
        axis /= np.linalg.norm(axis)
        normal = axis if row["front"] else -axis
        point = data.geom_xpos[glove] + 0.062 * normal
        expected_vectors = {
            "glove_position_world": data.geom_xpos[glove],
            "hand_position_world": data.xpos[hand],
            "equipment_axis_world": axis,
            "normal_from_glove_world": normal,
            "point_world": point,
        }
        for name, vector in expected_vectors.items():
            actual = np.array(row[name], np.float64)
            if (
                actual.shape != (3,)
                or not np.isfinite(actual).all()
                or np.max(np.abs(actual - vector)) > geometry_tolerance
            ):
                raise ValueError("Saved native face geometry: " + name)
        if row["glove_radius_m"] != 0.062:
            raise ValueError("Saved native face radius")
        opponent = "B" if row["glove"][0] == "A" else "A"
        target = model.geom(
            opponent + ("_shell_head" if row["zone"] == "head" else "_shell_chest")
        ).id
        trunk = model.body(row["glove"][0] + "_thoracic12").id

        def derivative(body, sample_point, data=data, plus=plus, minus=minus):
            local = data.xmat[body].reshape(3, 3).T @ (sample_point - data.xpos[body])
            pp = plus.xpos[body] + plus.xmat[body].reshape(3, 3) @ local
            pm = minus.xpos[body] + minus.xmat[body].reshape(3, 3) @ local
            return (pp - pm) / (2 * epsilon)

        velocities, errors = [], []
        for field, body in zip(
            ("velocity_glove_fd", "velocity_target_fd", "velocity_trunk_fd"),
            (gb, int(model.geom_bodyid[target]), trunk),
            strict=True,
        ):
            fd = derivative(body, point)
            saved_velocity = np.array(row[field], np.float64)
            if saved_velocity.shape != (3,) or not np.isfinite(saved_velocity).all():
                raise ValueError("Saved native face nonfinite velocity")
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, point, body)
            error = float(np.max(np.abs(fd - jacp @ data.qvel)))
            if (
                np.max(np.abs(fd - row[field])) > velocity_tolerance
                or error > velocity_tolerance
            ):
                raise ValueError(
                    "Saved native face independent material-point velocity"
                )
            velocities.append(fd)
            errors.append(error)
        maximum_error = max(maximum_error, *errors)
        witness = float(
            np.linalg.norm(velocities[0] - derivative(gb, data.geom_xpos[glove]))
        )
        rotation_witness = max(rotation_witness, witness)
        touch = row["staged"]
        if not np.isfinite(
            [
                touch[k]
                for k in (
                    "normal_force_N",
                    "approach_mps",
                    "active_speed_mps",
                    "face_cos",
                )
            ]
            + [row["maximum_velocity_error_mps"], row["point_rotation_witness_mps"]]
        ).all():
            raise ValueError("Saved native face nonfinite staged value")
        expected_active = float(np.dot(velocities[0] - velocities[2], normal))
        expected_approach = float(np.dot(velocities[0] - velocities[1], normal))
        if (
            touch["zone"] != row["zone"]
            or touch["normal_force_N"] != 10.0
            or touch["striking_face"] is not row["front"]
            or abs(touch["face_cos"] - np.dot(axis, normal)) > geometry_tolerance
            or abs(touch["active_speed_mps"] - expected_active) > velocity_tolerance
            or abs(touch["approach_mps"] - expected_approach) > velocity_tolerance
            or abs(row["maximum_velocity_error_mps"] - max(errors)) > velocity_tolerance
            or abs(row["point_rotation_witness_mps"] - witness) > velocity_tolerance
            or row["status"] != "passed"
        ):
            raise ValueError("Saved native face synthetic staged predicate")
        semantic = {k: v for k, v in touch.items() if k != "sample_id"}
        if key[:-1] in orders and orders[key[:-1]] != semantic:
            raise ValueError("Saved native face contact-order invariance")
        orders[key[:-1]] = semantic
    if (
        seen != expected_keys
        or maximum_error > velocity_tolerance
        or rotation_witness <= 1e-4
    ):
        raise ValueError("Saved native face complete calibration gate")
    if (
        abs(report["maximum_velocity_error_mps"] - maximum_error) > velocity_tolerance
        or abs(report["maximum_point_rotation_witness_mps"] - rotation_witness)
        > velocity_tolerance
    ):
        raise ValueError("Saved native face aggregate metrics")
    return {
        "schema": "joxing-native-face-independent-audit-v1",
        "status": "passed",
        "rows": len(seen),
        "prescribed_states": len(states),
        "native_model_binary_sha256": bits(buffer),
        "maximum_velocity_error_mps": maximum_error,
        "maximum_point_rotation_witness_mps": rotation_witness,
        "production_referee_or_point_velocity_called": False,
        "new_physics": 0,
        "native_boxing_proof_passed": False,
    }
