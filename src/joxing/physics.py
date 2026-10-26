"""CPU reference contact staging and an independent momentum-consistency fixture."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Contact:
    geom1: int
    geom2: int
    point: np.ndarray
    normal: np.ndarray
    force_on_geom2: np.ndarray
    normal_force_N: float
    approach_mps: float
    penetration_m: float


def point_velocity(model, data, body_id, point):
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
    return jacp @ data.qvel


def step_contacts(model, data) -> list[Contact]:
    """Controls already held. Save geometry/velocities before solve, wrenches after."""
    if model.opt.integrator not in (
        mujoco.mjtIntegrator.mjINT_EULER,
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    ):
        raise ValueError("Unsupported contact staging integrator")
    time_before = data.time
    warning_before = np.array([w.number for w in data.warning])
    mujoco.mj_step1(model, data)
    pre = []
    for c in data.contact:
        point, normal = c.pos.copy(), c.frame[:3].copy()
        v1 = point_velocity(model, data, model.geom_bodyid[c.geom1], point)
        v2 = point_velocity(model, data, model.geom_bodyid[c.geom2], point)
        pre.append(
            (
                int(c.geom1),
                int(c.geom2),
                point,
                normal,
                float(np.dot(v1 - v2, normal)),
                float(c.dist),
            )
        )
    mujoco.mj_step2(model, data)
    if (
        not np.isfinite(data.qpos).all()
        or not np.isfinite(data.qvel).all()
        or abs(data.time - time_before - model.opt.timestep) > 1e-9
        or np.any(np.array([w.number for w in data.warning]) > warning_before)
    ):
        raise RuntimeError("Invalid physics substep: quarantine episode")
    if data.ncon != len(pre):
        raise RuntimeError("Contact staging identity changed")
    result = []
    for i, (g1, g2, p, n, v, dist) in enumerate(pre):
        c = data.contact[i]
        if (c.geom1, c.geom2) != (g1, g2):
            raise RuntimeError("Contact staging order changed")
        if c.efc_address < 0:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        if not np.isfinite(wrench).all():
            raise RuntimeError("Nonfinite contact wrench")
        result.append(
            Contact(
                g1,
                g2,
                p,
                n,
                c.frame.reshape(3, 3).T @ wrench[:3],
                max(0.0, float(wrench[0])),
                v,
                max(0.0, -dist),
            )
        )
    return result


def momentum_fixture(dt=0.0005, angle=0.0, reverse=False) -> dict:
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    start = -0.25 * direction
    moving = f'<body name="ball" pos="{start[0]} {start[1]} 0"><freejoint/><geom name="ball" type="sphere" size=".1" mass="1"/></body>'
    fixed = '<geom name="target" type="sphere" size=".1"/>'
    xml = f'''<mujoco><option timestep="{dt}" gravity="0 0 0" integrator="implicitfast" iterations="100" tolerance="1e-12"/>
        <default><geom friction="0 0 0" condim="1" solref=".01 1"/></default>
        <worldbody>{moving + fixed if reverse else fixed + moving}</worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[:3] = 2 * direction
    initial = data.qvel[:3].copy()
    impulse = np.zeros(3)
    gid = model.geom("ball").id
    peak, count, approaches = 0.0, 0, []
    trace = []
    for tick in range(round(0.2 / dt)):
        contacts = step_contacts(model, data)
        force = np.zeros(3)
        for c in contacts:
            force += c.force_on_geom2 if c.geom2 == gid else -c.force_on_geom2
            if c.normal_force_N > 0:
                approaches.append(c.approach_mps)
                count += 1
        impulse += force * dt
        peak = max(peak, float(np.linalg.norm(force)))
        trace.append([tick * dt, float(np.linalg.norm(force)), *impulse.tolist()])
    delta = data.qvel[:3] - initial
    return {
        "dt_s": dt,
        "angle_rad": angle,
        "reverse": reverse,
        "impulse_Ns": impulse.tolist(),
        "momentum_change_kg_mps": delta.tolist(),
        "error": float(np.max(np.abs(impulse - delta))),
        "peak_model_force_N": peak,
        "force_samples": count,
        "first_approach_mps": approaches[0],
        "trace": trace,
        "provenance": "simulation_consistency_fixture_not_glove_calibration",
    }


def scoring_fixture(zone="head") -> dict:
    """A real force-producing glove/target rig drives the candidate event referee.

    Target is fixed and its entire surface is one known fixture zone. Anatomical
    region fitting and multi-surface boxing are separate qualification work.
    """
    from .rules import HitTracker, Touch

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep=".001" gravity="0 0 0" integrator="implicitfast"/>
      <default><geom condim="1" friction="0 0 0" solref=".015 1"/></default>
      <worldbody><geom name="target" type="sphere" size=".1"/>
      <body name="glove" pos="-.3 0 0"><freejoint/><geom name="glove" type="sphere" size=".062" mass="1"/></body></worldbody>
      </mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[0] = 2.0
    tracker = HitTracker("physical-fixture", "A/right")
    glove = model.geom("glove").id
    target = model.geom("target").id
    for k in range(250):
        contacts = step_contacts(model, data)
        touches = []
        for i, c in enumerate(contacts):
            if {c.geom1, c.geom2} != {glove, target}:
                continue
            n = c.normal if c.geom1 == glove else -c.normal
            face = float(np.dot(n, [1.0, 0, 0]))
            membership = bool(np.dot(c.point - data.geom_xpos[glove], [1.0, 0, 0]) > 0)
            touches.append(
                Touch(
                    str(i),
                    zone,
                    c.normal_force_N,
                    c.approach_mps,
                    c.approach_mps,
                    face,
                    membership,
                )
            )
        clearance = float(
            np.linalg.norm(data.geom_xpos[glove] - data.geom_xpos[target]) - 0.162
        )
        extension = float(data.geom_xpos[glove, 0] + 0.4)
        tracker.update(
            k, touches, clearance_m=clearance, extension_m=extension, bell=k == 249
        )
    return {
        "zone": zone,
        "events": list(tracker.ledger.values()),
        "model": "fixed-target-physical-fixture",
        "rules_hash": tracker.rules.hash,
    }
