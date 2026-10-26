"""Compose two intact anatomies. Shell dimensions are provisional design estimates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .provenance import ROOT, digest, manifest, write_json

SOURCE = ROOT / "third_party/ms-human-700/MS-Human-700.xml"


@dataclass
class Scene:
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    report: dict
    actuator_ids: dict[str, np.ndarray]


def build_scene(dt: float = .001, shells: bool = True) -> Scene:
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    source = mujoco.MjModel.from_xml_path(str(SOURCE))
    base = mujoco.MjData(source)
    mujoco.mj_resetDataKeyframe(source, base, 0)
    mujoco.mj_forward(source, base)
    world = mujoco.MjSpec.from_string('''<mujoco model="joxing-dual-smoke">
      <compiler angle="radian"/><option integrator="implicitfast" solver="Newton" iterations="100" tolerance="1e-10"/>
      <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048" offsamples="4"/></visual>
      <worldbody><geom name="floor" type="plane" size="4 4 .1" contype="1" conaffinity="6" friction=".9 .005 .0001" rgba=".12 .16 .22 1"/>
      <light pos="0 0 4" dir="0 0 -1"/></worldbody></mujoco>''')
    world.option.timestep = dt
    children = []
    shell_roles = {}
    for actor, x, yaw, color in (("A", -.8, 0., [1., .18, .12, 1.]), ("B", .8, np.pi, [.15, .4, 1., 1.])):
        child = mujoco.MjSpec.from_file(str(SOURCE))
        children.append(child)
        bit, affinity = (2, 5) if actor == "A" else (4, 3)
        # Preserve existing colliders, disabled anatomy, routing and native inertia.
        for geom in child.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype, geom.conaffinity = bit, affinity
        if shells:
            def shell(name, parent, role, size, pos, kind=mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                child.body(parent).add_geom(name=name, type=kind, pos=pos, size=size,
                    mass=0, contype=bit, conaffinity=affinity, group=2,
                    rgba=color[:3] + [.35], solref=[.015, 1.], friction=[.5, .005, .0001])
                shell_roles[actor + "_" + name] = role
            # Source local X = forward, local Y = up. No phantom jaw/skull bodies.
            shell("shell_head", "head_neck", "head", [.10, .125, .09], [0, .12, 0])
            shell("shell_chest", "thoracic12", "torso", [.13, .18, .17], [.025, .13, 0])
            shell("shell_abdomen", "lumbar5", "torso", [.12, .105, .145], [.015, .055, 0])
            for hand in ("r", "l"):
                parent = child.body("hand_" + hand)
                # Explicit child inertia adds equipment mass exactly once.
                glove = parent.add_body(name="glove_" + hand, pos=[0, -.045, 0])
                glove.mass = .283
                glove.inertia = np.full(3, .4 * .283 * .062**2)
                glove.add_geom(name="glove_" + hand, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[.062, 0, 0], mass=0, contype=bit, conaffinity=affinity,
                    group=2, rgba=color, solref=[.015, 1.])
                shell_roles[actor + "_glove_" + hand] = "glove"
                for body, radius, length in (("humerus_", .045, .24), ("ulna_", .04, .21)):
                    name = "shell_" + body + hand
                    child.body(body + hand).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        fromto=[0, 0, 0, 0, -length, 0], size=[radius, 0, 0], mass=0,
                        contype=bit, conaffinity=affinity, group=2, rgba=color[:3]+[.35])
                    shell_roles[actor + "_" + name] = "guard"
        frame = world.worldbody.add_frame(name=actor+"_spawn", pos=[x, 0, 0], quat=[np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        frame.attach_body(child.body("pelvis"), prefix=actor+"_")
    model = world.compile()
    ids = {a: np.array([model.actuator(a+"_"+source.actuator(i).name).id for i in range(source.nu)]) for a in ("A", "B")}
    preserved = {}
    for field in ("actuator_dynprm", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange"):
        preserved[field] = {a: float(np.max(np.abs(getattr(model, field)[ix] - getattr(source, field)))) for a, ix in ids.items()}
    data = mujoco.MjData(model)
    # Restore source key pose by semantic coordinate; attachment removes child keys.
    for actor in ids:
        for j in range(source.njnt):
            data.qpos[model.joint(actor+"_"+source.joint(j).name).qposadr[0]] = base.qpos[source.jnt_qposadr[j]]
    mujoco.mj_forward(model, data)
    geoms = []
    for i in range(model.ngeom):
        name = model.geom(i).name
        if model.geom_contype[i] or model.geom_conaffinity[i]:
            geoms.append({"id": i, "name": name, "actor": name[0] if name.startswith(("A_", "B_")) else None,
                "role": shell_roles.get(name, "floor" if name == "floor" else "native_non_scoring"),
                "body": model.body(model.geom_bodyid[i]).name,
                "contype": int(model.geom_contype[i]), "conaffinity": int(model.geom_conaffinity[i])})
    report = manifest() | {"profile": "composition-smoke-not-competition", "dt_s": dt,
        "scene_hash": digest(world.to_xml()), "dimensions": {k: int(getattr(model,k)) for k in ("nq","nv","nu","na","nbody","ntendon","neq","ngeom")},
        "mass_kg": float(model.body_mass.sum()), "expected_mass_kg": float(2*source.body_mass.sum() + (4*.283 if shells else 0)),
        "preserved_actuator_max_error": preserved, "actuator_ids": {a: ix.tolist() for a,ix in ids.items()},
        "initial_qpos": data.qpos.tolist(), "physical_geoms": geoms,
        "gates_pending": ["shell fit and full surface coverage", "explicit self-contact pairs", "native collider overlap review", "stable guard reset bank", "free-root singularity study", "calibration", "foul/fall/exit adjudication"]}
    return Scene(world, model, report, ids)


def save_scene(scene: Scene, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(scene.spec.to_xml())
    write_json(out / "scene.json", scene.report)
