"""Native-root, ordinary-gravity whole-body diagnostic with audited reset preparation."""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .physics import step_contacts
from .provenance import digest
from .scene import SOURCE
from .support import floor_loads


@dataclass(frozen=True)
class BalanceConfig:
    dt_s: float = 0.001
    control_s: float = 0.01
    episode_s: float = 20.0
    preparation_s: float = 0.20
    max_reset_penetration_m: float = 0.002

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Expected positive finite balance configuration")
        for period in (self.control_s, self.episode_s, self.preparation_s):
            if abs(period / self.dt_s - round(period / self.dt_s)) > 1e-8:
                raise ValueError("Timing must use integer physics ticks")
        if self.preparation_s < 0.1 or self.episode_s < self.control_s:
            raise ValueError("Invalid preparation/horizon")


def project_equalities(model, data):
    """Reset-only Newton projection; leave root coordinates untouched."""
    for _ in range(8):
        mujoco.mj_forward(model, data)
        residual = data.efc_pos[: data.ne].copy()
        if np.max(abs(residual), initial=0) < 1e-10:
            break
        jac = data.efc_J.reshape(-1, model.nv)[: data.ne].copy()
        jac[:, :6] = 0
        delta = -np.linalg.lstsq(jac, residual, rcond=1e-10)[0]
        mujoco.mj_integratePos(model, data.qpos, delta, 1.0)
    mujoco.mj_forward(model, data)
    if np.max(abs(data.efc_pos[: data.ne]), initial=0) > 1e-8:
        raise RuntimeError("Reset equality projection did not converge")


class BalanceModel:
    def __init__(self, config=None):
        config = config or BalanceConfig()
        self.config = config
        spec = mujoco.MjSpec.from_file(str(SOURCE))
        spec.option.timestep = config.dt_s
        spec.option.gravity = [0, 0, -9.81]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        spec.visual.global_.offwidth = 960
        spec.visual.global_.offheight = 540
        spec.visual.quality.shadowsize = 2048
        spec.visual.quality.offsamples = 4
        for side, color in (("r", [1, 0.15, 0.1, 1]), ("l", [0.1, 0.3, 1, 1])):
            body = spec.body("hand_" + side).add_body(
                name="balance_glove_" + side, pos=[0, -0.045, 0]
            )
            body.mass = 0.283
            body.inertia = np.full(3, 0.4 * 0.283 * 0.062**2)
            body.add_geom(
                name="balance_glove_" + side,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.062, 0, 0],
                mass=0,
                contype=1,
                conaffinity=1,
                rgba=color,
            )
        self.model = spec.compile()
        self.hash = digest(
            {
                "xml": spec.to_xml(),
                "config": asdict(config),
                "variant": "native-root-gravity-v0.0.4",
            }
        )
        self.floor = self.model.geom("floor").id
        self.pelvis = self.model.body("pelvis").id
        self.weight_N = float(sum(self.model.body_mass) * 9.81)
        self.roles = {}
        for i in range(self.model.ngeom):
            if self.model.geom_contype[i] or self.model.geom_conaffinity[i]:
                body = self.model.body(self.model.geom_bodyid[i]).name
                self.roles[i] = (
                    "right"
                    if body in ("calcn_r", "toes_r")
                    else "left"
                    if body in ("calcn_l", "toes_l")
                    else "nonfoot"
                )

    def prepare(self, scenario_id=0):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        mujoco.mj_forward(self.model, data)
        source_residual = float(np.max(abs(data.efc_pos[: data.ne]), initial=0))
        rng = np.random.default_rng(scenario_id)
        # Small local starts, purpose-separated from policy/random-action RNG.
        data.qpos[self.model.joint("pelvis_tx").qposadr[0]] += rng.uniform(
            -0.005, 0.005
        )
        data.qpos[self.model.joint("pelvis_tilt").qposadr[0]] += rng.uniform(
            -0.01, 0.01
        )
        data.qpos[self.model.joint("pelvis_list").qposadr[0]] += rng.uniform(
            -0.01, 0.01
        )
        project_equalities(self.model, data)
        data.ctrl[:] = 0.02
        data.act[:] = 0.02
        loads = []
        for _ in range(round(self.config.preparation_s / self.config.dt_s)):
            contacts = step_contacts(self.model, data)
            loads.append(floor_loads(contacts, self.floor, self.roles))
        # Explicit reset-only projection after compliant loaded constraints settle.
        loaded_residual = float(np.max(abs(data.efc_pos[: data.ne]), initial=0))
        before = data.qpos.copy()
        project_equalities(self.model, data)
        jac = data.efc_J.reshape(-1, self.model.nv)[: data.ne].copy()
        correction = np.linalg.lstsq(jac, jac @ data.qvel, rcond=1e-10)[0]
        data.qvel -= correction
        mujoco.mj_forward(self.model, data)
        penetration = max((max(0.0, -float(c.dist)) for c in data.contact), default=0.0)
        history = loads[-round(0.1 / self.config.dt_s) :]
        bilateral = all(row["right"] > 1 and row["left"] > 1 for row in history)
        valid = bilateral and penetration <= self.config.max_reset_penetration_m
        audit = {
            "scenario_id": int(scenario_id),
            "valid": valid,
            "source_equality_error": source_residual,
            "loaded_equality_error": loaded_residual,
            "projected_equality_error": float(
                np.max(abs(data.efc_pos[: data.ne]), initial=0)
            ),
            "max_penetration_m": penetration,
            "bilateral_100ms": bilateral,
            "reset_qpos_projection_norm": float(np.linalg.norm(data.qpos - before)),
            "reset_qvel_projection_norm": float(np.linalg.norm(correction)),
            "preparation": "200ms gravity, excitation .02, reset-only equality and velocity projection",
            "assistance_during_evaluation": [],
        }
        if not valid:
            raise ValueError(f"Invalid reset: {audit}")
        data.time = 0.0
        data.xfrc_applied[:] = 0
        data.qfrc_applied[:] = 0
        return data, audit

    def audit(self):
        source = mujoco.MjModel.from_xml_path(str(SOURCE))
        m = self.model
        fields = (
            "actuator_dynprm",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_ctrlrange",
        )
        ids = [m.actuator(source.actuator(i).name).id for i in range(source.nu)]
        bodies = [m.body(source.body(i).name).id for i in range(source.nbody)]
        return {
            "model_hash": self.hash,
            "variant": "native-root-gravity-v0.0.4",
            "config": asdict(self.config),
            "nq": m.nq,
            "nv": m.nv,
            "nu": m.nu,
            "na": m.na,
            "neq": m.neq,
            "mass_kg": float(sum(m.body_mass)),
            "glove_mass_increment_kg": float(sum(m.body_mass) - sum(source.body_mass)),
            "preserved_muscle_parameters": all(
                np.array_equal(getattr(m, k)[ids], getattr(source, k)) for k in fields
            ),
            "preserved_segment_mass_inertia": bool(
                np.array_equal(m.body_mass[bodies], source.body_mass)
                and np.array_equal(m.body_inertia[bodies], source.body_inertia)
            ),
            "gravity": m.opt.gravity.tolist(),
            "root_joint_names": [m.joint(i).name for i in range(6)],
            "physical_surface_roles": {
                m.geom(i).name: r for i, r in self.roles.items()
            },
            "limitations": [
                "native colliders and exclusions; incomplete boxing envelope",
                "reset-only equality projection before and after gravity preparation; not a stand-up skill",
                "no motor competence established by model audit",
            ],
        }
