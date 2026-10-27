"""Hash-bound native compiled-scene input. No XML rebuild or model surgery."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .bodies import NativeBodyAdapter
from .scene import SOURCE

DIMENSIONS = {
    "nq": 170,
    "nv": 170,
    "nu": 1400,
    "na": 1400,
    "nbody": 165,
    "ntendon": 1400,
    "neq": 84,
    "ngeom": 715,
}


def file_sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@dataclass(frozen=True)
class FrozenScene:
    model: mujoco.MjModel
    report: dict
    actuator_ids: dict

    @property
    def spec(self):
        raise RuntimeError("Frozen binary scene cannot export/rebuild an XML spec")


def load_frozen_scene(binary, report_path, *, binary_sha256, report_sha256, scene_hash):
    binary, report_path = Path(binary), Path(report_path)
    if (
        not 0 < binary.stat().st_size <= 256 * 1024**2
        or not 0 < report_path.stat().st_size <= 1024**2
        or file_sha(binary) != binary_sha256
        or file_sha(report_path) != report_sha256
    ):
        raise ValueError("Frozen model/report bytes differ")
    report = json.loads(report_path.read_text())
    if report["scene_hash"] != scene_hash or report["dimensions"] != DIMENSIONS:
        raise ValueError("Frozen scene identity/dimensions differ")
    model = mujoco.MjModel.from_binary_path(str(binary))
    if (
        any(int(getattr(model, k)) != v for k, v in DIMENSIONS.items())
        or model.opt.timestep != 0.001
        or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        or report["dt_s"] != 0.001
        or report["profile"] != "composition-smoke-not-competition"
        or abs(float(model.body_mass.sum()) - report["mass_kg"]) > 1e-10
    ):
        raise ValueError(
            "Frozen compiled model differs from native diagnostic contract"
        )
    native = mujoco.MjModel.from_xml_path(str(SOURCE))
    if native.nu != 700 or native.na != 700:
        raise ValueError("Pinned anatomy lost native muscles")
    ids = {
        a: np.array(
            [model.actuator(a + "_" + native.actuator(i).name).id for i in range(700)],
            np.int64,
        )
        for a in ("A", "B")
    }
    for a, ix in ids.items():
        if ix.tolist() != report["actuator_ids"][a]:
            raise ValueError("Frozen semantic muscle order differs")
        for field in (
            "actuator_dynprm",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_ctrlrange",
        ):
            if not np.array_equal(getattr(model, field)[ix], getattr(native, field)):
                raise ValueError("Frozen native muscle parameters changed: " + field)
    registry = report["physical_geoms"]
    active = {
        i
        for i in range(model.ngeom)
        if model.geom_contype[i] or model.geom_conaffinity[i]
    }
    if (
        len({r["id"] for r in registry}) != len(registry)
        or {r["id"] for r in registry} != active
    ):
        raise ValueError("Frozen physical registry incomplete/duplicated")
    for r in registry:
        i = r["id"]
        name = model.geom(i).name
        if (
            r["name"] != name
            or r["body"] != model.body(model.geom_bodyid[i]).name
            or r["contype"] != int(model.geom_contype[i])
            or r["conaffinity"] != int(model.geom_conaffinity[i])
            or r["actor"] != (name[0] if name.startswith(("A_", "B_")) else None)
            or r["role"]
            not in {"floor", "glove", "head", "torso", "guard", "native_non_scoring"}
        ):
            raise ValueError("Frozen physical registry differs from compiled surfaces")
    qpos = np.asarray(report["initial_qpos"], np.float64)
    if qpos.shape != (170,) or not np.isfinite(qpos).all():
        raise ValueError("Frozen initial configuration malformed")
    scene = FrozenScene(model, report, ids)
    for a, ix in ids.items():
        NativeBodyAdapter(scene, a)  # Native root/joint/activation/ownership checks.
        ix.setflags(write=False)
    return scene
