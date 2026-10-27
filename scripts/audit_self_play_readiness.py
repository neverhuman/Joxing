"""Exploratory zero-step scene checks for the supplied self-play proposals.

Direct scratch geom placement below is a geometric query, never evaluated boxing,
physical contact/impulse evidence or a model change. No mj_step or actor is called.
"""

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from joxing.provenance import ROOT, manifest, write_json
from joxing.scene import build_scene


def audit():
    began = time.monotonic()
    scene = build_scene()
    model = scene.model
    data = mujoco.MjData(model)
    data.qpos[:] = scene.report["initial_qpos"]
    mujoco.mj_forward(model, data)
    geoms = {g["name"]: g for g in scene.report["physical_geoms"]}
    pelvis = {a: data.xpos[model.body(a + "_pelvis").id].copy() for a in ("A", "B")}
    key_names = [
        "sternum",
        "head_neck",
        "toes_r",
        "toes_l",
        "proximal_row_r",
        "proximal_row_l",
    ]
    features = {
        a: np.asarray(
            [data.xpos[model.body(a + "_" + b).id] - pelvis[a] for b in key_names]
        )
        for a in ("A", "B")
    }
    rotation = np.diag([-1.0, -1.0, 1.0])
    pairs = []
    for a, b in [
        ("A_glove_r", "A_shell_head"),
        ("A_shell_ulna_r", "A_shell_head"),
        ("A_glove_r", "B_shell_head"),
        ("A_glove_r", "B_glove_l"),
    ]:
        i, j = model.geom(a).id, model.geom(b).id
        enabled = bool(
            (model.geom_contype[i] & model.geom_conaffinity[j])
            or (model.geom_contype[j] & model.geom_conaffinity[i])
        )
        pairs.append({"geoms": [a, b], "collision_bits_enable": enabled})
    probe = model.geom("A_glove_r").id
    head = model.geom("B_shell_head").id
    center = data.geom_xpos[head].copy()
    fwd = data.xmat[model.body("B_pelvis").id].reshape(3, 3)[:, 0].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, fwd)
    candidates = [
        g
        for g in geoms.values()
        if g["actor"] == "B"
        and np.linalg.norm(data.geom_xpos[g["id"]] - center)
        <= 0.512 + model.geom_rbound[g["id"]]
    ]
    samples = []
    for el in range(-40, 41, 10):
        for az in range(-80, 81, 10):
            if time.monotonic() - began > 45:
                raise TimeoutError("Static geometry audit45s cap")
            a, e = np.deg2rad([az, el])
            direction = (
                np.cos(e) * (np.cos(a) * fwd + np.sin(a) * left) + np.sin(e) * up
            )
            first = None
            for distance in np.arange(0.45, -0.2001, -0.004):
                point = center + distance * direction
                data.geom_xpos[probe] = point
                hits = []
                for g in candidates:
                    gid = g["id"]
                    if (
                        np.linalg.norm(point - data.geom_xpos[gid])
                        > 0.062 + model.geom_rbound[gid] + 1e-9
                    ):
                        continue
                    gap = mujoco.mj_geomDistance(model, data, probe, gid, 1.0, None)
                    if gap <= 0:
                        hits.append((gap, gid, g))
                if hits:
                    # First sampled depth; largest penetration resolves sampled ties.
                    gap, gid, g = min(hits, key=lambda x: (x[0], x[1]))
                    first = {
                        "geom": g["name"],
                        "role": g["role"],
                        "gap_m": float(gap),
                        "sample_distance_m": float(distance),
                    }
                    break
            samples.append({"azimuth_deg": az, "elevation_deg": el, "first": first})
    # Restore scratch query placement before reporting actual spawn gaps.
    mujoco.mj_forward(model, data)
    return manifest() | {
        "scope": "exploratory_static_geometry_not_physical_hit_or_training_qualification",
        "source_files": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve(),
                ROOT / "src/joxing/scene.py",
                ROOT / "src/joxing/rules.py",
            ]
        },
        "new_physics_steps": 0,
        "optimizer_updates": 0,
        "boxing_games": 0,
        "dimensions": scene.report["dimensions"],
        "scene_hash": scene.report["scene_hash"],
        "integrator": mujoco.mjtIntegrator(int(model.opt.integrator)).name,
        "physics_dt_s": float(model.opt.timestep),
        "solver_tolerance": float(model.opt.tolerance),
        "collision_pairs": pairs,
        "skull": geoms.get("B_skull"),
        "key_body_names": key_names,
        "world_axis_key_features_corner_max_error_m": float(
            np.max(abs(features["A"] - features["B"]))
        ),
        "yaw_canonical_key_features_corner_max_error_m": float(
            np.max(abs(features["A"] - features["B"] @ rotation.T))
        ),
        "spawn_glove_head_surface_gap_m": float(
            mujoco.mj_geomDistance(model, data, probe, head, 10.0, None)
        ),
        "geometry_probe": {
            "sphere_radius_m": 0.062,
            "march_step_m": 0.004,
            "poses": 1,
            "direct_scratch_geom_placement": True,
            "samples": samples,
            "first_roles": dict(
                Counter(s["first"]["role"] if s["first"] else "none" for s in samples)
            ),
            "limitations": "Neutral pose only;4mm sampled first depth and tie rule. No physical punch, impulses, HitTracker award, guard-pose or convergence result.",
        },
        "wall_s": time.monotonic() - began,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit()
    write_json(args.out, result)
    print(
        {k: v for k, v in result.items() if k not in {"geometry_probe", "source_files"}}
    )
    print(result["geometry_probe"]["first_roles"])
