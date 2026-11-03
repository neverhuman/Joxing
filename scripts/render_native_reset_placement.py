"""Post-task compact rendering of exact archived frames; never advance physics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def render(folder, label, source_sha, report):
    folder = Path(folder)
    from joxing.native_reset_placement import (
        NativeResetPlacementSpec,
        build_reset_scene,
    )
    from joxing.native_tape_storage import load
    from joxing.provenance import ROOT as PROJECT_ROOT
    from joxing.shared_motor_task import NativeReferenceDriver
    from joxing.upstream_motor import NativeTask

    summary = json.loads((folder / "evaluation.json").read_text())
    arrays = load(folder / "frames.npz", summary["frame_tape"])
    report = Path(report)
    report.mkdir(parents=True, exist_ok=False)
    target = report / (label + ".mp4")
    if target.exists():
        raise FileExistsError("Fixed video must not be overwritten")
    donor = NativeTask(
        json.loads((PROJECT_ROOT / "experiments/v0.0.6/protocol.json").read_text())[
            "environment"
        ]
    )
    initial = {p: NativeReferenceDriver(donor, summary["seed"]).initial for p in "AB"}
    scene = build_reset_scene(
        donor.model,
        initial,
        spec=NativeResetPlacementSpec(summary["placement"]),
        seed=summary["seed"],
    )
    donor.close()
    if scene.report["scene_hash"] != summary["contract"]["scene_hash"]:
        raise ValueError("Recorded stance scene identity")
    data = mujoco.MjData(scene.model)
    camera = mujoco.MjvCamera()
    camera.distance = 5.0
    camera.azimuth = 110
    camera.elevation = -15
    pelvis = [scene.model.body(p + "_pelvis").id for p in "AB"]
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    count = len(arrays["qpos"])
    if not 2 <= count <= 151 or count != summary["frames"]:
        raise ValueError("Complete bounded archived evaluation required")
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    selected = sorted({0, count // 2, count - 1})
    with (
        mujoco.Renderer(scene.model, height=288, width=512) as renderer,
        imageio.get_writer(
            target,
            fps=50,
            codec="libx264",
            quality=7,
            ffmpeg_params=["-threads", "1", "-maxrate", "2M", "-bufsize", "2M"],
        ) as writer,
    ):
        for i in range(count):
            data.qpos[:] = arrays["qpos"][i]
            data.qvel[:] = arrays["qvel"][i]
            data.act[:] = arrays["activation"][i]
            data.ctrl[:] = arrays["control"][i]
            data.time = float(arrays["time"][i])
            mujoco.mj_forward(scene.model, data)
            camera.lookat[:] = [*np.mean(data.xpos[pelvis, :2], axis=0), 0.9]
            visible = [
                g
                for g in range(scene.model.ngeom)
                if scene.model.geom(g).name.startswith(("A_", "B_"))
                and scene.model.geom_group[g] < 3
                and scene.model.geom_rgba[g, 3] > 0
            ]
            radius = max(
                np.linalg.norm(data.geom_xpos[g] - camera.lookat)
                + scene.model.geom_rbound[g]
                for g in visible
            )
            # Bounding sphere fits inside the smaller vertical FOV with1.6margin,
            # also leaving both HUD bands clear. Never crop a chosen athlete.
            camera.distance = max(
                5.0, 1.6 * radius / np.sin(np.deg2rad(scene.model.vis.global_.fovy / 2))
            )
            renderer.update_scene(data, camera=camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            image = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(image)
            d.rectangle((0, 0, 512, 44), fill=(16, 24, 39))
            d.text(
                (8, 5),
                f"H34 PHASE-INDEPENDENT STARTS | {label}",
                fill="white",
            )
            d.text(
                (8, 22),
                f"t={float(data.time):.3f}s | 700 muscles each | source {source_sha[:8]}",
                fill="white",
            )
            d.rectangle((0, 254, 512, 288), fill=(16, 24, 39))
            d.text(
                (8, 258),
                f"Full archived evaluation | points {summary['candidate_points']} | motor only",
                fill="white",
            )
            writer.append_data(np.asarray(image))
            if i in selected:
                image.save(report / f"{label}-frame-{i:03d}.png")
    result = {
        "profile": "reset512x288-fullbody-fit1.6-no-tendon-shadow-reflection-v1",
        "camera": "allvisibleathletegeom bounding sphere / sin(verticalfov/2),1.6margin,min5m; per actual archived frame",
        "source_frames_sha256": summary["frame_tape"]["sha256"],
        "physics_muscles_unchanged": 1400,
        "frames": count,
        "fps": 50,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bytes": target.stat().st_size,
        "render_physics": 0,
        "selection": "Entire first fixed reset71000 of each registered condition, including falls",
        "reference_profile": summary["reference_profile"],
        "placement": summary["placement"],
        "source_commit": source_sha,
    }
    (report / (label + "-video.json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    print(json.dumps(render(a.folder, a.label, a.source_sha, a.report)))
