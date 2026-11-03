"""Render actual saved prototype frames only; never advance physics."""

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
from joxing.native_tape_storage import load
from joxing.scene import build_scene


def render(folder, label, source_sha):
    folder = Path(folder)
    summary = json.loads((folder / "evaluation.json").read_text())
    arrays = load(folder / "frames.npz", summary["frame_tape"])
    target = folder / (label + ".mp4")
    if target.exists():
        raise FileExistsError("Fixed video must not be overwritten")
    scene = build_scene()
    data = mujoco.MjData(scene.model)
    camera = mujoco.MjvCamera()
    camera.distance = 5.0
    camera.azimuth = 110
    camera.elevation = -15
    pelvis = [scene.model.body(p + "_pelvis").id for p in "AB"]
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    count = len(arrays["qpos"])
    selected = sorted({0, count // 2, count - 1})
    with (
        mujoco.Renderer(scene.model, height=448, width=800) as renderer,
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
            renderer.update_scene(data, camera=camera, scene_option=option)
            image = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(image)
            d.rectangle((0, 0, 800, 62), fill=(16, 24, 39))
            d.text(
                (12, 8),
                f"H31 NATIVE TWO-HUMANOID MOTOR PPO PROTOTYPE | {label}",
                fill="white",
            )
            d.text(
                (12, 28),
                f"t={float(data.time):.3f}s | 700 muscles each | source {source_sha[:8]}",
                fill="white",
            )
            d.text(
                (12, 46),
                "Motor learning witness; competitive boxing / sporting points unqualified",
                fill="white",
            )
            d.rectangle((0, 410, 800, 448), fill=(16, 24, 39))
            d.text(
                (12, 420),
                f"Entire fixed evaluation | candidate points {summary['candidate_points']} | failure {summary['motor_failures']}",
                fill="white",
            )
            writer.append_data(np.asarray(image))
            if i in selected:
                image.save(folder / f"{label}-frame-{i:03d}.png")
    result = {
        "frames": count,
        "fps": 50,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bytes": target.stat().st_size,
        "render_physics": 0,
        "selection": "Entire first fixed registered before/after evaluation, including motor failures",
        "source_commit": source_sha,
    }
    (folder / (label + "-video.json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--label", choices=["before", "after"], required=True)
    ap.add_argument("--source-sha", required=True)
    a = ap.parse_args()
    print(json.dumps(render(a.folder, a.label, a.source_sha)))
