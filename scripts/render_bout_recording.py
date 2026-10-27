"""Render saved shared-world states, with no additional simulation or scoring."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from joxing.bout_recording import STATE, frame_schedule, load_recording, recording_hash
from joxing.provenance import write_json
from joxing.scene import build_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = build_scene()
    report, states = load_recording(scene, args.recording)
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite a recording render")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(scene.model)  # Scratch display state, never the live match.
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.9]
    camera.distance, camera.azimuth, camera.elevation = 4.2, 100, -12
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    indices = frame_schedule(report)
    with (
        mujoco.Renderer(scene.model, height=576, width=1024) as renderer,
        imageio.get_writer(args.out, fps=25, codec="libx264", quality=8) as writer,
    ):
        for index in indices:
            mujoco.mj_setState(scene.model, data, states[index], STATE)
            mujoco.mj_forward(scene.model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1024, 65), fill=(16, 24, 39))
            draw.text((14, 10), report["classification"], fill="white")
            row = report["frames"][index]
            draw.text(
                (14, 32),
                f"t={row['time_s']:.3f}s | Points unavailable | Contact peak {row['contact_peak_N']:.1f} N (model)",
                fill="white",
            )
            draw.rectangle((0, 530, 1024, 576), fill=(16, 24, 39))
            draw.text((14, 540), "A: " + report["policies"]["A"][:70], fill="#ff8576")
            draw.text((520, 540), "B: " + report["policies"]["B"][:70], fill="#83baff")
            writer.append_data(np.asarray(frame))
    write_json(
        args.out.with_suffix(".json"),
        {
            "recording_hash": recording_hash(report),
            "frames": len(indices),
            "fps": 25,
            "new_physics_steps": 0,
            "scope": "Scratch visual playback only; no new actor decisions or referee events",
        },
    )


if __name__ == "__main__":
    main()
