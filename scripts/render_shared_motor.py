"""Render only the first fixed source, including any physical failure. Zero steps."""

import argparse
import json
import resource
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from joxing.env import INTEGRATION
from joxing.evidence_storage import read_arrays, read_json
from joxing.motor_storage import RawCredit, write_json
from joxing.shared_motor_audit import byte_sha
from joxing.shared_motor_proof import RUN, file_sha, read_packet
from joxing.shared_motor_task import build_motor_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    p = read_packet(root, args.protocol_sha256, args.source_sha)
    run = root / RUN
    audited = json.loads((run / "independent-audit.json").read_text())
    if (
        not audited["passed"]
        or audited["source_commit"] != args.source_sha
        or audited["protocol_sha256"] != args.protocol_sha256
    ):
        raise ValueError("Saved evidence audit prerequisite")
    scene = build_motor_scene()
    folder = run / p["recording_source"]
    trace, tape = read_json(folder / "trace.json"), read_arrays(folder / "tape.npz")
    if trace["contract"]["scene_hash"] != scene.report["scene_hash"]:
        raise ValueError("Render scene drift")
    target = run / "first-pretrained-trial.mp4"
    if target.exists():
        raise FileExistsError("Never overwrite the first trial clip")
    credit = RawCredit(run, p["raw_bytes"] - 512 * 1024)
    credit.check(target, 4 * 1024**2 + 65536)
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024**2, 4 * 1024**2))
    data = mujoco.MjData(scene.model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 5.5, 115, -12
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    pelvis = [scene.model.body(a + "_pelvis").id for a in ("A", "B")]
    with (
        mujoco.Renderer(scene.model, height=576, width=1024) as renderer,
        imageio.get_writer(
            target,
            fps=25,
            codec="libx264",
            quality=8,
            ffmpeg_params=["-threads", "1", "-maxrate", "3M", "-bufsize", "3M"],
        ) as writer,
    ):
        for tick, state in zip(tape["video_ticks"], tape["video_states"], strict=True):
            if tick and byte_sha(state) != trace["state_sha256"][int(tick) - 1]:
                raise ValueError("Recorded frame/engine drift")
            mujoco.mj_setState(scene.model, data, state, INTEGRATION)
            mujoco.mj_forward(scene.model, data)  # Scratch visual data only.
            camera.lookat[:] = [*np.mean(data.xpos[pelvis, :2], axis=0), 0.9]
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1024, 75), fill=(16, 24, 39))
            draw.text(
                (14, 10),
                "FROZEN PRETRAINED LOCOMOTION - SHARED MOTOR DIAGNOSTIC",
                fill="white",
            )
            draw.text(
                (14, 31),
                f"First registered trial 71000 | t={tick / 1000:.3f}s | 700 muscles per athlete | No training updates",
                fill="white",
            )
            draw.text(
                (14, 52), "Walking transfer; boxing points unavailable", fill="white"
            )
            draw.rectangle((0, 530, 1024, 576), fill=(16, 24, 39))
            for x, a, color in [(14, "A", "#ff8576"), (520, "B", "#83baff")]:
                physical, tracking = (
                    trace["physical_failure"][a],
                    trace["tracking_failure"][a],
                )
                status = (
                    physical["reason"]
                    if physical and tick >= physical["tick"]
                    else "no physical latch yet"
                )
                if tracking and tick >= tracking["tick"]:
                    status += " | tracking failed"
                draw.text((x, 540), a + ": " + status, fill=color)
            writer.append_data(np.asarray(frame))
    write_json(
        run / "first-pretrained-trial.json",
        {
            "source_commit": args.source_sha,
            "protocol_sha256": args.protocol_sha256,
            "sha256": file_sha(target),
            "bytes": target.stat().st_size,
            "frames": len(tape["video_ticks"]),
            "fps": 25,
            "new_render_physics_steps": 0,
            "seed": 71000,
            "selection": "first fixed trial, complete retained horizon",
            "short_motor_gate_passed": audited["short_motor_gate_passed"],
        },
    )


if __name__ == "__main__":
    main()
