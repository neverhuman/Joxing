"""Render measured reach learning and recorded physical states; no new policy rollout."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from joxing.motor import ReachConfig, ReachRig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.run / "result.json").read_text())
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    fig.patch.set_facecolor("#101827")
    for ax in axes:
        ax.set_facecolor("#101827")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    history = report["history"]
    axes[0].plot(
        [r["episodes"] for r in history],
        [100 * r["mean_error_m"] for r in history],
        color="#66e0bf",
        lw=2.5,
    )
    axes[0].set(
        xlabel="Training episodes",
        ylabel="Mean hand-target error (cm)",
        title="384 episodes of muscle policy search",
    )
    bars = axes[1].bar(
        ["Initial", "Learned"],
        100 * np.array([report["initial_error_m"], report["trained_error_m"]]),
        color=["#8397b5", "#66e0bf"],
        width=0.55,
    )
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{bar.get_height():.1f} cm",
            ha="center",
        )
    axes[1].set(
        title="32 held-out paired resets",
        ylabel="Terminal-window error (cm)",
        ylim=(0, 40),
    )
    fig.suptitle(
        "Joxing · first muscle-learning experiment", fontsize=17, x=0.07, ha="left"
    )
    fig.text(
        0.07,
        0.015,
        "Assisted rig: fixed pelvis, zero gravity · CEM open-loop muscle pulses · Not boxing/self-play",
        fontsize=9,
        color="#aabbd0",
    )
    fig.tight_layout(rect=[0.03, 0.045, 0.99, 0.91])
    fig.savefig(args.out / "reach-learning.png", dpi=160)
    plt.close(fig)
    if not args.video:
        return
    config = ReachConfig(
        **{**report["config"], "target": tuple(report["config"]["target"])}
    )
    rig = ReachRig(config)
    m = rig.model
    m.vis.quality.shadowsize = 1024
    m.vis.quality.offsamples = 2
    m.vis.global_.offwidth = 640
    m.vis.global_.offheight = 640
    replay = [
        np.load(args.run / f"{name}_replay.npz") for name in ("initial", "trained")
    ]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.08, -0.03, 1.18]
    camera.distance = 1.75
    camera.azimuth = 125
    camera.elevation = -8
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    font = (
        ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 23)
        if Path("/System/Library/Fonts/Helvetica.ttc").exists()
        else ImageFont.load_default()
    )
    frames = []
    with mujoco.Renderer(m, height=640, width=640) as renderer:
        for k in range(0, len(replay[0]["qpos"]), 7):
            panels = []
            for j, record in enumerate(replay):
                rig.data.qpos[:] = record["qpos"][k]
                mujoco.mj_forward(m, rig.data)
                renderer.update_scene(rig.data, camera=camera, scene_option=option)
                scene = renderer.scene
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.025, 0, 0]),
                    record["target"],
                    np.eye(3).ravel(),
                    np.array([0.3, 1.0, 0.5, 1.0]),
                )
                scene.ngeom += 1
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 640, 65), fill=(16, 24, 39))
                draw.text(
                    (20, 18),
                    "INITIAL" if j == 0 else "LEARNED MUSCLE PULSES",
                    font=font,
                    fill=(235, 242, 250),
                )
                panels.append(np.array(panel))
            canvas = Image.fromarray(np.concatenate(panels, axis=1))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 575, 1280, 640), fill=(16, 24, 39))
            draw.text(
                (20, 590),
                f"ASSISTED MOTOR RIG | fixed pelvis, zero gravity | t={k * config.dt_s:.3f}s | slow motion",
                font=font,
                fill=(235, 242, 250),
            )
            frames.append(np.array(canvas))
    frames += [frames[-1]] * 20
    imageio.mimsave(
        args.out / "reach-before-after.mp4", frames, fps=30, macro_block_size=16
    )


if __name__ == "__main__":
    main()
