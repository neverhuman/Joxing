"""Render v0.0.2 frozen-protocol evidence, never select a favorable evaluation clip."""

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

from joxing.strike import StrikeConfig, StrikeRig


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/v0.0.2"))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    protocol = read(args.run / "protocol.json")
    result = read(args.run / "result.json")
    if protocol["pilot"]:
        raise ValueError(
            "Proof report requires the final protocol, not the development pilot"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = result["seeds"]
    # Fixed by protocol: first training seed, first final-evaluation scenario.
    directory = args.run / f"seed-{protocol['training_seeds'][0]}"
    replay = [
        np.load(directory / f"{name}-replay.npz", allow_pickle=False)
        for name in ("initial", "trained")
    ]
    records = [
        read(directory / f"{name}-replay.json") for name in ("initial", "trained")
    ]
    palette = ["#64dec1", "#70aafc", "#ffca75"]
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.patch.set_facecolor("#101827")
    for ax in axes.flat:
        ax.set_facecolor("#101827")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    for color, seed in zip(palette, seeds):
        history = seed["history"]
        axes[0, 0].plot(
            [h["episodes"] for h in history],
            [100 * h["mean_strike_withdrawal_success"] for h in history],
            color=color,
            label=f"Seed {seed['training_seed']}",
            lw=2,
        )
    axes[0, 0].set(
        title="Learning: strike + withdrawal",
        xlabel="Training episodes per seed",
        ylabel="Candidate success (%)",
        ylim=(-3, 103),
    )
    axes[0, 0].legend(frameon=False)
    x = np.arange(len(seeds))
    width = 0.32
    for i, (name, color, label) in enumerate(
        (("initial", "#8397b5", "Reach baseline"), ("trained", "#64dec1", "v0.0.2"))
    ):
        values = [
            100 * s["evaluation"][name]["strike_withdrawal_success_rate"] for s in seeds
        ]
        bars = axes[0, 1].bar(
            x + (i - 0.5) * width, values, width=width, color=color, label=label
        )
        for bar, value in zip(bars, values):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                max(2, value + 1),
                f"{value:.0f}%",
                ha="center",
                fontsize=10,
            )
    axes[0, 1].set(
        title="64 fresh paired resets per seed",
        xticks=x,
        xticklabels=[str(s["training_seed"]) for s in seeds],
        xlabel="Independent training seed",
        ylabel="Strike + withdrawal success (%)",
        ylim=(0, 128),
    )
    axes[0, 1].legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
    for name, color, archive in zip(
        ("Reach baseline", "v0.0.2"), ("#8397b5", "#64dec1"), replay
    ):
        trace = archive["trace"]
        axes[1, 0].plot(trace[:, 0], trace[:, 4], color=color, label=name, lw=1.8)
        axes[1, 1].plot(trace[:, 0], 100 * trace[:, 5], color=color, label=name, lw=1.8)
    axes[1, 0].set(
        title="Measured glove contact force",
        xlabel="Force-evaluation time (s)",
        ylabel="Normal force (N)",
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 1].axhline(2.0, color="#ffca75", ls="--", lw=1.2, label="2 cm clearance")
    axes[1, 1].set(
        title="Physical withdrawal from target",
        xlabel="Force-evaluation time (s)",
        ylabel="Glove/target clearance (cm)",
    )
    axes[1, 1].legend(frameon=False)
    state = "PASS" if result["passed"] else "GATE NOT MET"
    fig.suptitle(
        f"Joxing v0.0.2 · muscle → contact → score → withdrawal · {state}",
        fontsize=17,
        x=0.055,
        ha="left",
    )
    fig.text(
        0.055,
        0.025,
        "Assisted rig: fixed pelvis, zero gravity · All 700 muscles retained · Model-predicted force · No self-play",
        fontsize=10,
        color="#aabbd0",
    )
    fig.tight_layout(rect=[0.025, 0.055, 0.99, 0.94])
    fig.savefig(args.out / "proof.png", dpi=160)
    plt.close(fig)
    if not args.video:
        return
    config = StrikeConfig(
        **{
            **protocol["config"],
            "target": tuple(protocol["config"]["target"]),
            "phase_ends_s": tuple(protocol["config"]["phase_ends_s"]),
        }
    )
    rig = StrikeRig(config)
    model = rig.model
    model.vis.quality.shadowsize = 1024
    model.vis.quality.offsamples = 2
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 640
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.12, -0.03, 1.18]
    camera.distance = 1.8
    camera.azimuth = 125
    camera.elevation = -8
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    font_path = Path("/System/Library/Fonts/Helvetica.ttc")
    font = (
        ImageFont.truetype(str(font_path), 20)
        if font_path.exists()
        else ImageFont.load_default()
    )
    frames = []
    with mujoco.Renderer(model, height=640, width=640) as renderer:
        for k in range(0, len(replay[0]["qpos"]), 4):
            panels = []
            for j, archive in enumerate(replay):
                rig.data.qpos[:] = archive["qpos"][k]
                rig.data.mocap_pos[0] = archive["target"]
                mujoco.mj_forward(model, rig.data)
                renderer.update_scene(rig.data, camera=camera, scene_option=option)
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                trace = archive["trace"][k]
                draw.rectangle((0, 0, 640, 92), fill=(16, 24, 39))
                draw.text(
                    (16, 12),
                    "REACH BASELINE" if j == 0 else "v0.0.2 TRAINED STRIKE + WITHDRAW",
                    font=font,
                    fill=(235, 242, 250),
                )
                draw.text(
                    (16, 43),
                    f"Points {int(trace[7])}  |  Force {trace[4]:.0f} N  |  Clearance {100 * trace[5]:.1f} cm",
                    font=font,
                    fill=(235, 242, 250),
                )
                panels.append(np.array(panel))
            canvas = Image.fromarray(np.concatenate(panels, axis=1))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 556, 1280, 640), fill=(16, 24, 39))
            draw.text(
                (16, 568),
                f"ASSISTED RIG | fixed pelvis, zero gravity | t={replay[0]['trace'][k, 0]:.3f}s | slow-motion state replay",
                font=font,
                fill=(235, 242, 250),
            )
            draw.text(
                (16, 600),
                f"Fixed proof scenario {protocol['diagnostic_scenario_seed']} | Seed {protocol['training_seeds'][0]} | Final withdrawal: baseline {records[0]['strike_withdrawal_success']}, trained {records[1]['strike_withdrawal_success']}",
                font=font,
                fill=(170, 187, 208),
            )
            frames.append(np.array(canvas))
    frames += [frames[-1]] * 20
    imageio.mimsave(
        args.out / "strike-before-after.mp4", frames, fps=30, macro_block_size=16
    )


if __name__ == "__main__":
    main()
