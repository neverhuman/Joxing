"""Fixed development-scenario evidence; labels failed balance gates explicitly."""

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from joxing.stance_env import StanceEnv


def main():
    import torch
    from stable_baselines3 import SAC

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    result = json.loads((args.run / "result.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    before = json.loads((args.run / "initial-evaluations.json").read_text())
    after = json.loads((args.run / "trained-evaluations.json").read_text())
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    episode_rows = []
    for path in args.run.glob("worker-*.monitor.csv"):
        with path.open() as f:
            episode_rows.extend(
                csv.DictReader(line for line in f if not line.startswith("#"))
            )
    episode_rows.sort(key=lambda x: float(x["t"]))
    durations = np.array([float(x["l"]) * 0.01 for x in episode_rows])
    times = np.array([float(x["t"]) for x in episode_rows])
    axes[0].scatter(times, durations, s=3, alpha=0.18, color="#72daca")
    if len(durations) >= 32:
        axes[0].plot(
            times[31:],
            np.convolve(durations, np.ones(32) / 32, mode="valid"),
            color="#f6c45e",
            label="32-episode mean",
        )
        axes[0].legend()
    axes[0].set(
        title="Training episode duration",
        xlabel="Learner wall time (s)",
        ylabel="Simulated seconds (10 ms per control step)",
    )
    x = np.arange(len(before))
    axes[1].plot(
        x, [r["elapsed_sim_s"] for r in before], "o-", label="Initial", color="#929db3"
    )
    axes[1].plot(
        x, [r["elapsed_sim_s"] for r in after], "o-", label="Trained", color="#72daca"
    )
    axes[1].set(
        title="Paired development resets",
        xlabel="Scenario index",
        ylabel="Survival seconds",
    )
    axes[1].legend()
    s = result["summary"]
    text = f"{result['training_transitions']:,} SAC transitions\nSeed {result['seed']} | CPU MuJoCo\n\nMean survival\n{s['initial_mean_survival_s']:.3f} -> {s['trained_mean_survival_s']:.3f} s\n\n5-second survivors: {s['trained_5s_survivors']}/{len(after)}\n20-second successes: {s['trained_20s_successes']}/{len(after)}\nInvalids: {s['invalid']}\n\nMotor gate: NOT PASSED\nDevelopment pilot; no self-play"
    axes[2].axis("off")
    axes[2].text(0.03, 0.96, text, va="top", fontsize=12, linespacing=1.45)
    fig.suptitle(
        "Joxing v0.0.4 | real gravity, full muscles, feedback learning", fontsize=15
    )
    fig.tight_layout()
    fig.savefig(args.out / "balance-learning.png", dpi=160)
    plt.close(fig)
    if not args.video:
        return
    env = StanceEnv(training=False)
    recordings = []
    for checkpoint in ("initial", "final"):
        policy = SAC.load(args.run / f"{checkpoint}.zip", device="cpu")
        obs, _ = env.reset(options={"scenario_id": protocol["eval_ids"][0]})
        states = [(env.data.qpos.copy(), env.data.time, 0.0, None)]
        while not env.done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
            states.append(
                (
                    env.data.qpos.copy(),
                    env.data.time,
                    info["max_tilt_rad"],
                    info["reason"],
                )
            )
        recordings.append(states)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.9]
    camera.distance = 2.7
    camera.azimuth = 115
    camera.elevation = -10
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    with (
        mujoco.Renderer(env.model, height=480, width=640) as renderer,
        imageio.get_writer(
            args.out / "balance-before-after.mp4", fps=25, codec="libx264", quality=8
        ) as writer,
    ):
        # Shared simulated clock; ended policy holds its last frame with explicit label.
        for tick in range(0, max(map(len, recordings)), 4):
            frames = []
            for label, states in zip(("INITIAL", "TRAINED"), recordings):
                index = min(tick, len(states) - 1)
                qpos, time_s, tilt, reason = states[index]
                env.data.qpos[:] = qpos
                mujoco.mj_forward(env.model, env.data)
                renderer.update_scene(env.data, camera=camera, scene_option=option)
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 640, 72), fill=(15, 22, 34))
                draw.text(
                    (12, 10),
                    f"{label} | scenario {protocol['eval_ids'][0]} | t={time_s:.2f}s",
                    fill="white",
                )
                draw.text(
                    (12, 29),
                    "Gravity -9.81 | native root | 700 muscles | no evaluation assistance",
                    fill="#78dfc9",
                )
                draw.text(
                    (12, 48),
                    f"Max pelvis tilt {np.rad2deg(tilt):.1f} deg | {reason or 'running'}"
                    + (" | ENDED, FRAME HELD" if tick >= len(states) else ""),
                    fill="#ffd58a",
                )
                frames.append(np.asarray(panel))
            writer.append_data(np.concatenate(frames, axis=1))


if __name__ == "__main__":
    main()
