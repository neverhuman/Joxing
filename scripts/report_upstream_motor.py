"""Curate native qualification evidence; weights remain in the pinned release."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from joxing.provenance import write_json
from joxing.stance_train import file_hash
from joxing.upstream_motor import MODES, NativeTask


def read(path):
    return json.loads(path.read_text())


def plots(run, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read(run / "evaluations.json")
    summary = read(run / "result.json")["modes"]
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#8896ab", "#72daca"]
    for i, mode in enumerate(MODES):
        s = summary[mode]
        axes[0].bar(i, s["horizon_completions"], color=colors[i])
        axes[0].text(
            i,
            s["horizon_completions"] + 0.4,
            f"{s['horizon_completions']}/20",
            ha="center",
        )
        if s["mean_elapsed_sim_s"] is not None:
            axes[1].bar(i, s["mean_elapsed_sim_s"], color=colors[i])
    for seed in range(71000, 71020):
        pair = [
            next(r for r in rows if r["seed"] == seed and r["mode"] == m) for m in MODES
        ]
        if all(r["status"] == "completed" for r in pair):
            axes[2].plot([0, 1], [r["elapsed_sim_s"] for r in pair], "o-", alpha=0.35)
    labels = ["Default load", "Restored schedule"]
    axes[0].axhline(16, linestyle="--", color="#f6c45e", label="Restored gate:16/20")
    axes[0].set(
        title="Native three-cycle completions",
        ylim=(0, 22),
        ylabel="Completed tasks /20planned",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(title="Mean task duration", ylabel="Native simulated seconds")
    axes[2].set(title="Every paired start", ylabel="Native simulated seconds")
    for ax in axes:
        ax.set_xticks([0, 1], labels, rotation=10)
    fig.suptitle(
        "Joxing v0.0.6 | frozen upstream locomotion | no training or boxing claim"
    )
    fig.tight_layout()
    fig.savefig(out / "qualification.png", dpi=160, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def video(run, out):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    p = read(run / "protocol.json")
    seed = p["replay_seed"]
    tasks, renderers, cameras, actions, panels = [], [], [], [], []
    for mode in MODES:
        task = NativeTask(p["environment"])
        task.reset(seed)
        task.restore(read(run / f"starts/{mode}-{seed}.json"))
        task.model.vis.quality.shadowsize = 2048
        task.model.vis.quality.offsamples = 4
        renderer = mujoco.Renderer(task.model, height=360, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = task.pelvis
        cam.distance, cam.azimuth, cam.elevation = 3.0, 120, -15
        cam.lookat[:] = task.data.xpos[task.pelvis]
        tasks.append(task)
        renderers.append(renderer)
        cameras.append(cam)
        with np.load(run / f"actions-{mode}-{seed}.npz") as data:
            actions.append(data["actions"].copy())
        panels.append(None)
    length = max(len(a) for a in actions)
    try:
        with imageio.get_writer(out / "native-controller.mp4", fps=25) as writer:
            for step in range(max(length, 50)):
                for i, task in enumerate(tasks):
                    if step < len(actions[i]):
                        task.step(actions[i][step])
                    if step % 2 == 0:
                        renderers[i].update_scene(task.data, camera=cameras[i])
                        frame = Image.new("RGB", (640, 440), "#101722")
                        frame.paste(Image.fromarray(renderers[i].render()), (0, 80))
                        draw = ImageDraw.Draw(frame)
                        draw.text(
                            (10, 8),
                            f"{MODES[i]} | seed{seed} | native t={task.data.time:.2f}s",
                            fill="white",
                        )
                        draw.text(
                            (10, 27),
                            "700native muscles | ordinary gravity | recorded commands",
                            fill="#72daca",
                        )
                        draw.text(
                            (10, 46),
                            "ENDED - final frame held"
                            if step >= len(actions[i])
                            else "Native locomotion diagnostic; no boxing or training claim",
                            fill="#f6c45e",
                        )
                        panels[i] = np.asarray(frame)
                if step % 2 == 0:
                    writer.append_data(np.concatenate(panels, axis=1))
    finally:
        for renderer in renderers:
            renderer.close()
        for task in tasks:
            task.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.run.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".npz", ".txt"}:
            continue
        target = args.out / path.relative_to(args.run)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise FileExistsError(f"Refusing to rewrite raw evidence: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    plots(args.run, args.out)
    if args.video:
        video(args.run, args.out)
    write_json(
        args.out / "artifact-manifest.json",
        {
            str(p.relative_to(args.out)): file_hash(p)
            for p in sorted(args.out.rglob("*"))
            if p.is_file() and p.name != "artifact-manifest.json"
        },
    )
