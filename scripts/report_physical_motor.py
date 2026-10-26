"""Small Git evidence plus full raw manifest; large trajectories stay in raw archive."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from joxing.physical_motor import PhysicalMotorTask
from joxing.provenance import write_json
from joxing.stance_train import file_hash


def read(p):
    return json.loads(p.read_text())


def plots(run, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read(run / "evaluations.json")
    p = read(run / "protocol.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#8896ab", "#72daca", "#e5b15b", "#b3a0eb"]
    for i, v in enumerate(p["variants"]):
        rr = [r for r in rows if r["variant"] == v["id"] and r["status"] == "completed"]
        wins = sum(r["horizon_reached"] for r in rr)
        axes[0].bar(i, wins, color=colors[i])
        axes[0].text(i, wins + 0.4, f"{wins}/20", ha="center")
        axes[1].scatter(
            np.full(len(rr), i),
            [r["elapsed_sim_s"] for r in rr],
            color=colors[i],
            alpha=0.35,
        )
        tracking = [
            r["first_native_tracking_failure_s"]
            for r in rr
            if r["first_native_tracking_failure_s"] is not None
        ]
        axes[1].scatter(
            np.full(len(tracking), i),
            tracking,
            color="#b44",
            marker="x",
            alpha=0.6,
            label="Original tracking failure" if i == 0 else None,
        )
        axes[2].scatter(
            np.full(len(rr), i),
            [r["mean_clipped_fraction"] for r in rr],
            color=colors[i],
            alpha=0.35,
        )
    axes[0].axhline(16, color="#b44", linestyle="--", label="Registered16/20threshold")
    axes[0].set(
        title="Physical20-second completions",
        ylim=(0, 22),
        ylabel="Completions /20planned",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(title="Every attempted seed", ylabel="Physical simulated seconds")
    axes[1].axhline(20, color="#777", linestyle=":")
    axes[1].legend(fontsize=8)
    axes[2].set(
        title="Frozen normalizer clipping", ylabel="Mean fraction of features at clip"
    )
    for ax in axes:
        ax.set_xticks(
            range(len(p["variants"])), [v["id"] for v in p["variants"]], rotation=20
        )
    fig.suptitle(
        "Joxing v0.0.8 | physical motor continuation | zero training; no boxing claim"
    )
    fig.tight_layout()
    fig.savefig(
        out / "physical-motor.png", dpi=160, bbox_inches="tight", pad_inches=0.2
    )
    plt.close(fig)


def video(run, out, audit):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    p = read(run / "protocol.json")
    seed = p["replay"]["seed"]
    remaining = (
        p["caps"]["mission_physics_steps_with_audit_video"]
        - read(run / "result.json")["physics_steps"]
        - audit["audit_physics_steps"]
    )
    wall_remaining = (
        p["caps"]["wall_s"]
        - read(run / "result.json")["wall_s"]
        - audit["audit_wall_s"]
    )
    started = time.perf_counter()
    steps = 0
    tasks = []
    renderers = []
    actions = []
    cameras = []
    try:
        for variant in p["variants"]:
            task = PhysicalMotorTask(p["environment"], variant)
            folder = run / variant["id"]
            task.restore(read(folder / f"starts/{seed}.json"))
            tasks.append(task)
            task.model.vis.global_.offwidth = 512
            task.model.vis.global_.offheight = 288
            task.model.vis.quality.shadowsize = 1024
            task.model.vis.quality.offsamples = 2
            renderers.append(mujoco.Renderer(task.model, height=288, width=512))
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = task.pelvis
            cam.distance = 3.0
            cam.azimuth = 120
            cam.elevation = -15
            cam.lookat[:] = task.data.xpos[task.pelvis]
            cameras.append(cam)
            actions.append(np.load(folder / f"actions-{seed}.npz")["actions"])
        with imageio.get_writer(out / "physical-motor.mp4", fps=25) as writer:
            for tick in range(max(len(a) for a in actions)):
                panels = []
                for i, task in enumerate(tasks):
                    if tick < len(actions[i]):
                        if (
                            steps + task.env.frame_skip > remaining
                            or time.perf_counter() - started >= wall_remaining
                        ):
                            raise TimeoutError("Video mission cap")
                        task.step(actions[i][tick])
                        steps += task.env.frame_skip
                    if tick % 2 == 0:
                        renderers[i].update_scene(task.data, camera=cameras[i])
                        frame = Image.new("RGB", (512, 352), "#101722")
                        frame.paste(Image.fromarray(renderers[i].render()), (0, 64))
                        d = ImageDraw.Draw(frame)
                        d.text(
                            (8, 6),
                            f"{p['variants'][i]['id']} | seed{seed} | t={task.data.time:.2f}s",
                            fill="white",
                        )
                        d.text(
                            (8, 23),
                            "700native muscles | frozen policy commands | ordinary gravity",
                            fill="#72daca",
                        )
                        d.text(
                            (8, 40),
                            "ENDED - final frame held"
                            if tick >= len(actions[i])
                            else "Native motion diagnostic; no training or boxing claim",
                            fill="#e5b15b",
                        )
                        panels.append(np.asarray(frame))
                if tick % 2 == 0:
                    writer.append_data(np.concatenate(panels, axis=1))
        write_json(
            out / "render.json",
            {
                "seed": seed,
                "physics_steps": steps,
                "wall_s": time.perf_counter() - started,
                "scope": "Both first-scenario recorded-command CPU continuations; EGL is rendering only.",
                "report_script_sha256": file_hash(Path(__file__)),
            },
        )
    finally:
        for r in renderers:
            r.close()
        for t in tasks:
            t.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    p = read(args.run / "protocol.json")
    raw = {
        str(f.relative_to(args.run)): file_hash(f)
        for f in sorted(args.run.rglob("*"))
        if f.is_file() and f.suffix not in (".zip", ".pkl")
    }
    write_json(args.out / "raw-manifest.json", raw)
    for rel in raw:
        f = args.run / rel
        # Keep all attempt rows, protocols, maps, start states and telemetry in
        # Git; only first-seed full commands/traces. All40full trajectories are
        # retained in the separate public release archive, avoiding large Git runs.
        first = str(p["replay"]["seed"]) in f.stem or f.name == "replay.json"
        if (
            f.name.startswith(("actions-", "trace-", "substeps-", "partial-"))
            and not first
        ):
            continue
        target = args.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
    plots(args.run, args.out)
    if args.video:
        video(args.run, args.out, read(args.audit))
    write_json(
        args.out / "artifact-manifest.json",
        {
            str(f.relative_to(args.out)): file_hash(f)
            for f in sorted(args.out.rglob("*"))
            if f.is_file() and f.name != "artifact-manifest.json"
        },
    )
