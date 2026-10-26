"""Curate all experiment records and produce fixed-scenario evidence."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from joxing.provenance import write_json
from joxing.stance_train import file_hash
from joxing.wave import bootstrap_pair


def read(path):
    return json.loads(path.read_text())


def conclusions(run):
    wave = read(run / "result.json")
    selection = wave["selection"]
    comparison = {}
    screen_constant = read(run / "B0/screen-constant-evaluations.json")
    for arm in "ABCD":
        folder = run / f"S-{arm}"
        if not (folder / "result.json").exists():
            continue
        result = read(folder / "result.json")
        rows = read(folder / "trained-evaluations.json")
        comparison[f"S-{arm}"] = {
            "initial_to_final": result["summary"],
            "constant_to_final": bootstrap_pair(screen_constant, rows),
            "wall_s": result["wall_s"],
            "transitions_per_s_including_evaluation": result["training_transitions"]
            / result["wall_s"],
            "updates_per_optimizer_s": result["updates"] / result["update_wall_s"],
        }
    replicated = []
    if selection["qualified"]:
        baseline = read(run / "B0/confirmation-constant-evaluations.json")
        for seed in (263, 431):
            reference = run / f"R-A-{seed}"
            selected = reference if selection["arm"] == "A" else run / f"R-W-{seed}"
            a = read(reference / "trained-evaluations.json")
            b = read(selected / "trained-evaluations.json")
            r = read(selected / "result.json")
            contrast = bootstrap_pair(a, b)
            comparison[f"confirmation-{seed}"] = {
                "initial_to_final": r["summary"],
                "reference_to_selected": contrast,
                "constant_to_selected": bootstrap_pair(baseline, b),
            }
            replicated.append(
                r["scheduled_training_complete"]
                and r["summary"]["paired_reset_bootstrap_95ci_s"][0] > 0
                and (
                    selection["arm"] == "A"
                    or contrast["paired_reset_bootstrap_95ci_s"][0] > 0
                )
            )
    return {
        "selection": selection,
        "comparisons": comparison,
        "limited_replicated_improvement": len(replicated) == 2 and all(replicated),
        "motor_gate_passed": False,
        "scope": "Exploratory screen; two-seed confirmation only when qualified. Reset bootstrap is not training-seed uncertainty. No boxing/self-play claim.",
    }


def plot(run, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for i, arm in enumerate("ABCD"):
        folder = run / f"S-{arm}"
        if not (folder / "result.json").exists():
            continue
        r = read(folder / "result.json")
        s = r["summary"]
        axes[0].bar(
            i - 0.18,
            s["initial_mean_survival_s"],
            width=0.36,
            color="#8896ab",
            label="Initial" if i == 0 else None,
        )
        axes[0].bar(
            i + 0.18,
            s["trained_mean_survival_s"],
            width=0.36,
            color="#72daca",
            label="Final" if i == 0 else None,
        )
        ci = s["paired_reset_bootstrap_95ci_s"]
        gain = s["paired_survival_gain_s"]
        axes[1].errorbar(
            i,
            gain,
            yerr=[[gain - ci[0]], [ci[1] - gain]],
            fmt="o",
            color="#f6c45e",
            capsize=5,
        )
        p = read(folder / "progress.json")
        axes[2].plot([x["wall_s"] for x in p], [x["transitions"] for x in p], label=arm)
    baseline = read(run / "B0/screen-constant-evaluations.json")
    axes[0].axhline(
        np.mean([r["elapsed_sim_s"] for r in baseline]),
        linestyle="--",
        color="#f6c45e",
        label="Chosen constant",
    )
    axes[0].set(
        title="Screen: mean survival, 32 starts",
        ylabel="Physical seconds",
        xticks=range(4),
        xticklabels=["A auto/raw", "B fixed/raw", "C auto/x1000", "D fixed/x1000"],
    )
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="#8896ab")
    axes[1].axhline(
        0.25, color="#72daca", linestyle="--", label="Replication gain threshold"
    )
    axes[1].set(
        title="Paired survival gain + 95% reset CI",
        ylabel="Final minus initial seconds",
        xticks=range(4),
        xticklabels=list("ABCD"),
    )
    axes[1].legend(fontsize=8)
    axes[2].set(
        title="End-to-end progress",
        xlabel="Wall seconds (incl. setup/eval)",
        ylabel="Training transitions",
    )
    axes[2].legend()
    fig.suptitle(
        "Joxing v0.0.5 | controlled motor screen | no autonomous boxing", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(out / "wave-results.png", dpi=160)
    plt.close(fig)


def video(run, out):
    import imageio.v2 as imageio
    import mujoco
    import torch
    from PIL import Image, ImageDraw
    from stable_baselines3.sac.policies import Actor

    from joxing.stance_env import StanceEnv

    torch.set_num_threads(1)
    selection = read(run / "selection.json")
    arm = selection.get("arm") or "A"
    folder = run / f"S-{arm}"
    if not folder.exists():
        return
    env = StanceEnv(training=False)
    recordings = []
    for checkpoint in ("initial", "final"):
        actor = Actor.load(folder / f"{checkpoint}-actor.pt", device="cpu")
        obs, _ = env.reset(options={"scenario_id": 60000})
        states = [(env.data.qpos.copy(), 0.0, None)]
        while not env.done:
            action, _ = actor.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            states.append((env.data.qpos.copy(), env.data.time, env.result()["reason"]))
        recordings.append(states)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.9]
    camera.distance, camera.azimuth, camera.elevation = 2.7, 115, -10
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    with (
        mujoco.Renderer(env.model, height=480, width=640) as renderer,
        imageio.get_writer(
            out / "wave-before-after.mp4", fps=25, codec="libx264", quality=8
        ) as writer,
    ):
        for tick in range(0, max(map(len, recordings)) + 25, 4):
            panels = []
            for label, states in zip(("INITIAL", "FINAL"), recordings):
                qpos, seconds, reason = states[min(tick, len(states) - 1)]
                env.data.qpos[:] = qpos
                mujoco.mj_forward(env.model, env.data)
                renderer.update_scene(env.data, camera=camera, scene_option=option)
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 640, 90), fill=(15, 22, 34))
                draw.text(
                    (12, 10),
                    f"v0.0.5 SCREEN {arm} {label} | fixed scenario60000 | t={seconds:.2f}s",
                    fill="white",
                )
                draw.text(
                    (12, 30),
                    "Native gravity | 700 muscles | no evaluation assistance",
                    fill="#78dfc9",
                )
                draw.text(
                    (12, 50),
                    f"{reason or 'running'}"
                    + (" | ENDED, FRAME HELD" if tick >= len(states) else ""),
                    fill="#ffd58a",
                )
                draw.text(
                    (12, 70),
                    "Arm selected by registered screen rule; development clip, not proof",
                    fill="white",
                )
                panels.append(np.asarray(panel))
            writer.append_data(np.concatenate(panels, axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.run.rglob("*"):
        if path.is_file() and (
            path.suffix in {".json", ".jsonl", ".pt", ".npz", ".csv"}
        ):
            target = args.out / path.relative_to(args.run)
            if target.exists() and target.read_bytes() != path.read_bytes():
                raise FileExistsError(f"Refusing to rewrite raw evidence: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    write_json(args.out / "conclusion.json", conclusions(args.run))
    plot(args.run, args.out)
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


if __name__ == "__main__":
    main()
