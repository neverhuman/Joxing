from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from .physics import momentum_fixture, scoring_fixture
from .provenance import manifest, write_json
from .scene import build_scene, save_scene


def benchmark(out: Path, steps: int, repeats: int = 3):
    scene = build_scene()
    model = scene.model
    data = mujoco.MjData(model)
    samples = []
    for repeat in range(repeats):
        warnings = 0
        contacts = 0
        started = time.perf_counter()
        for k in range(steps):
            if k % 50 == 0:
                mujoco.mj_resetData(model, data)
                data.qpos[:] = scene.report["initial_qpos"]
                data.ctrl[:] = 0.02
            mujoco.mj_step(model, data)
            contacts += data.ncon
            warnings += sum(w.number for w in data.warning)
            if not np.isfinite(data.qpos).all() or warnings:
                raise RuntimeError("Invalid benchmark; preserve failure before scaling")
        elapsed = time.perf_counter() - started
        samples.append(
            {
                "repeat": repeat,
                "steps": steps,
                "wall_s": elapsed,
                "physics_substeps_per_s": steps / elapsed,
                "contact_samples": contacts,
            }
        )
    result = manifest() | {
        "scene_hash": scene.report["scene_hash"],
        "profile": "short-reset-composition-throughput",
        "physics_dt_s": model.opt.timestep,
        "dimensions": scene.report["dimensions"],
        "samples": samples,
        "median_physics_substeps_per_s": float(
            np.median([x["physics_substeps_per_s"] for x in samples])
        ),
        "limitations": [
            "simulator-only",
            "one world",
            "reset every 50 ms",
            "not contact-heavy boxing",
            "no inference, referee or optimizer",
        ],
    }
    write_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Joxing reference diagnostics; no production self-play command yet"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--out", type=Path, default=Path("artifacts/generated/scene"))
    fixture = sub.add_parser("fixture")
    fixture.add_argument(
        "--out", type=Path, default=Path("artifacts/generated/fixture.json")
    )
    bench = sub.add_parser("benchmark")
    bench.add_argument(
        "--out", type=Path, default=Path("artifacts/generated/benchmark.json")
    )
    bench.add_argument("--steps", type=int, default=500)
    train = sub.add_parser("train-reach")
    train.add_argument("--out", type=Path, default=Path("runs/reach"))
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--generations", type=int, default=16)
    train.add_argument("--population", type=int, default=24)
    train.add_argument("--seed", type=int, default=7)
    for name in ("strike-pilot", "train-strike"):
        strike = sub.add_parser(name)
        strike.add_argument("--out", type=Path, required=True)
        strike.add_argument("--workers", type=int, default=4)
        strike.add_argument(
            "--generations", type=int, default=8 if name == "strike-pilot" else 12
        )
        strike.add_argument(
            "--population", type=int, default=16 if name == "strike-pilot" else 24
        )
    balance = sub.add_parser(
        "balance-probe", help="v0.0.4 native-gravity reset/root audit"
    )
    balance.add_argument("--out", type=Path, required=True)
    stance = sub.add_parser(
        "train-stance",
        help="v0.0.4 bounded SAC development pilot, not formal motor proof",
    )
    stance.add_argument("--out", type=Path, required=True)
    stance.add_argument("--steps", type=int, default=100000)
    stance.add_argument("--workers", type=int, default=4)
    stance.add_argument("--seed", type=int, default=73)
    stance.add_argument("--wall-s", type=float, default=1800.0)
    stance.add_argument("--eval-count", type=int, default=32)
    render = sub.add_parser("render")
    render.add_argument(
        "--out", type=Path, default=Path("artifacts/generated/scene.png")
    )
    args = parser.parse_args()
    if args.command == "inspect":
        scene = build_scene()
        save_scene(scene, args.out)
        result = scene.report
    elif args.command == "fixture":
        result = manifest() | {
            "fixtures": [
                momentum_fixture(angle=angle, reverse=rev)
                for angle in (0.0, 0.7)
                for rev in (False, True)
            ],
            "physical_scoring_fixtures": [
                scoring_fixture(zone) for zone in ("head", "torso", "guard")
            ],
        }
        write_json(args.out, result)
    elif args.command == "benchmark":
        if args.steps < 1:
            parser.error("--steps must be positive")
        result = benchmark(args.out, args.steps)
    elif args.command == "train-reach":
        from .motor import ReachConfig, train_reach

        result = train_reach(
            ReachConfig(
                workers=args.workers,
                generations=args.generations,
                population=args.population,
                elite=max(2, args.population // 4),
                seed=args.seed,
            ),
            args.out,
        )
    elif args.command in ("strike-pilot", "train-strike"):
        from .strike import train_strike_suite

        pilot = args.command == "strike-pilot"
        result = train_strike_suite(
            args.out,
            workers=args.workers,
            generations=args.generations,
            population=args.population,
            seeds=(7,) if pilot else (7, 19, 41),
            eval_count=8 if pilot else 64,
            pilot=pilot,
        )
    elif args.command == "balance-probe":
        from .stance_train import balance_probe

        result = balance_probe(args.out)
    elif args.command == "train-stance":
        from .stance_train import train_stance

        result = train_stance(
            args.out,
            steps=args.steps,
            workers=args.workers,
            seed=args.seed,
            wall_s=args.wall_s,
            eval_count=args.eval_count,
        )
    else:
        import imageio.v3 as iio

        scene = build_scene()
        data = mujoco.MjData(scene.model)
        data.qpos[:] = scene.report["initial_qpos"]
        mujoco.mj_forward(scene.model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 1.0]
        camera.distance = 3.5
        camera.azimuth = 100
        camera.elevation = -15
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        with mujoco.Renderer(scene.model, height=720, width=1280) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(args.out, renderer.render())
        result = {
            "image": str(args.out),
            "caption": "Two intact MS-Human-700 models; static composition inspection, no trained boxing",
        }
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "fixtures",
                    "traces",
                    "physical_geoms",
                    "actuator_ids",
                    "initial_qpos",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
