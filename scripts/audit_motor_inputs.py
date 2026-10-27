"""Preregistered zero-step audit of retained motor inputs and cyclic reference."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, manifest, write_json
from joxing.stance_train import file_hash

PROTOCOL_SHA = "a5dc183dbb271e19054adde27df91ca7ac1daa4685713199e6f566403843057f"


def load_query(path, motion):
    spec = importlib.util.spec_from_file_location("motor_input_trajectory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LocomotionCycleTrajectory(str(motion))


def normalized_statistics(observations, normalizer, slices):
    obs = np.asarray(observations)
    mean, var = np.asarray(normalizer["mean"]), np.asarray(normalizer["var"])
    if obs.ndim != 2 or obs.shape[1] != len(mean) or not np.isfinite(obs).all():
        raise ValueError("Malformed retained observations")
    if mean.shape != var.shape or np.any(var < 0) or normalizer["epsilon"] <= 0:
        raise ValueError("Malformed frozen normalization")
    covered = [i for start, end in slices.values() for i in range(start, end)]
    if sorted(covered) != list(range(obs.shape[1])):
        raise ValueError("Observation slices do not partition the input")
    z = (obs - mean) / np.sqrt(var + normalizer["epsilon"])
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite normalized input")
    clipped = abs(z) > normalizer["clip"]
    return {
        key: {
            "width": end - start,
            "fraction_clipped": float(clipped[:, start:end].mean()),
            "clipped_count_by_control": clipped[:, start:end].sum(axis=1).tolist(),
            "max_abs_unclipped_z_by_control": abs(z[:, start:end]).max(axis=1).tolist(),
        }
        for key, (start, end) in slices.items()
    }


def seam_witness(query):
    """Compare reported seam velocity with the query's own unwrapped positions."""
    t = query.trajectories[0]
    duration, dt = float(t["terminate_time"]), float(t["dt"])
    last_time = (t["num_frames"] - 1) * dt
    last, _, reported = query.query(last_time, 0)
    following, _, _ = query.query(duration, 0)
    assert np.array_equal(last, t["qpos_traj"][-1])
    assert np.array_equal(reported, t["qvel_traj"][-1])
    assert abs(duration - last_time - dt) < 1e-12
    finite_difference = (following.astype(np.float64) - last.astype(np.float64)) / dt
    error = abs(reported.astype(np.float64) - finite_difference)
    reset_times = np.arange(int(duration // 0.02), dtype=np.float64) * 0.02
    reset_frames = np.clip(
        np.rint(reset_times * float(t["framerate"])).astype(np.int64),
        0,
        t["num_frames"] - 1,
    )
    return {
        "last_time_s": last_time,
        "following_time_s": duration,
        "dt_s": dt,
        "last_qpos": last.tolist(),
        "following_qpos": following.tolist(),
        "reported_qvel": reported.tolist(),
        "unwrapped_forward_difference_qvel": finite_difference.tolist(),
        "absolute_error": error.tolist(),
        "max_absolute_error": float(error.max()),
        "max_error_coordinate": int(error.argmax()),
        "reset_grid_frame_indices": reset_frames.tolist(),
        "last_frame_reachable_on_native_reset_grid": bool(
            np.any(reset_frames == t["num_frames"] - 1)
        ),
    }


def cycle_kinematics(query):
    """Scratch FK only; no physics step, live task, pose overwrite or policy."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "third_party/ms-human-700/MS-Human-700.xml")
    )
    data = mujoco.MjData(model)
    duration = float(query.trajectories[0]["terminate_time"])
    errors = []
    frames = query.trajectories[0]["num_frames"]
    dt = float(query.trajectories[0]["dt"])
    for phase in (
        float(i * dt) for i in (0, frames // 4, frames // 2, 3 * frames // 4)
    ):
        q, x, _ = query.query(phase, 0)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        start = data.xpos.copy()
        q1, x1, _ = query.query(duration + phase, 0)
        assert np.array_equal(q[3:], q1[3:])
        data.qpos[:] = q1
        mujoco.mj_kinematics(model, data)
        assert data.xpos.shape == x.shape
        observed_delta = data.xpos - start
        reference_delta = x1.astype(np.float64) - x.astype(np.float64)
        errors.append(
            {
                "phase_s": phase,
                "max_delta_error_m": float(abs(observed_delta - reference_delta).max()),
            }
        )
    return errors


def read_case(folder, packet, seed, specification, normalizer, query):
    if packet == "H1":
        name = f"{seed}-unprofiled"
        receipt = next(
            r
            for r in json.loads((folder / "result.json").read_text())["cases"]
            if r["case_id"] == f"{seed}:unprofiled"
        )
        trace = json.loads((folder / name / "trace.json").read_text())
        initial, final, telemetry = trace["initial"], trace["final"], trace["telemetry"]
        reset_seeds = receipt["resets"]
    else:
        name = f"0-baseline-{seed}"
        result = json.loads((folder / "result.json").read_text())
        receipt = next(
            r
            for w in result["waves"]
            if w["repeat"] == 0 and w["arm"] == "baseline"
            for r in w["cases"]
            if r["seed"] == seed
        )
        initial = json.loads((folder / name / "initial.json").read_text())
        final = json.loads((folder / name / "final.json").read_text())
        telemetry = json.loads((folder / name / "trace.json").read_text())
        reset_seeds = [r["seed"] for r in receipt["resets"]]
    case = folder / name
    assert receipt["status"] == "completed" and len(telemetry) == 300
    assert file_hash(case / "trajectory.npz") == receipt["arrays_sha256"]
    assert file_hash(case / "trace.json") == receipt["trace_sha256"]
    with np.load(case / "trajectory.npz", allow_pickle=False) as raw:
        obs = raw["observations"].copy()
        assert obs.shape == (300, 3601) and obs.dtype == np.float32
    slices = specification["observation_slices"]
    blocks = normalized_statistics(obs, normalizer, slices)
    starts = [0] + [
        i + 1 for i, t in enumerate(telemetry[:-1]) if t["terminated"] or t["truncated"]
    ]
    assert len(starts) == len(reset_seeds)
    duration = float(query.trajectories[0]["terminate_time"])
    phases = [
        float(np.random.default_rng(s).integers(0, int(duration // 0.02)) * 0.02)
        for s in reset_seeds
    ]
    assert phases[0] == initial["application"]["init_time"]
    assert phases[-1] == final["application"]["init_time"]
    root_errors, episodes = [], []
    for episode, start in enumerate(starts):
        end = starts[episode + 1] if episode + 1 < len(starts) else 300
        phase = phases[episode]
        cycle_indices, seam_distances = [], []
        for i in range(start, end):
            before_time = 0.0 if i == start else telemetry[i - 1]["time_s"]
            # Reset's reference is at phase; subsequent cached decisions retain +dt.
            ref_time = phase if i == start else before_time + phase + 0.02
            times = ref_time + np.arange(6, dtype=np.float64) * 0.02
            q, _, _ = query.query_batch(times, 0)
            a, b = slices["qpos_ref"]
            assert np.array_equal(obs[i, a:b], q[0]), (
                name,
                i,
                "cached reference mismatch",
            )
            a, b = slices["qpos_ref_future"]
            assert np.array_equal(obs[i, a:b], q[1:].ravel()), (
                name,
                i,
                "future reference mismatch",
            )
            a, b = slices["qpos"]
            root_errors.append(abs(obs[i, a : a + 3] - q[0, :3]).tolist())
            cycle_indices.append(int(np.floor(ref_time / duration)))
            remainder = ref_time % duration
            seam_distances.append(min(remainder, duration - remainder))
        last = telemetry[end - 1]
        episodes.append(
            {
                "seed": reset_seeds[episode],
                "first_control": start,
                "controls": end - start,
                "initial_reference_phase_s": phase,
                "cycle_index_by_control": cycle_indices,
                "seam_distance_s_by_control": seam_distances,
                "first_tracking_failure_s": last["first_native_tracking_failure_s"],
                "physical_failure_s": last["physical_failure_s"],
                "physical_failure": last["physical_failure"],
                "completed_motor_episode": bool(
                    last["terminated"] or last["truncated"]
                ),
                "ended_by_collection_boundary": not (
                    last["terminated"] or last["truncated"]
                ),
            }
        )
    return {
        "packet": packet,
        "seed": seed,
        "status": "completed",
        "controls": 300,
        "episodes": episodes,
        "normalization": blocks,
        "root_coordinate_absolute_error_by_control": root_errors,
        "max_root_coordinate_error": np.max(root_errors, axis=0).tolist(),
        "cached_reference_exact": True,
        "new_physics_steps": 0,
    }


def run(protocol, h1, h3, out):
    p = json.loads(protocol.read_text())
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Unregistered motor-input audit protocol")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = manifest() | {
        "status": "running",
        "protocol_sha256": digest(p),
        "script_sha256": file_hash(Path(__file__)),
        "cases": [
            {"packet": packet, "seed": seed, "status": "not_run"}
            for packet, seeds in p["seeds"].items()
            for seed in seeds
        ],
        "new_physics_steps": 0,
        "boxing_games": 0,
    }
    write_json(out / "protocol.json", p)
    try:
        from joxing.wave import check_resources

        result["initial_capacity"] = check_resources(out, p["caps"], True)
        for path, expected in p["source_files"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError("Source input changed: " + path)
        for packet, folder in [("H1", h1), ("H3", h3)]:
            for path, expected in p["retained_files"][packet].items():
                if file_hash(folder / path) != expected:
                    raise ValueError("Retained evidence changed: " + path)
        specification = json.loads((ROOT / p["specification"]).read_text())
        normalizer = json.loads((ROOT / p["normalizer"]).read_text())
        assert digest(normalizer) == p["normalizer_stats_hash"]
        query = load_query(ROOT / p["query_source"], ROOT / p["motion"])
        result["seam_witness"] = seam_witness(query)
        result["cycle_kinematics"] = cycle_kinematics(query)
        for row in result["cases"]:
            if time.monotonic() - started >= p["caps"]["wall_s"]:
                raise TimeoutError("Motor input audit wall cap")
            check_resources(out, p["caps"])
            if (
                sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
                >= p["caps"]["raw_bytes"]
            ):
                raise TimeoutError("Motor input audit raw cap")
            row["status"] = "running"
            row.update(
                read_case(
                    h1 if row["packet"] == "H1" else h3,
                    row["packet"],
                    row["seed"],
                    specification,
                    normalizer,
                    query,
                )
            )
            write_json(out / "result.json", result)
        # A mathematical witness is not evidence that the seam caused the falls.
        result["reference_inconsistencies"] = []
        if (
            result["seam_witness"]["max_absolute_error"]
            > p["criteria"]["velocity_consistency_tolerance"]
        ):
            result["reference_inconsistencies"].append(
                "reported seam velocity disagrees with consecutive unwrapped reference positions"
            )
        if (
            max(x["max_delta_error_m"] for x in result["cycle_kinematics"])
            > p["criteria"]["translation_tolerance_m"]
        ):
            result["reference_inconsistencies"].append(
                "cyclic qpos and world-position reference translations disagree"
            )
        result.update(
            status="completed",
            residual_training_allowed=not result["reference_inconsistencies"],
            scope="Offline retained-input/cycle reference audit. Scratch FK, zero mj_step and no actor/optimizer execution. A continuity witness does not establish the cause of physical collapse.",
        )
    except Exception as error:  # noqa: BLE001 -- all partial attempts and future rows retained
        for row in result["cases"]:
            if row["status"] == "running":
                row.update(status="invalid", reason=str(error))
        result.update(
            status="halted", error=str(error), residual_training_allowed=False
        )
    finally:
        result["wall_s"] = time.monotonic() - started
        write_json(out / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments/v0.0.10/input-audit-protocol.json",
    )
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--h3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.h1, args.h3, args.out)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in [
                    "status",
                    "error",
                    "reference_inconsistencies",
                    "residual_training_allowed",
                    "wall_s",
                ]
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["status"] == "completed" else 1)
