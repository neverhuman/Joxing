"""R1: qualify a named seam-velocity variant without stepping physics."""

import argparse
import fcntl
import json
import time
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, manifest, write_json
from joxing.reference_seam import VARIANT, UnwrappedSeamTrajectory
from joxing.stance_train import file_hash
from joxing.upstream_motor import pinned_import
from joxing.wave import check_resources

PROTOCOL_SHA = "1c6f405215cba2e49f18afd083311e39f0a0ad45974352b59ecec0043b5071f9"


def run(protocol, out):
    p = json.loads(protocol.read_text())
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Unregistered seam qualification")
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".reference-seam-r1.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    began = time.monotonic()
    rows = [
        {"cycle": cycle, "interface": mode, "status": "not_run"}
        for cycle in p["cycles"]
        for mode in ("scalar", "batch")
    ]
    report = manifest() | {
        "status": "running",
        "script_sha256": file_hash(Path(__file__)),
        "protocol_sha256": digest(p),
        "variant": VARIANT,
        "cases": rows,
        "new_physics_steps": 0,
        "boxing_games": 0,
    }
    write_json(out / "protocol.json", p)

    def guard():
        if time.monotonic() - began >= p["caps"]["wall_s"]:
            raise TimeoutError("R1 wall cap")
        if (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            >= p["caps"]["raw_bytes"]
        ):
            raise TimeoutError("R1 raw cap")
        check_resources(out, p["caps"])

    try:
        report["initial_capacity"] = check_resources(out, p["caps"], True)
        for path, expected in p["source_files"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError("R1 input/source mismatch: " + path)
        legacy = pinned_import(
            "msgym.envs.imitation_trajectory"
        ).LocomotionCycleTrajectory(str(ROOT / p["motion"]))
        original_velocity = legacy.trajectories[0]["qvel_traj"].copy()
        fixed = UnwrappedSeamTrajectory(legacy)
        assert np.array_equal(original_velocity, legacy.trajectories[0]["qvel_traj"])
        trajectory = legacy.trajectories[0]
        count, dt = trajectory["num_frames"], float(trajectory["dt"])
        duration = float(trajectory["terminate_time"])
        first, last = trajectory["qpos_traj"][[0, -1]].astype(np.float64)
        following = first.copy()
        following[[0, 2]] += last[[0, 2]]
        template_velocity = (following - last) / dt
        assert (
            abs(
                fixed.trajectories[0]["qvel_traj"][-1].astype(np.float64)
                - template_velocity
            ).max()
            <= p["criteria"]["template_velocity_tolerance"]
        )
        report["corrections"] = fixed.corrections
        reset_times = np.arange(int(duration // 0.02)) * 0.02
        old_reset, new_reset = (
            legacy.query_batch(reset_times, 0),
            fixed.query_batch(reset_times, 0),
        )
        assert all(
            np.array_equal(a, b) for a, b in zip(old_reset, new_reset, strict=True)
        )
        report["native_reset_grid"] = {
            "phases_s": reset_times.tolist(),
            "all_qpos_xpos_qvel_exact": True,
        }
        for row in rows:
            guard()
            row["status"] = "running"
            times = row["cycle"] * duration + np.arange(count) * dt
            if row["interface"] == "batch":
                old, new = legacy.query_batch(times, 0), fixed.query_batch(times, 0)
            else:
                old = tuple(
                    np.stack([legacy.query(float(t), 0)[i] for t in times])
                    for i in range(3)
                )
                new = tuple(
                    np.stack([fixed.query(float(t), 0)[i] for t in times])
                    for i in range(3)
                )
            assert np.array_equal(old[0], new[0]) and np.array_equal(old[1], new[1])
            frames = np.clip(
                np.rint(
                    np.mod(times, duration) * float(trajectory["framerate"])
                ).astype(int),
                0,
                count - 1,
            )
            allowed = np.zeros(old[2].shape, dtype=bool)
            allowed[frames == count - 1, 0] = True
            allowed[frames == count - 1, 2] = True
            assert not np.any((old[2] != new[2]) & ~allowed)
            assert np.array_equal(
                new[2][frames != count - 1], old[2][frames != count - 1]
            )
            assert np.allclose(
                new[2][frames == count - 1],
                template_velocity,
                atol=p["criteria"]["template_velocity_tolerance"],
                rtol=0,
            )
            path = out / f"cycle-{row['cycle']}-{row['interface']}.npz"
            np.savez_compressed(
                path,
                times=times,
                frame_indices=frames,
                old_qpos=old[0],
                new_qpos=new[0],
                old_xpos=old[1],
                new_xpos=new[1],
                old_qvel=old[2],
                new_qvel=new[2],
            )
            row.update(
                status="completed",
                samples=count,
                changed_velocity_values=int(np.count_nonzero(old[2] != new[2])),
                all_positions_exact=True,
                all_nonseam_velocities_exact=True,
                raw_sha256=file_hash(path),
                raw_file=path.name,
            )
            write_json(out / "result.json", report)
        guard()
        report.update(
            status="completed",
            reference_gate_passed=True,
            residual_training_allowed=False,
            next_gate="D1 neutral residual interface/full physical source-v8 replay remains required; R1 alone never enables a pilot",
        )
    except Exception as error:  # noqa: BLE001 -- keep every partial and future row
        for row in rows:
            if row["status"] == "running":
                row.update(status="invalid", reason=str(error))
        report.update(
            status="halted",
            error=str(error),
            reference_gate_passed=False,
            residual_training_allowed=False,
        )
    finally:
        report["wall_s"] = time.monotonic() - began
        write_json(out / "result.json", report)
        lease.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments/v0.0.10/reference-seam-protocol.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.out)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ["status", "reference_gate_passed", "error", "wall_s"]
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["status"] == "completed" else 1)
