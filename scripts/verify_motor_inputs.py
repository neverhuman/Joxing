"""Independent arithmetic and all-seven-case audit of the frozen D0 output."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, write_json
from joxing.stance_train import file_hash


def verify(folder, h1, h3, receipts):
    r = json.loads((folder / "result.json").read_text())
    p = json.loads((folder / "protocol.json").read_text())
    assert (
        digest(p)
        == r["protocol_sha256"]
        == "a5dc183dbb271e19054adde27df91ca7ac1daa4685713199e6f566403843057f"
    )
    executable = subprocess.check_output(
        ["git", "show", r["code_commit"] + ":scripts/audit_motor_inputs.py"], cwd=ROOT
    )
    assert hashlib.sha256(executable).hexdigest() == r["script_sha256"]
    assert (
        r["status"] == "completed" and r["new_physics_steps"] == r["boxing_games"] == 0
    )
    assert r["wall_s"] < p["caps"]["wall_s"]
    for path, expected in p["source_files"].items():
        assert file_hash(ROOT / path) == expected
    for packet, root in [("H1", h1), ("H3", h3)]:
        for path, expected in p["retained_files"][packet].items():
            assert file_hash(root / path) == expected
    n = json.loads((ROOT / p["normalizer"]).read_text())
    assert digest(n) == p["normalizer_stats_hash"]
    assert n["norm_obs"] and not n["training"] and not n["norm_reward"]
    spec = json.loads((ROOT / p["specification"]).read_text())
    with np.load(ROOT / p["motion"], allow_pickle=False) as source:
        q = source["qpos_traj"].astype(np.float32)
        fps = float(source["framerate"])
    count, dt = len(q), 1 / fps
    duration = count / fps

    def direct_position(t):
        cycle = int(np.floor(t / duration))
        frame = int(np.clip(np.rint((t % duration) * fps), 0, count - 1))
        position = q[frame].copy()
        for j in (0, 2):
            position[j] += np.float32(cycle) * q[-1, j]
        return position

    witness = r["seam_witness"]
    following = q[0].copy()
    following[0] += q[-1, 0]
    following[2] += q[-1, 2]
    expected_velocity = (following.astype(np.float64) - q[-1].astype(np.float64)) / dt
    reported = ((q[0] - q[-1]).astype(np.float64) / dt).astype(np.float32)
    assert np.array_equal(witness["last_qpos"], q[-1])
    assert np.array_equal(witness["following_qpos"], following)
    assert np.array_equal(witness["reported_qvel"], reported)
    assert np.array_equal(
        witness["unwrapped_forward_difference_qvel"], expected_velocity
    )
    error = abs(reported.astype(np.float64) - expected_velocity)
    assert witness["max_absolute_error"] == error.max()
    assert witness["max_error_coordinate"] == error.argmax()
    reset_frames = np.rint(np.arange(int(duration // 0.02)) * 0.02 * fps).astype(int)
    assert witness["reset_grid_frame_indices"] == reset_frames.tolist()
    assert witness["last_frame_reachable_on_native_reset_grid"] == bool(
        np.any(reset_frames == count - 1)
    )
    assert (
        max(x["max_delta_error_m"] for x in r["cycle_kinematics"])
        < p["criteria"]["translation_tolerance_m"]
    )
    assert error.max() > p["criteria"]["velocity_consistency_tolerance"]
    assert not r["residual_training_allowed"] and r["reference_inconsistencies"] == [
        "reported seam velocity disagrees with consecutive unwrapped reference positions"
    ]
    expected_cases = [
        (packet, seed) for packet, seeds in p["seeds"].items() for seed in seeds
    ]
    assert [(c["packet"], c["seed"]) for c in r["cases"]] == expected_cases
    findings = []
    for c in r["cases"]:
        assert c["status"] == "completed" and c["controls"] == 300
        root = h1 if c["packet"] == "H1" else h3
        name = (
            f"{c['seed']}-unprofiled"
            if c["packet"] == "H1"
            else f"0-baseline-{c['seed']}"
        )
        payload = json.loads((root / name / "trace.json").read_text())
        trace = payload["telemetry"] if isinstance(payload, dict) else payload
        with np.load(root / name / "trajectory.npz", allow_pickle=False) as tape:
            obs = tape["observations"]
            assert obs.shape == (300, 3601) and np.isfinite(obs).all()
            means, variances = np.asarray(n["mean"]), np.asarray(n["var"])
            for key, (start, end) in spec["observation_slices"].items():
                # Compute independently one observation block at a time.
                values = (obs[:, start:end] - means[start:end]) / np.sqrt(
                    variances[start:end] + n["epsilon"]
                )
                clipped = (values < -n["clip"]) | (values > n["clip"])
                block = c["normalization"][key]
                assert block["clipped_count_by_control"] == clipped.sum(axis=1).tolist()
                assert (
                    block["fraction_clipped"]
                    == np.count_nonzero(clipped) / clipped.size
                )
                assert np.allclose(
                    block["max_abs_unclipped_z_by_control"],
                    abs(values).max(axis=1),
                    atol=1e-12,
                    rtol=1e-12,
                )
            starts = [0] + [
                i + 1
                for i, t in enumerate(trace[:-1])
                if t["terminated"] or t["truncated"]
            ]
            assert starts == [e["first_control"] for e in c["episodes"]]
            root_errors = []
            for e in c["episodes"]:
                phase = float(
                    np.random.default_rng(e["seed"]).integers(0, int(duration // 0.02))
                    * 0.02
                )
                assert e["initial_reference_phase_s"] == phase
                for i in range(e["first_control"], e["first_control"] + e["controls"]):
                    t = (
                        phase
                        if i == e["first_control"]
                        else trace[i - 1]["time_s"] + phase + 0.02
                    )
                    for k in range(6):
                        expected = direct_position(t + k * 0.02)
                        start = (
                            spec["observation_slices"]["qpos_ref"][0]
                            if k == 0
                            else spec["observation_slices"]["qpos_ref_future"][0]
                            + 85 * (k - 1)
                        )
                        assert np.array_equal(obs[i, start : start + 85], expected)
                        if k == 0:
                            root_errors.append(abs(obs[i, :3] - expected[:3]))
                last = trace[e["first_control"] + e["controls"] - 1]
                assert e["physical_failure_s"] == last["physical_failure_s"]
                assert (
                    e["first_tracking_failure_s"]
                    == last["first_native_tracking_failure_s"]
                )
            assert np.array_equal(
                c["root_coordinate_absolute_error_by_control"], root_errors
            )
            assert np.array_equal(
                c["max_root_coordinate_error"], np.max(root_errors, axis=0)
            )
        findings.append(
            {
                "packet": c["packet"],
                "seed": c["seed"],
                "controls": 300,
                "episodes": len(c["episodes"]),
                "raw_statistics_and_cached_references_verified": True,
            }
        )
    exited = json.loads((receipts / "process-exit.json").read_text())
    assert not exited["owned_processes"] and not exited["source_diff"]
    assert all(not x["exists"] for x in exited["recorded_processes"])
    supervisor = json.loads((receipts / "supervisor.json").read_text())
    assert supervisor["status"] == "completed" and all(
        x["returncode"] == 0 for x in supervisor["stages"]
    )
    assert "2 passed" in (receipts / "tests.txt").read_text()
    return {
        "passed": True,
        "source_commit": r["code_commit"],
        "auditor_sha256": file_hash(Path(__file__)),
        "scope": "Independent seam arithmetic from raw motion NPZ, reset-grid exposure, all7raw normalization blocks/cached goal coordinates/reset/failure records and receipts. Source-pinned scratch FK checks read from frozen output; no second FK/actor/physics execution.",
        "case_counts": dict(Counter(c["status"] for c in r["cases"])),
        "new_audit_physics_steps": 0,
        "reference_velocity_error": float(error.max()),
        "reference_error_coordinate": int(error.argmax()),
        "seam_reachable_on_reset_grid": bool(np.any(reset_frames == count - 1)),
        "residual_training_allowed": False,
        "causal_contribution_to_falls": "unproven; seam frame excluded by native reset grid",
        "findings": findings,
        "result_sha256": file_hash(folder / "result.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("folder", "h1", "h3", "receipts"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.folder, args.h1, args.h3, args.receipts)
    write_json(args.folder / "audit.json", result)
    print(json.dumps(result, indent=2))
