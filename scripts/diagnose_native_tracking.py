"""Post-hoc first-seed tracking/physical-state diagnostic; no rule override."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from joxing.native_transfer import TransferTask, load_controller, predict
from joxing.provenance import manifest, write_json
from joxing.stance_train import file_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = json.loads((args.run / "protocol.json").read_text())
    seed = p["replay"]["seed"]
    rows = []
    steps = 0
    result = json.loads((args.run / "result.json").read_text())
    audit = json.loads((args.run.parent / "audit-linux.json").read_text())
    render = json.loads((args.run.parent / "report/render.json").read_text())
    spent = (
        result["physics_steps"] + audit["audit_physics_steps"] + render["physics_steps"]
    )
    expected = sum(
        len(np.load(args.run / v["id"] / f"actions-{seed}.npz")["actions"])
        * round(0.02 / v["physics_dt_s"])
        for v in p["variants"]
    )
    if spent + expected > p["caps"]["mission_physics_steps_with_audit_video"]:
        raise RuntimeError("Diagnostic exceeds remaining mission physics reserve")
    deadline = (
        time.perf_counter()
        + p["caps"]["wall_s"]
        - result["wall_s"]
        - audit["audit_wall_s"]
        - render["wall_s"]
    )
    model, n = load_controller(args.run / "inputs")
    for variant in p["variants"]:
        t = TransferTask(p["environment"], variant)
        folder = args.run / variant["id"]
        try:
            t.restore(json.loads((folder / f"starts/{seed}.json").read_text()))
            actions = np.load(folder / f"actions-{seed}.npz")["actions"]
            clips = []
            max_action_error = 0.0
            for a in actions:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Remaining mission diagnostic wall cap")
                old_ref = t.env.qpos_ref.copy()
                actual, clip = predict(model, n, t.last_obs)
                clips.append(clip)
                max_action_error = max(max_action_error, float(np.max(abs(a - actual))))
                _, _, terminated, truncated, _, telemetry = t.step(a)
            errors = abs(t.data.qpos - old_ref)
            top = np.argsort(errors[3:])[::-1][:12] + 3
            coordinates = []
            for j in top:
                joint = int(np.flatnonzero(t.model.jnt_qposadr == j)[0])
                coordinates.append(
                    {
                        "qpos_index": int(j),
                        "joint": t.model.joint(joint).name,
                        "absolute_error": float(errors[j]),
                        "qpos": float(t.data.qpos[j]),
                        "reference_used_for_termination": float(old_ref[j]),
                        "qvel": float(t.data.qvel[j]),
                    }
                )
            velocities = {}
            for name in ("pelvis", "head_neck", "proximal_row_r", "proximal_row_l"):
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    t.model,
                    t.data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    t.model.body(name).id,
                    velocity,
                    0,
                )
                velocities[name] = velocity.tolist()
            rows.append(
                {
                    "variant": variant["id"],
                    "seed": seed,
                    "time_s": float(t.data.time),
                    "native_terminated": terminated,
                    "native_truncated": truncated,
                    "health_threshold": t.env.qpos_diff_th,
                    "mean_qpos_error_at_termination": float(np.mean(errors[3:])),
                    "top_error_coordinates": coordinates,
                    "final_sampled_telemetry": telemetry,
                    "body_world_spatial_velocity_rotation_then_translation": velocities,
                    "mean_clipped_fraction": float(np.mean(clips)),
                    "max_recorded_vs_predicted_action_error": max_action_error,
                    "physics_steps": t.physics_total,
                }
            )
            steps += t.physics_total
        finally:
            t.close()
    n.check()
    write_json(
        args.out,
        manifest()
        | {
            "scope": "Post-hoc first registered seed in all four variants; no continuation beyond native termination, gate change, optimization or causal blame claim.",
            "script_sha256": file_hash(Path(__file__)),
            "physics_steps": steps,
            "findings": rows,
        },
    )
    print("Diagnostic physics steps", steps)


if __name__ == "__main__":
    main()
