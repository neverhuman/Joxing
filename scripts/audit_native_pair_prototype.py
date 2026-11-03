"""Independent native PPO/decoder witness audit, including640counted replay steps."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    a = ap.parse_args()
    p = json.loads((ROOT / a.packet).read_text())
    out = ROOT / p["output"]
    native = out / "native"
    import mujoco
    import numpy as np
    import torch
    from source_shard_support import check_junit

    from joxing.native_pair_motor_rl import NativePairMotorEnv, residual_command
    from joxing.native_pair_motor_train import AuditedPPO, actor_hash
    from joxing.native_tape_storage import load
    from joxing.native_transfer import predict
    from joxing.provenance import digest

    torch.set_num_threads(1)
    result = {
        "schema": "joxing-native-pair-independent-prototype-audit-v1",
        "passed": False,
        "audit_physics": 0,
    }
    env = None
    try:
        source = check_junit((out / "source.xml").read_text(), p["test_node_ids"], [])
        if not source["qualified"]:
            raise ValueError("Complete registered source prerequisite")
        r = json.loads((native / "result.json").read_text())
        credit = json.loads((out / "physics-final.json").read_text())
        replay = json.loads((native / "native-replay.json").read_text())
        if (
            r["status"] != "completed"
            or [x["status"] for x in r["rows"]] != ["completed"] * 3
            or r["native_physics"] != credit["entered_physics"]
            or credit["entered_physics"] + p["source_fixture_physics_reserve"]
            > p["physics_steps"]
        ):
            raise ValueError("Native result/physics/denominator")
        if (
            not replay["passed"]
            or replay["sample_tick"] != 203
            or replay["remaining"] != 17
            or replay["source"] != replay["replayed"]
        ):
            raise ValueError("Actual pretrained replay proof")
        arrays = load(
            native / "training/native-transitions.npz", r["training"]["transition_tape"]
        )
        rows = [
            json.loads(x)
            for x in (native / "training/transitions.jsonl").read_text().splitlines()
        ]
        chain = "0" * 64
        if (
            len(rows) != 32
            or arrays["source_obs"].shape != (32, 2, 7901)
            or arrays["applied_actions"].shape != (32, 2, 138)
            or arrays["commands"].shape != (32, 2, 700)
            or r["training"]["actor_samples"] != 64
            or r["training"]["rollout_updates"] != 4
        ):
            raise ValueError("Actual paired transition/actor sample shape")
        env = NativePairMotorEnv.from_inputs(Path(p["teacher_inputs"]))
        scratch = mujoco.MjData(env.scene.model)
        for i, row in enumerate(rows):
            value = dict(row)
            saved = value.pop("chain")
            chain = digest({"previous": chain, "row": value})
            if saved != chain:
                raise ValueError("Native transition hash chain")
            for j, side in enumerate("AB"):
                for field, key in [
                    ("source_obs", "source_obs_sha256"),
                    ("next_obs", "next_obs_sha256"),
                    ("applied_actions", "applied_actions_sha256"),
                    ("commands", "commands_sha256"),
                ]:
                    if (
                        hashlib.sha256(arrays[field][i, j].tobytes()).hexdigest()
                        != row[key][side]
                    ):
                        raise ValueError("Full native transition array/hash:" + field)
                teacher, _ = predict(
                    env.teacher.model,
                    env.teacher.normalizer,
                    arrays["teacher_inputs"][i, j],
                )
                if (
                    teacher.tobytes() != arrays["teacher_commands"][i, j].tobytes()
                    or residual_command(
                        teacher, arrays["applied_actions"][i, j], env.groups
                    ).tobytes()
                    != arrays["commands"][i, j].tobytes()
                ):
                    raise ValueError(
                        "Frozen teacher/all700muscle decoder reconstruction"
                    )
            if not np.array_equal(
                np.clip(arrays["raw_policy_actions"][i], -1, 1),
                arrays["applied_actions"][i],
            ):
                raise ValueError("Policy sample/applied action mismatch")
            from joxing.env import INTEGRATION, excitation

            mujoco.mj_setState(
                env.scene.model, scratch, arrays["engine_before"][i], INTEGRATION
            )
            before = float(scratch.time)
            for j, side in enumerate("AB"):
                b = env.bodies[side]
                if not np.array_equal(
                    scratch.qpos[b.qpos_ids].astype(np.float32),
                    arrays["teacher_inputs"][i, j, :85],
                ) or not np.array_equal(
                    scratch.qvel[b.qvel_ids].astype(np.float32),
                    arrays["teacher_inputs"][i, j, 85:170],
                ):
                    raise ValueError("Native engine/own teacher state projection")
            mujoco.mj_setState(
                env.scene.model, scratch, arrays["engine_after"][i], INTEGRATION
            )
            if abs(float(scratch.time) - before - 0.02) > 1e-9:
                raise ValueError("Physical20ms joint transition")
            for j, side in enumerate("AB"):
                if not np.array_equal(
                    scratch.ctrl[env.bodies[side].actuator_ids],
                    excitation(arrays["commands"][i, j]),
                ):
                    raise ValueError("Actual native1400muscle excitation binding")
            # Independently replay every actual joint transition from saved full
            # integration state and recorded applied muscles; account BEFOREstep.
            from joxing.physics import step_contacts

            mujoco.mj_setState(
                env.scene.model, scratch, arrays["engine_before"][i], INTEGRATION
            )
            for j, side in enumerate("AB"):
                scratch.ctrl[env.bodies[side].actuator_ids] = excitation(
                    arrays["commands"][i, j]
                )
            for _ in range(20):
                if (
                    credit["entered_physics"]
                    + result["audit_physics"]
                    + 1
                    + p["source_fixture_physics_reserve"]
                    > p["physics_steps"]
                ):
                    raise RuntimeError("Independent replay physics first limit")
                result["audit_physics"] += 1
                step_contacts(env.scene.model, scratch)
            replayed = np.empty(mujoco.mj_stateSize(env.scene.model, INTEGRATION))
            mujoco.mj_getState(env.scene.model, scratch, replayed, INTEGRATION)
            if replayed.tobytes() != arrays["engine_after"][i].tobytes():
                raise ValueError(
                    "Independent actual20step native integration replay differs"
                )

        if chain != r["training"]["chain"]:
            raise ValueError("Final transition chain")
        initial = AuditedPPO.load(native / "training/initial-policy.zip", device="cpu")
        final = AuditedPPO.load(native / "training/final-policy.zip", device="cpu")
        if (
            actor_hash(initial.policy) != r["training"]["actor_before"]
            or actor_hash(final.policy) != r["training"]["actor_after"]
            or actor_hash(initial.policy) == actor_hash(final.policy)
        ):
            raise ValueError("Independent actor parameter change")
        updates = []
        for k in range(4):
            model = AuditedPPO.load(
                native / f"training/update-{k:03d}-before.zip", device="cpu"
            )
            errors = []
            for i in range(k * 8, (k + 1) * 8):
                with torch.no_grad():
                    values, logprob, _ = model.policy.evaluate_actions(
                        torch.as_tensor(arrays["source_obs"][i]),
                        torch.as_tensor(arrays["raw_policy_actions"][i]),
                    )
                err = max(
                    float(np.max(abs(values.numpy() - arrays["values"][i]))),
                    float(np.max(abs(logprob.numpy() - arrays["log_probs"][i]))),
                )
                errors.append(err)
                if err > p["policy_recompute_atol"]:
                    raise ValueError("Behavior value/logprob replay")
            after = AuditedPPO.load(
                native / f"training/update-{k:03d}-after.zip", device="cpu"
            )
            if actor_hash(model.policy) == actor_hash(after.policy):
                raise ValueError("No actor update in registered rollout")
            updates.append(
                {
                    "number": k,
                    "max_value_logprob_error": max(errors),
                    "actor_changed": True,
                }
            )
        evals = {}
        for label in ("before", "after"):
            e = json.loads((native / f"{label}-eval/evaluation.json").read_text())
            frames = load(native / f"{label}-eval/frames.npz", e["frame_tape"])
            video = json.loads(
                (native / f"{label}-eval/{label}-video.json").read_text()
            )
            blob = (native / f"{label}-eval/{label}.mp4").read_bytes()
            if (
                len(frames["time"]) != e["frames"]
                or np.any(np.diff(frames["time"]) <= 0)
                or video["frames"] != e["frames"]
                or video["render_physics"] != 0
                or len(blob) != video["bytes"]
                or hashlib.sha256(blob).hexdigest() != video["sha256"]
            ):
                raise ValueError("Fixed actual frame/clip identity")
            evals[label] = {
                "final_tick": e["final_tick"],
                "frames": e["frames"],
                "candidate_points": e["candidate_points"],
                "motor_failures": e["motor_failures"],
                "returns": e["returns"],
                "video_sha256": video["sha256"],
            }
        result.update(
            passed=True,
            source_counts=source["counts"],
            actual_native_joint_transitions=32,
            actor_samples=64,
            rollout_updates=4,
            optimizer_epochs=r["training"]["optimizer_epochs"],
            native_physics=credit["entered_physics"],
            source_physics_reserved_not_measured=p["source_fixture_physics_reserve"],
            replay_passed=True,
            full700decoder_reconstruction=True,
            updates=updates,
            evaluations=evals,
            teacher_bundle=env.bundle,
            boxing_qualified=False,
            performance_improvement_claimed=False,
            platform_scope="Local Mac prototype only; not pinned Linux parity or100kboxinggames",
        )
    except Exception as error:  # noqa: BLE001 -- retain independent faults
        result.update(error=type(error).__name__ + ": " + str(error))
    finally:
        if env is not None:
            env.close()
        (out / "independent-audit.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "passed",
                    "error",
                    "actual_native_joint_transitions",
                    "actor_samples",
                    "native_physics",
                )
            }
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
