"""Independent H31 audit: all training integrations and all fixed evaluations."""

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--input-root")
    ap.add_argument("--report-root")
    a = ap.parse_args()
    p = json.loads((ROOT / a.packet).read_text())
    out = Path(a.input_root) if a.input_root else ROOT / p["output"]
    report = Path(a.report_root) if a.report_root else out
    native = out / "native"
    # Post-task replay imports the ORIGINAL frozen physical modules so even
    # any absolute asset paths in compiled scene identities remain unchanged.
    if a.input_root:
        sys.path.insert(0, str(out.parents[2] / "src"))
    import mujoco
    import numpy as np
    import torch
    from source_shard_support import check_junit

    from joxing.env import INTEGRATION, excitation
    from joxing.native_pair_motor_evidence import EvidenceBudget, read_chunk
    from joxing.native_pair_motor_pilot import MotorMetrics, PilotAuditedPPO
    from joxing.native_pair_motor_rl import (
        NativePairMotorEnv,
        PairMotorConfig,
        residual_command,
    )
    from joxing.native_pair_motor_train import actor_hash
    from joxing.native_tape_storage import load
    from joxing.native_transfer import predict
    from joxing.physics import step_contacts

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    budget = EvidenceBudget(
        report, raw_limit=p["raw_bytes"], text_limit=p["receipt_bytes"]
    )
    primary = json.loads((out / "physics-final.json").read_text())["entered_physics"]
    started = time.monotonic()
    entered = 0
    world = None
    ledger = (report / "audit-entered.bin").open("xb+", buffering=0)
    result = {
        "schema": "joxing-native-pair-motor-pilot-audit-v1",
        "passed": False,
        "seed_rows": [],
        "physics": 0,
    }

    def guard(n):
        nonlocal entered
        if (
            primary + entered + n + p["source_fixture_physics_reserve"]
            > p["physics_steps"]
        ):
            raise RuntimeError("Audit entered physics first limit")
        if time.monotonic() - started > p["audit_wall_s"] - 5:
            raise TimeoutError("Audit original deadline")
        if n:
            entered += n
            ledger.seek(0)
            ledger.write(struct.pack("<QQ", entered, entered ^ ((1 << 64) - 1)))

    try:
        source = check_junit((out / "source.xml").read_text(), p["test_node_ids"], [])
        if not source["qualified"]:
            raise ValueError("Complete source prerequisite")
        r = json.loads((native / "result.json").read_text())
        if (
            r["status"] != "completed"
            or [s["status"] for s in r["seed_rows"]] != ["completed"] * 2
        ):
            raise ValueError("Native attempted seed denominator")
        world = NativePairMotorEnv.from_inputs(
            Path(p["teacher_inputs"]),
            config=PairMotorConfig(horizon_ticks=3000),
            guard=guard,
        )
        if world.contract_hash != r["contract_hash"]:
            raise ValueError("World/teacher/profile identity")
        scratch = mujoco.MjData(world.scene.model)
        for seed in r["seed_rows"]:
            folder = native / f"seed-{seed['seed']}"
            train = seed["training"]
            if (
                train["joint_transitions"] != 1024
                or train["actor_samples"] != 2048
                or len(train["updates"]) != 8
                or len(train["checkpoints"]) != 9
            ):
                raise ValueError("Actual pilot training counts")
            from joxing.native_tape_storage import sha_file

            for checkpoint in train["checkpoints"]:
                for suffix, key, maximum in (
                    ("zip", "policy", 13 * 1024**2),
                    ("pt", "continuation", 1024**2),
                ):
                    path = (
                        folder
                        / "checkpoints"
                        / f"boundary-{checkpoint['number']:03d}.{suffix}"
                    )
                    if (
                        path.stat().st_size > maximum
                        or path.stat().st_size != checkpoint[key]["bytes"]
                        or sha_file(path) != checkpoint[key]["sha256"]
                    ):
                        raise ValueError(
                            "Full checkpoint/continuation byte identity/bound"
                        )
            cursor = None
            chain = "0" * 64
            maxerr = 0.0
            count = 0
            for k, chunk in enumerate(train["chunks"]):
                rows, cursor, chain = read_chunk(
                    folder / "training" / chunk["path"],
                    expected_sha256=chunk["sha256"],
                    contract_hash=world.contract_hash,
                    cursor=cursor,
                    previous_chain=chain,
                )
                behavior = train["checkpoints"][k]
                policy_path = folder / "checkpoints" / f"boundary-{k:03d}.zip"
                if sha_file(policy_path) != behavior["policy"]["sha256"]:
                    raise ValueError("Behavior checkpoint bytes")
                model = PilotAuditedPPO.load(policy_path, device="cpu")
                if actor_hash(model.policy) != behavior["actor_hash"]:
                    raise ValueError("Behavior actor identity")
                for row in rows:
                    identity = row["metadata"]["info"]["behavior"]
                    if identity != {
                        "boundary": k,
                        "policy_sha256": behavior["policy"]["sha256"],
                        "actor_hash": behavior["actor_hash"],
                    }:
                        raise ValueError("Row behavior identity")
                    with torch.no_grad():
                        value, logprob, _ = model.policy.evaluate_actions(
                            torch.as_tensor(row["source_obs"]),
                            torch.as_tensor(row["raw_policy_actions"]),
                        )
                    error = max(
                        float(np.max(abs(value.numpy() - row["values"]))),
                        float(np.max(abs(logprob.numpy() - row["log_probs"]))),
                    )
                    maxerr = max(error, maxerr)
                    if error > p["policy_recompute_atol"]:
                        raise ValueError("PPO behavior recomputation")
                    for j, side in enumerate("AB"):
                        teacher, _ = predict(
                            world.teacher.model,
                            world.teacher.normalizer,
                            row["teacher_inputs"][j],
                        )
                        if (
                            teacher.tobytes() != row["teacher_commands"][j].tobytes()
                            or residual_command(
                                teacher, row["applied_actions"][j], world.groups
                            ).tobytes()
                            != row["commands"][j].tobytes()
                        ):
                            raise ValueError(
                                "Actual released teacher/all700muscle reconstruction"
                            )
                    mujoco.mj_setState(
                        world.scene.model, scratch, row["engine_before"], INTEGRATION
                    )
                    for j, side in enumerate("AB"):
                        body = world.bodies[side]
                        if (
                            scratch.qpos[body.qpos_ids].astype(np.float32).tobytes()
                            != row["teacher_inputs"][j, :85].tobytes()
                            or scratch.qvel[body.qvel_ids].astype(np.float32).tobytes()
                            != row["teacher_inputs"][j, 85:170].tobytes()
                        ):
                            raise ValueError("Native teacher/own state binding")
                        scratch.ctrl[body.actuator_ids] = excitation(row["commands"][j])
                    for _ in range(20):
                        guard(1)
                        step_contacts(world.scene.model, scratch)
                    state = np.empty(len(row["engine_after"]))
                    mujoco.mj_getState(world.scene.model, scratch, state, INTEGRATION)
                    if state.tobytes() != row["engine_after"].tobytes():
                        raise ValueError("Independent native training replay")
                    count += 1
                del model
            if count != 1024 or chain != train["chain"]:
                raise ValueError("Complete training hash chain")
            final = train["checkpoints"][-1]
            if train["checkpoints"][0]["actor_hash"] == final["actor_hash"]:
                raise ValueError("No changed learner")
            if (
                sum(x["optimizer_minibatch_steps"] for x in train["updates"]) != 128
                or train["optimizer_epochs"] != 32
            ):
                raise ValueError("Actual optimizer minibatch/epoch counts")
            model = PilotAuditedPPO.load(
                folder / "checkpoints/boundary-008.zip", device="cpu"
            )
            if actor_hash(model.policy) != final["actor_hash"]:
                raise ValueError("Final frozen actor identity")
            evaluations = []
            for phase in ("before", "after"):
                for item in seed["evaluations"][phase]:
                    if item["status"] != "completed":
                        raise ValueError("Evaluation denominator")
                    f = folder / f"{phase}-{item['seed']}"
                    e = item["result"]
                    trace = load(f / "evaluation-trace.npz", e["trace_tape"])
                    frames = load(f / "frames.npz", e["frame_tape"])
                    snap = json.loads((f / "initial-snapshot.json").read_text())
                    if snap["sha256"] != e["initial_snapshot_sha256"]:
                        raise ValueError("Evaluation start snapshot identity")
                    world.reset(seed=item["seed"])
                    world.restore(snap)
                    if len(frames["time"]) != len(trace["actions"]) + 1 or len(
                        trace["metrics_1ms"]
                    ) != 20 * len(trace["actions"]):
                        raise ValueError("Complete frame/physics trace")
                    metrics = MotorMetrics()
                    for i, actual in enumerate(trace["actions"]):
                        obs = world.observations()
                        expected = np.stack(
                            [
                                np.zeros(138, np.float32)
                                if phase == "before"
                                else model.predict(obs[s], deterministic=True)[0]
                                for s in "AB"
                            ]
                        )
                        if expected.tobytes() != actual.tobytes():
                            raise ValueError("Frozen evaluation policy/action replay")
                        d = world.core.data
                        if float(frames["time"][i]) != float(d.time):
                            raise ValueError("Fixed frame time")
                        for field, v in (
                            ("qpos", d.qpos),
                            ("qvel", d.qvel),
                            ("activation", d.act),
                            ("control", d.ctrl),
                        ):
                            if frames[field][i].tobytes() != v.tobytes():
                                raise ValueError("Actual fixed frame binding")
                        world.begin_control({s: actual[j] for j, s in enumerate("AB")})
                        if (
                            np.stack([world.core.commands[s] for s in "AB"]).tobytes()
                            != trace["commands"][i].tobytes()
                        ):
                            raise ValueError("Evaluation muscles")
                        for _ in range(20):
                            world.advance_substep()
                            metrics.observe(world)
                        if (
                            hashlib.sha256(world.core.state().tobytes()).digest()
                            != trace["engine_sha256"][i].tobytes()
                        ):
                            raise ValueError("Exact full evaluation native replay")
                    d = world.core.data
                    for field, v in (
                        ("qpos", d.qpos),
                        ("qvel", d.qvel),
                        ("activation", d.act),
                        ("control", d.ctrl),
                    ):
                        if frames[field][-1].tobytes() != v.tobytes():
                            raise ValueError("Final actual fixed frame binding")
                    if float(frames["time"][-1]) != float(d.time):
                        raise ValueError("Final frame time")
                    if (
                        not world.done
                        or world.core.tick != e["final_tick"]
                        or world.failures != e["motor_failures"]
                        or list(world.referee.points) != list(e["candidate_points"])
                    ):
                        raise ValueError("Fixed evaluation physical end/points/falls")
                    if (
                        np.array(metrics.rows, np.float64).tobytes()
                        != trace["metrics_1ms"].tobytes()
                        or metrics.summary(world) != e["metrics"]
                    ):
                        raise ValueError("Continuous1msbehavior measurement")
                    evaluations.append(
                        {
                            "phase": phase,
                            "seed": item["seed"],
                            "metrics": e["metrics"],
                            "final_tick": e["final_tick"],
                            "candidate_points": e["candidate_points"],
                        }
                    )
            gates = {}
            for side in "AB":
                before = [
                    x["metrics"][side] for x in evaluations if x["phase"] == "before"
                ]
                after = [
                    x["metrics"][side] for x in evaluations if x["phase"] == "after"
                ]
                delta = float(
                    np.mean([x["mean_target_error_m"] for x in before])
                    - np.mean([x["mean_target_error_m"] for x in after])
                )
                survival = float(
                    np.mean([x["survival_s"] for x in after])
                    - np.mean([x["survival_s"] for x in before])
                )
                cycles = sum(x["reach_return_cycles"] >= 1 for x in after)
                standing = sum(x["completed_3s"] for x in after)
                gates[side] = {
                    "target_error_improvement_m": delta,
                    "survival_change_s": survival,
                    "full_cycle_starts": cycles,
                    "completed_3s_starts": standing,
                    "passed": delta >= 0.05
                    and survival >= -0.1
                    and cycles >= 2
                    and standing >= 3,
                }
            result["seed_rows"].append(
                {
                    "seed": seed["seed"],
                    "native_training_transitions": count,
                    "max_behavior_error": maxerr,
                    "evaluations": evaluations,
                    "gates": gates,
                }
            )
        result.update(
            passed=True,
            source_counts=source["counts"],
            behavioral_criteria_passed=all(
                v["passed"] for r in result["seed_rows"] for v in r["gates"].values()
            ),
            sporting_qualified=False,
        )
    except Exception as error:  # noqa: BLE001 -- retain every independent audit fault
        result["error"] = type(error).__name__ + ": " + str(error)
    finally:
        ledger.close()
        if world is not None:
            world.close()
        result.update(physics=entered, wall_s=time.monotonic() - started)
        reserve = budget.reserve
        budget.reserve = 0
        try:
            budget.write_json(report / "independent-audit.json", result)
        finally:
            budget.reserve = reserve
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "passed",
                    "error",
                    "behavioral_criteria_passed",
                    "physics",
                    "wall_s",
                )
            }
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
