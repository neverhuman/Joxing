"""D1 neutral residual/source-v8 replay; no training or gate promotion."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from joxing.motor_residual import MotorResidualEnv
from joxing.physical_motor import INTEGRATION
from joxing.provenance import ROOT, digest, manifest, write_json
from joxing.stance_train import file_hash
from joxing.wave import check_resources

PROTOCOL_SHA = "136427ced9b436e6ff3ed13b2859e618db4d17f4e85c816480dacc8f914200d1"


def run(protocol, inputs, out):
    p = json.loads(protocol.read_text())
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Unregistered neutral residual protocol")
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".neutral-residual-d1.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    began = time.monotonic()
    rows = [
        {"snapshot": point, "path": mode, "status": "not_run"}
        for point in ("boundary", "mid_control")
        for mode in ("recorded_actions", "closed_loop")
    ]
    report = manifest() | {
        "status": "running",
        "pid": os.getpid(),
        "script_sha256": file_hash(Path(__file__)),
        "protocol_sha256": digest(p),
        "cases": rows,
        "physics_steps": 0,
        "boxing_games": 0,
        "optimizer_updates": 0,
        "maximum_allocated_physics_steps": p["caps"]["physics_steps"],
        "physics_accounting_exact": False,
    }
    write_json(out / "protocol.json", p)
    write_json(out / "result.json", report)
    env = None
    active = None

    def guard():
        if time.monotonic() - began >= p["caps"]["wall_s"]:
            raise TimeoutError("D1 wall cap")
        if env is not None and env.task.physics_total + 10 > p["caps"]["physics_steps"]:
            raise TimeoutError("D1 physics cap")
        if (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            >= p["caps"]["raw_bytes"]
        ):
            raise TimeoutError("D1 raw cap")
        check_resources(out, p["caps"])

    try:
        import torch

        report["initial_capacity"] = check_resources(out, p["caps"], True)
        for path, sha in p["source_files"].items():
            if file_hash(ROOT / path) != sha:
                raise ValueError("D1 source/reference mismatch: " + path)
        base = ROOT / "artifacts/v0.0.8/physical/native_2ms"
        source = json.loads((base / "replay.json").read_text())
        source_trace = json.loads((base / "trace-74000.json").read_text())
        with np.load(base / "actions-74000.npz", allow_pickle=False) as tape:
            commands = tape["actions"].copy()
        env = MotorResidualEnv.from_inputs(inputs)
        report.update(
            bundle=env.contract,
            bundle_hash=env.contract_hash,
            groups=[list(g) for g in env.groups],
            threads=torch.get_num_threads(),
        )
        assert torch.get_num_threads() == 1
        group_by_muscle = {
            env.task.model.actuator(i).name: k
            for k, group in enumerate(env.groups)
            for i in group
        }
        report["independent_arm_pairs"] = {}
        for name in p["independent_arm_pairs"]:
            pair = [group_by_muscle[name + "_r"], group_by_muscle[name + "_l"]]
            if pair[0] == pair[1]:
                raise ValueError(
                    "Residual partition aliases contralateral arms: " + name
                )
            report["independent_arm_pairs"][name] = pair
        with torch.inference_mode():
            for row in rows:
                active = row
                row["status"] = "running"
                stem = row["snapshot"] + "-" + row["path"]
                before = env.task.physics_total
                physical = source[
                    "snapshot" if row["snapshot"] == "boundary" else "mid_snapshot"
                ]
                first = (
                    physical["physical_control"]["completed_controls"] * 10
                    + physical["physical_control"]["substep"]
                )
                env.import_neutral_source(physical, 74000)
                start = env.snapshot()
                write_json(out / (stem + "-start.json"), start)
                env.restore(json.loads((out / (stem + "-start.json")).read_text()))
                assert digest(start) == digest(env.snapshot())
                trace, actions = [], []
                index = source["continuation_start"]
                write_json(out / "result.json", report)
                while not env.task.done:
                    guard()
                    if env.task.pending:
                        result = env.resume_control()
                    elif row["path"] == "recorded_actions":
                        env.begin_recorded_neutral(commands[index])
                        result = env.resume_control()
                    else:
                        result = env.step(np.zeros(138, np.float32))
                    transition = env.last_transition
                    command = np.asarray(transition["command"], dtype=np.float32)
                    assert np.array_equal(command, commands[index])
                    assert np.array_equal(transition["residual"], np.zeros(138))
                    telemetry = transition["telemetry"]
                    for key, expected in source_trace[index].items():
                        if key not in {
                            "normalizer_clipped_fraction",
                            "mean_absolute_action_change",
                        }:
                            assert telemetry[key] == expected, (index, key)
                    assert (
                        result[4]["teacher_clipped_fraction"]
                        == source_trace[index]["normalizer_clipped_fraction"]
                    )
                    actions.append(command)
                    trace.append(
                        {
                            "source_control_index": index,
                            "motor_reward": result[1],
                            "terminated": result[2],
                            "truncated": result[3],
                            "native_reward": transition["native_reward"],
                            "telemetry": telemetry,
                        }
                    )
                    index += 1
                    if index % 10 == 0:
                        row.update(
                            controls_completed=index - source["continuation_start"],
                            physics_steps=env.task.physics_total - before,
                        )
                        report["physics_steps"] = env.task.physics_total
                        write_json(out / "result.json", report)
                assert index == len(commands)
                assert digest(env.task.snapshot()) == digest(source["final_snapshot"])
                env.check_bundle()
                write_json(out / (stem + "-final.json"), env.snapshot())
                write_json(out / (stem + "-trace.json"), trace)
                np.savez_compressed(
                    out / (stem + "-actions.npz"), actions=np.asarray(actions)
                )
                env.task.dump_substeps(out / (stem + "-substeps.npz"))
                with (
                    np.load(base / "substeps-74000.npz", allow_pickle=False) as a,
                    np.load(out / (stem + "-substeps.npz"), allow_pickle=False) as b,
                ):
                    offset = int(a["contact_offsets"][first])
                    for key in a.files:
                        expected = (
                            a[key][first:] - offset
                            if key == "contact_offsets"
                            else a[key][first:]
                            if key in {"intervals_s", "loads_N", "tilt_drop"}
                            else a[key][offset:]
                        )
                        assert expected.dtype == b[key].dtype and np.array_equal(
                            expected, b[key]
                        ), key
                row.update(
                    status="completed",
                    first_source_substep=first,
                    controls=index - source["continuation_start"],
                    physics_steps=env.task.physics_total - before,
                    final_snapshot_sha256=file_hash(out / (stem + "-final.json")),
                    source_state_actions_contacts_telemetry_exact=True,
                )
                report["physics_steps"] = env.task.physics_total
                write_json(out / "result.json", report)
        report.update(
            status="completed",
            interface_gate_passed=True,
            training_allowed=False,
            next_gate="D2pilot runner/machine protocol/held-out evaluation and bounded supervision still required",
        )
    except Exception as error:  # noqa: BLE001 -- raw failure and every future row retained
        if active is not None and active["status"] == "running":
            active.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
                physics_steps=env.task.physics_total - before,
            )
        if env is not None:
            state = np.empty(mujoco.mj_stateSize(env.task.model, INTEGRATION))
            mujoco.mj_getState(env.task.model, env.task.data, state, INTEGRATION)
            np.savez_compressed(
                out / "fault-arrays.npz", state=state, observation=env.task.last_obs
            )
        report.update(
            status="halted",
            error=str(error),
            interface_gate_passed=False,
            training_allowed=False,
        )
    finally:
        report.update(
            wall_s=time.monotonic() - began,
            physics_steps=0 if env is None else env.task.physics_total,
            physics_accounting_exact=True,
        )
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "halted_before_case"
        write_json(out / "result.json", report)
        if env is not None:
            env.close()
        lease.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments/v0.0.10/neutral-interface-protocol.json",
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.inputs, args.out)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in [
                    "status",
                    "interface_gate_passed",
                    "physics_steps",
                    "wall_s",
                    "error",
                ]
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["status"] == "completed" else 1)
