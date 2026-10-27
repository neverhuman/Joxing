"""D2 held-out physical evidence and exact frozen-policy mid-command replay."""

import json
from pathlib import Path

import numpy as np

from .provenance import digest, write_json
from .wave import canonical_actor_hash


def transition_record(env, result, index):
    observation, reward, terminated, truncated, info = result
    if not np.isfinite(observation).all() or not np.isfinite(reward):
        raise ValueError("Nonfinite evaluation transition")
    if info["bootstrap_mask"] != float(not terminated) or info["trace_mask"] != float(
        not (terminated or truncated)
    ):
        raise ValueError("Evaluation mask mismatch")
    if (terminated or truncated) and not np.array_equal(
        observation, info["terminal_observation"]
    ):
        raise ValueError("Evaluation terminal observation mismatch")
    return {
        "control_index": index,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "bootstrap_mask": info["bootstrap_mask"],
        "trace_mask": info["trace_mask"],
        "alive_seconds": info["alive_seconds"],
        "teacher_clipped_fraction": info["teacher_clipped_fraction"],
        "observation_hash": digest(observation.tolist()),
        "native_reward": info["native_reward"],
        "telemetry": env.last_transition["telemetry"],
    }


def save_fault(env, folder, error):
    import mujoco

    from .physical_motor import INTEGRATION

    state = np.empty(mujoco.mj_stateSize(env.task.model, INTEGRATION))
    mujoco.mj_getState(env.task.model, env.task.data, state, INTEGRATION)
    observation = (
        np.empty(0, np.float32) if env.task.last_obs is None else env.task.last_obs
    )
    np.savez_compressed(
        folder / "fault-arrays.npz", state=state, observation=observation
    )
    write_json(
        folder / "fault.json",
        {
            "type": type(error).__name__,
            "reason": str(error),
            "physics_steps_cumulative": env.task.physics_total,
        },
    )


def collect(env, model, folder, row, guard):
    """Persist each attempted evaluation; no reset or source state is hidden.

    guard(10) must reserve the next complete command against global limits.
    Initial/boundary/mid/final snapshots and all substeps are actual saved state.
    Parent stops further rows on any returned invalid/incomplete status.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = env.task.physics_total
    row = dict(
        row,
        status="running",
        bundle_hash=env.contract_hash,
        residual_actor_hash=canonical_actor_hash(model.actor),
        acting_batch_size=1,
    )
    write_json(folder / "result.json", row)
    trace, residuals, teachers, commands = [], [], [], []
    try:
        guard(0)
        observation, _ = env.reset(seed=row["evaluation_seed"])
        write_json(folder / "initial.json", env.snapshot())
        index = 0
        while not env.task.done:
            guard(10)
            residual = np.asarray(
                model.predict(observation, deterministic=True)[0], dtype=np.float32
            )
            if row["stage"] == "initial" and not np.array_equal(
                residual, np.zeros(138, np.float32)
            ):
                raise ValueError("Initial deterministic residual must be exactly zero")
            if index == 10:
                write_json(folder / "boundary.json", env.snapshot())
            env.begin_control(residual)
            if index == 10:
                env.advance_substeps(3)
                write_json(folder / "mid.json", env.snapshot())
            result = env.resume_control()
            record = transition_record(env, result, index)
            trace.append(record)
            transition = env.last_transition
            residuals.append(transition["residual"])
            teachers.append(transition["teacher"])
            commands.append(transition["command"])
            observation = result[0]
            index += 1
            if index % 10 == 0:
                write_json(
                    folder / "progress.json",
                    {
                        "controls": index,
                        "physics_steps": env.task.physics_total - before,
                        "time_s": record["telemetry"]["time_s"],
                    },
                )
        env.check_bundle()
        assert canonical_actor_hash(model.actor) == row["residual_actor_hash"]
        write_json(folder / "final.json", env.snapshot())
        failure = env.task.physical_failure_s
        row.update(
            status="completed",
            survival_s=min(20.0, 20.0 if failure is None else failure),
            physical_failure=env.task.physical_failure,
            terminated=result[2],
            truncated=result[3],
            first_tracking_failure_s=env.task.first_tracking_failure_s,
            motor_return=float(sum(r["reward"] for r in trace)),
        )
    except Exception as error:  # noqa: BLE001 -- retain every failed evaluation/replay
        row.update(
            status="incomplete" if isinstance(error, TimeoutError) else "invalid",
            reason=type(error).__name__ + ": " + str(error),
        )
        save_fault(env, folder, error)
    finally:
        row.update(
            controls=len(trace),
            physics_steps=env.task.physics_total - before,
            physics_accounting_exact=True,
            mid_snapshot_available=(folder / "mid.json").exists(),
        )
        np.savez_compressed(
            folder / "actions.npz",
            residuals=np.asarray(residuals, dtype=np.float32).reshape(-1, 138),
            teachers=np.asarray(teachers, dtype=np.float32).reshape(-1, 700),
            commands=np.asarray(commands, dtype=np.float32).reshape(-1, 700),
        )
        write_json(folder / "trace.json", trace)
        env.task.dump_substeps(folder / "substeps.npz")
        write_json(folder / "result.json", row)
    return row


def replay(env, model, source, folder, guard):
    """Actual scalar source-host continuation, using the separately loaded model.

    Three held substeps precede the restore point and are excluded from attempted
    replay physics; no fresh residual/teacher decision while that command is held.
    """
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    expected_row = json.loads((source / "result.json").read_text())
    expected_trace = json.loads((source / "trace.json").read_text())
    before = env.task.physics_total
    report = {"status": "running", "source": str(source), "physics_steps": 0}
    write_json(folder / "result.json", report)
    actual_trace, commands, residuals, teachers = [], [], [], []
    try:
        guard(0)
        assert (
            expected_row["status"] == "completed"
            and expected_row["mid_snapshot_available"]
        )
        assert canonical_actor_hash(model.actor) == expected_row["residual_actor_hash"]
        assert env.contract_hash == expected_row["bundle_hash"]
        env.restore(json.loads((source / "mid.json").read_text()))
        assert env.task.pending and env.task.substep == 3
        write_json(folder / "start.json", env.snapshot())
        index = 10
        while not env.task.done:
            guard(10 - env.task.substep if env.task.pending else 10)
            if env.task.pending:
                result = env.resume_control()
            else:
                observation = env.observation()
                residual = np.asarray(
                    model.predict(observation, deterministic=True)[0], np.float32
                )
                result = env.step(residual)
            record = transition_record(env, result, index)
            actual_trace.append(record)
            for records, key in [
                (commands, "command"),
                (teachers, "teacher"),
                (residuals, "residual"),
            ]:
                records.append(env.last_transition[key])
            assert record == expected_trace[index], ("trace", index)
            index += 1
            if index % 10 == 0:
                write_json(
                    folder / "progress.json",
                    {
                        "controls": len(actual_trace),
                        "physics_steps": env.task.physics_total - before,
                    },
                )
        final = env.snapshot()
        write_json(folder / "final.json", final)
        assert digest(final) == digest(json.loads((source / "final.json").read_text()))
        with np.load(source / "actions.npz", allow_pickle=False) as a:
            for key, records in [
                ("commands", commands),
                ("residuals", residuals),
                ("teachers", teachers),
            ]:
                assert np.array_equal(np.asarray(records, np.float32), a[key][10:]), key
        env.task.dump_substeps(folder / "substeps.npz")
        first = 103
        with (
            np.load(source / "substeps.npz", allow_pickle=False) as a,
            np.load(folder / "substeps.npz", allow_pickle=False) as b,
        ):
            offset = int(a["contact_offsets"][first])
            assert set(a.files) == set(b.files)
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
        report.update(
            status="completed",
            exact=True,
            source_first_substep=103,
            source_actor_hash=expected_row["residual_actor_hash"],
            bundle_hash=env.contract_hash,
        )
    except Exception as error:  # noqa: BLE001 -- retain every failed evaluation/replay
        report.update(
            status="incomplete" if isinstance(error, TimeoutError) else "invalid",
            reason=type(error).__name__ + ": " + str(error),
        )
        save_fault(env, folder, error)
    finally:
        report.update(
            physics_steps=env.task.physics_total - before,
            controls=len(actual_trace),
            physics_accounting_exact=True,
        )
        write_json(folder / "trace.json", actual_trace)
        np.savez_compressed(
            folder / "actions.npz",
            residuals=np.asarray(residuals, np.float32).reshape(-1, 138),
            teachers=np.asarray(teachers, np.float32).reshape(-1, 700),
            commands=np.asarray(commands, np.float32).reshape(-1, 700),
        )
        if not (folder / "substeps.npz").exists():
            env.task.dump_substeps(folder / "substeps.npz")
        write_json(folder / "result.json", report)
    return report
