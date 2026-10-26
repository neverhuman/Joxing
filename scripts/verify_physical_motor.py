"""Audit all raw physical outcomes, then replay both preselected first seeds."""

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np

from joxing.native_transfer import load_controller, predict
from joxing.physical_motor import ROLES, PhysicalMotorTask, summarize, validate_protocol
from joxing.provenance import digest, manifest, write_json
from joxing.stance_train import file_hash
from joxing.support import SupportMonitor


def read(path):
    return json.loads(path.read_text())


def audit_substeps(path, row, contract, initial):
    """Rebuild loads and persistence from retained signed manifold evidence."""
    dt = contract["physics_dt"]
    monitor = SupportMonitor()
    weight = sum(contract["body_mass"]) * 9.81
    reason = latch = None
    floor_ids = {int(k): v for k, v in contract["floor_roles"].items()}
    with np.load(path) as z:
        times, loads, posture = z["intervals_s"], z["loads_N"], z["tilt_drop"]
        offsets, geom, role = (
            z["contact_offsets"],
            z["contact_geom_ids"],
            z["contact_role"],
        )
        forces, positions = z["contact_force_N"], z["contact_position_m"]
        n = row["physics_steps"]
        assert (
            times.shape == (n, 2) and loads.shape == (n, 3) and posture.shape == (n, 2)
        )
        assert (
            offsets.shape == (n + 1,) and offsets[0] == 0 and offsets[-1] == len(geom)
        )
        assert geom.shape == (len(forces), 2) and forces.shape == positions.shape == (
            len(role),
            3,
        )
        assert np.all(np.diff(offsets) >= 0) and np.all(
            offsets == offsets.astype(np.int64)
        )
        assert np.isfinite(
            np.r_[
                times.ravel(),
                loads.ravel(),
                posture.ravel(),
                forces.ravel(),
                positions.ravel(),
            ]
        ).all()
        assert np.max(abs(times[:, 0] - np.arange(n) * dt)) < 1e-8
        assert np.max(abs(times[:, 1] - (np.arange(n) + 1) * dt)) < 1e-8
        # floor is native geom 0, established by compiled model spec below.
        for i in range(n):
            actual = np.zeros(3)
            for c in range(int(offsets[i]), int(offsets[i + 1])):
                g1, g2 = (int(v) for v in geom[c])
                assert (g1 == contract["floor_geom_id"]) != (
                    g2 == contract["floor_geom_id"]
                )
                other = g2 if g1 == contract["floor_geom_id"] else g1
                code = int(role[c])
                assert code in range(3) and floor_ids[other] == ROLES[code]
                actual[code] += max(0.0, forces[c, 2])
            assert np.array_equal(actual, loads[i])
            failure = monitor.update(
                dict(zip(ROLES, actual, strict=True)), weight, *posture[i], dt
            )
            if reason is None and failure is not None:
                reason, latch = failure, float(times[i, 1])
        assert reason == row["physical_failure"] and latch == row["physical_failure_s"]
        assert row["horizon_reached"] == (
            reason is None and times[-1, 1] + 1e-9 >= 20.0
        )
        if latch is not None:
            assert 0 <= times[-1, 1] - latch < 0.02 + 1e-9
        assert initial["physical_control"]["completed_controls"] == 0
        return {
            "substeps": n,
            "manifold_rows": len(geom),
            "physical_failure": reason,
            "latch_s": latch,
        }


def verify(run, out):
    started = time.perf_counter()
    p = validate_protocol(read(run / "protocol.json"))
    result, rows, prov = (
        read(run / f) for f in ("result.json", "evaluations.json", "provenance.json")
    )
    assert digest(p) == prov["protocol_hash"]
    assert result["source_unchanged"] and result["inputs_unchanged"]
    assert all(result[k] == v for k, v in summarize(rows).items())
    assert read(run / "progress.json")["completed"] == sum(
        r["status"] == "completed" for r in rows
    )
    findings = [
        {"variant": v["id"], "snapshot": boundary, "path": path, "status": "not_run"}
        for v in p["variants"]
        for boundary in ("boundary", "mid_control")
        for path in p["replay"]["paths"]
    ]
    report = manifest() | {
        "passed": False,
        "accounting_passed": False,
        "audit_physics_steps": 0,
        "attempts": len(rows),
        "findings": findings,
        "row_audits": [],
        "source_run_commit": prov["code_commit"],
        "run_protocol_hash": prov["protocol_hash"],
        "auditor_sha256": file_hash(Path(__file__)),
    }
    write_json(out, report)
    tasks = {}
    try:
        pairs = {}
        physics = 0
        for v in p["variants"]:
            task = PhysicalMotorTask(p["environment"], v)
            tasks[v["id"]] = task
            assert task.contract == read(run / v["id"] / "specification.json")
        for row in rows:
            assert row["status"] == "completed", row
            folder = run / row["variant"]
            task = tasks[row["variant"]]
            snap = read(folder / f"starts/{row['seed']}.json")
            assert digest(snap) == row["initial_snapshot_hash"]
            assert digest(row["paired_fields"]) == row["paired_fields_hash"]
            pairs.setdefault(row["seed"], set()).add(row["paired_fields_hash"])
            task.restore(snap)
            assert digest(task.paired_fields()) == row["paired_fields_hash"]
            with np.load(folder / f"actions-{row['seed']}.npz") as z:
                a = z["actions"]
            trace = read(folder / f"trace-{row['seed']}.json")
            assert (
                a.shape == (row["control_steps"], 700)
                and np.isfinite(a).all()
                and abs(a).max() <= 1
            )
            assert (
                len(trace) == len(a)
                and row["physics_steps"] == len(a) * task.env.frame_skip
            )
            assert (
                abs(
                    row["elapsed_sim_s"]
                    - row["physics_steps"] * task.model.opt.timestep
                )
                < 1e-8
            )
            assert (
                abs(
                    row["mean_clipped_fraction"]
                    - np.mean([t["normalizer_clipped_fraction"] for t in trace])
                )
                < 1e-12
            )
            assert (
                abs(
                    row["mean_native_qpos_error"]
                    - np.mean([t["native_qpos_mean_error"] for t in trace])
                )
                < 1e-12
            )
            variation = float(np.mean(abs(np.diff(a, axis=0)))) if len(a) > 1 else 0.0
            assert abs(row["mean_absolute_action_change"] - variation) < 1e-7
            tracking = next(
                (
                    t["time_s"]
                    for t in trace
                    if t["native_qpos_error_before_reference_update"]
                    > p["environment"]["qpos_diff_th"]
                ),
                None,
            )
            assert tracking == row["first_native_tracking_failure_s"]
            assert all(
                t["first_native_tracking_failure_s"]
                == (
                    tracking
                    if tracking is not None and t["time_s"] >= tracking
                    else None
                )
                for t in trace
            )
            contract = task.contract | {"floor_geom_id": task.floor}
            raw = audit_substeps(
                folder / f"substeps-{row['seed']}.npz", row, contract, snap
            )
            report["row_audits"].append({"attempt_id": row["attempt_id"], **raw})
            physics += row["physics_steps"]
        assert len(pairs) == 20 and all(len(h) == 1 for h in pairs.values())
        assert (
            read(run / "substep-oracle.json")["passed"]
            and read(run / "v7-source-oracle.json")["passed"]
        )
        bench = read(run / "benchmark.json")
        assert len(bench["cases"]) == 4 and all(
            r["all_status"] == "completed" for r in bench["cases"]
        )
        diagnostic = sum(
            read(run / f)["physics_steps"]
            for f in ("substep-oracle.json", "v7-source-oracle.json", "benchmark.json")
        )
        assert diagnostic == result["diagnostic_physics_steps"]
        assert physics + diagnostic == result["physics_steps"]
        report.update(
            accounting_passed=True,
            formal_physics_steps=physics,
            diagnostic_physics_steps=diagnostic,
        )
        write_json(out, report)
        model, normalizer = load_controller(run / "inputs")
        remaining = (
            p["caps"]["mission_physics_steps_with_audit_video"]
            - result["physics_steps"]
        )
        deadline = started + max(0.0, p["caps"]["wall_s"] - result["wall_s"])
        for finding in findings:
            task = tasks[finding["variant"]]
            rec = read(run / finding["variant"] / "replay.json")
            snapshot = (
                rec["snapshot"]
                if finding["snapshot"] == "boundary"
                else rec["mid_snapshot"]
            )
            assert snapshot is not None and rec["reason"] is None
            with np.load(run / finding["variant"] / rec["actions_file"]) as z:
                actions = z["actions"]
            task.restore(copy.deepcopy(snapshot))
            finding["status"] = "running"
            write_json(out, report)
            before = task.physics_total
            try:
                index = rec["continuation_start"]
                if task.pending:
                    assert np.array_equal(task.action, actions[index])
                    if (
                        report["audit_physics_steps"] + task.env.frame_skip > remaining
                        or time.perf_counter() >= deadline
                    ):
                        raise TimeoutError("Registered replay reserve")
                    task.resume_control()
                    index += 1
                while not task.done:
                    if (
                        report["audit_physics_steps"]
                        + task.physics_total
                        - before
                        + task.env.frame_skip
                        > remaining
                        or time.perf_counter() >= deadline
                    ):
                        raise TimeoutError("Registered replay wall/physics reserve")
                    assert index < len(actions)
                    action = (
                        actions[index]
                        if finding["path"] == "recorded_actions"
                        else predict(model, normalizer, task.last_obs)[0]
                    )
                    assert np.array_equal(action, actions[index])
                    task.step(action)
                    index += 1
                assert index == len(actions)
                snap, expected = task.snapshot(), rec["final_snapshot"]
                error = float(
                    np.max(abs(np.asarray(snap["state"]) - expected["state"]))
                )
                obs_error = float(
                    np.max(
                        abs(
                            np.asarray(snap["last_raw_observation"])
                            - expected["last_raw_observation"]
                        )
                    )
                )
                application_matches = all(
                    digest(snap[k]) == digest(expected[k])
                    for k in (
                        "application",
                        "rng",
                        "monitor",
                        "physical_control",
                        "done",
                        "sampled_failure",
                        "max_tilt",
                    )
                )
                assert error < 1e-9 and obs_error < 1e-9 and application_matches
                finding.update(
                    status="completed",
                    integration_state_max_error=error,
                    observation_max_error=obs_error,
                    application_matches=True,
                )
            except Exception as error:
                finding.update(
                    status="invalid",
                    error=str(error),
                    traceback=traceback.format_exc(),
                    last_state=task.snapshot(),
                )
                raise
            finally:
                used = task.physics_total - before
                finding["physics_steps"] = used
                report["audit_physics_steps"] += used
                write_json(out, report)
        normalizer.check()
        report.update(
            passed=True,
            reference_replay_passed=True,
            normalizer_unchanged=True,
            physical_completion_threshold_met=result[
                "physical_completion_threshold_met"
            ],
            physical_diagnostic_passed=result["physical_completion_threshold_met"],
            scope="Frozen muscle motor continuation; no stationary stance, learning or boxing claim.",
        )
    except Exception as error:  # noqa: BLE001 -- failed audits are evidence
        report.update(passed=False, error=str(error), traceback=traceback.format_exc())
    finally:
        for task in tasks.values():
            task.close()
        report["audit_wall_s"] = time.perf_counter() - started
        write_json(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.run, args.out)
    except Exception as error:  # noqa: BLE001 -- retain precondition failures
        report = manifest() | {
            "passed": False,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.out, report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("findings", "row_audits", "traceback")
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)
