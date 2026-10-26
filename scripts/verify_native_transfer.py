"""Independent raw accounting and strict first-seed source-host continuation."""

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np

from joxing.native_transfer import (
    TransferTask,
    load_controller,
    predict,
    summarize,
    validate_protocol,
)
from joxing.provenance import digest, manifest, write_json
from joxing.stance_train import file_hash


def read(path):
    return json.loads(path.read_text())


def verify(run):
    started = time.perf_counter()
    p = validate_protocol(read(run / "protocol.json"))
    result = read(run / "result.json")
    prov = read(run / "provenance.json")
    rows = read(run / "evaluations.json")
    assert digest(p) == prov["protocol_hash"]
    assert result["source_unchanged"] and result["inputs_unchanged"]
    assert all(result[k] == v for k, v in summarize(rows).items())
    physics = 0
    pair_hashes = {}
    first = []
    for row in rows:
        assert row["status"] == "completed", row
        v = next(v for v in p["variants"] if v["id"] == row["variant"])
        folder = run / v["id"]
        seed = row["seed"]
        snap = read(folder / f"starts/{seed}.json")
        assert digest(snap) == row["initial_snapshot_hash"]
        assert digest(row["paired_fields"]) == row["paired_fields_hash"]
        pair_hashes.setdefault(seed, set()).add(row["paired_fields_hash"])
        # Cross-check comparable fields against actual integration and app state.
        state = np.asarray(snap["state"])
        paired = row["paired_fields"]
        assert np.array_equal(state[1:86], paired["qpos"])
        assert np.array_equal(state[86:171], paired["qvel"])
        assert np.array_equal(state[171:871], paired["activation"])
        assert (
            paired["phase"] == snap["application"]["init_time"]
            and paired["rng"] == snap["rng"]
        )
        a = np.load(folder / f"actions-{seed}.npz")["actions"]
        trace = read(folder / f"trace-{seed}.json")
        assert (
            a.shape == (row["control_steps"], 700)
            and np.isfinite(a).all()
            and abs(a).max() <= 1
        )
        assert len(trace) == len(a)
        assert row["physics_steps"] == len(a) * round(0.02 / v["physics_dt_s"])
        assert (
            abs(row["elapsed_sim_s"] - row["physics_steps"] * v["physics_dt_s"]) < 1e-8
        )
        assert (
            abs(
                row["mean_native_qpos_error"]
                - np.mean([t["native_qpos_mean_error"] for t in trace])
            )
            < 1e-12
        )
        assert (
            abs(
                row["mean_clipped_fraction"]
                - np.mean([t["normalizer_clipped_fraction"] for t in trace])
            )
            < 1e-12
        )
        variation = (
            float(np.mean(np.mean(abs(np.diff(a, axis=0)), axis=1)))
            if len(a) > 1
            else 0
        )
        assert abs(row["mean_absolute_action_change"] - variation) < 1e-7
        physics += row["physics_steps"]
        if seed == p["replay"]["seed"]:
            first.append(row)
    assert all(len(h) == 1 for h in pair_hashes.values()) and len(pair_hashes) == 20
    assert physics + result["diagnostic_physics_steps"] == result["physics_steps"]
    bench = read(run / "benchmark.json")
    assert len(bench["cases"]) == 4 and all(
        r["all_status"] == "completed" for r in bench["cases"]
    )
    oracle = read(run / "native-oracle.json")
    assert (
        oracle["feature_oracle"]["passed"]
        and oracle["feature_oracle"]["max_feature_error"] == 0
    )
    assert (
        oracle["v6_actual_continuation"]["passed"]
        and oracle["v6_actual_continuation"]["max_reference_state_error"] == 0
    )
    model, n = load_controller(run / "inputs")
    deadline = started + max(0, p["caps"]["wall_s"] - result["wall_s"])
    remaining = (
        p["caps"]["mission_physics_steps_with_audit_video"] - result["physics_steps"]
    )
    findings = []
    audit_steps = 0
    for row in first:
        v = next(v for v in p["variants"] if v["id"] == row["variant"])
        folder = run / v["id"]
        task = TransferTask(p["environment"], v)
        try:
            assert task.contract == read(folder / "specification.json")
            rec = read(folder / "replay.json")
            if rec["snapshot"] is None or rec["reason"] is not None:
                findings.append(
                    {"variant": v["id"], "status": "not_run", "reason": rec["reason"]}
                )
                continue
            actions = np.load(folder / rec["actions_file"])["actions"][
                rec["continuation_start"] :
            ]
            repeated = []
            for path in p["replay"]["paths"]:
                task.restore(copy.deepcopy(rec["snapshot"]))
                previous = task.physics_total
                if path == "recorded_actions":
                    sequence = iter(actions)
                while not task.done:
                    if (
                        time.perf_counter() >= deadline
                        or audit_steps
                        + task.physics_total
                        - previous
                        + task.env.frame_skip
                        > remaining
                    ):
                        raise TimeoutError(
                            "Remaining registered mission wall/physics cap"
                        )
                    action = (
                        next(sequence)
                        if path == "recorded_actions"
                        else predict(model, n, task.last_obs)[0]
                    )
                    task.step(action)
                audit_steps += task.physics_total - previous
                snap = task.snapshot()
                expected = rec["final_snapshot"]
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
                findings.append(
                    {
                        "variant": v["id"],
                        "path": path,
                        "status": "completed",
                        "integration_state_max_error": error,
                        "observation_max_error": obs_error,
                        "done_matches": snap["done"] == expected["done"],
                        "sampled_failure_matches": snap["sampled_failure"]
                        == expected["sampled_failure"],
                    }
                )
                repeated.append(snap["state"])
            assert repeated[1] == repeated[2]
        finally:
            task.close()
    n.check()
    passed = len(findings) == 12 and all(
        f["status"] == "completed"
        and f["integration_state_max_error"] < 1e-9
        and f["observation_max_error"] < 1e-9
        and f["done_matches"]
        and f["sampled_failure_matches"]
        for f in findings
    )
    return manifest() | {
        "accounting_passed": True,
        "attempts": len(rows),
        "invalid": 0,
        "formal_physics_steps": physics,
        "diagnostic_physics_steps": result["diagnostic_physics_steps"],
        "audit_physics_steps": audit_steps,
        "audit_wall_s": time.perf_counter() - started,
        "reference_replay_passed": passed,
        "passed": passed,
        "completion_thresholds_met": result["completion_thresholds_met"],
        "native_transfer_diagnostic_passed": passed
        and result["completion_thresholds_met"],
        "findings": findings,
        "normalizer_unchanged": True,
        "auditor_sha256": file_hash(Path(__file__)),
        "source_run_commit": prov["code_commit"],
        "run_protocol_hash": prov["protocol_hash"],
        "scope": "Source-host frozen native equipment/timestep transfer; no new learning or boxing claim.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.run)
    except Exception as error:  # noqa: BLE001 -- persist a failed audit receipt
        report = manifest() | {
            "passed": False,
            "status": "audit_failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    write_json(args.out, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("findings", "traceback")},
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)
