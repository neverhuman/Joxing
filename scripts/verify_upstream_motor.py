"""Independent attempt, pairing, immutable-normalizer and native replay audit."""

import argparse
import json
from pathlib import Path

import numpy as np

from joxing.provenance import digest, manifest, write_json
from joxing.stance_train import file_hash
from joxing.upstream_motor import (
    MODES,
    FrozenNormalizer,
    NativeTask,
    load_dynsyn,
    set_decoder_mode,
    summarize,
    validate_protocol,
)
from joxing.wave import canonical_actor_hash


def read(path):
    return json.loads(path.read_text())


def verify(run):
    import torch

    torch.set_num_threads(1)
    p = validate_protocol(read(run / "protocol.json"))
    prov = read(run / "provenance.json")
    result = read(run / "result.json")
    rows = read(run / "evaluations.json")
    oracle = read(run / "loader-oracle.json")
    assert digest(p) == prov["protocol_hash"]
    summary = summarize(rows)
    assert all(result[k] == v for k, v in summary.items())
    assert result["source_unchanged"] and result["inputs_unchanged"]
    assert all(r["status"] == "completed" for r in rows)
    physics = 0
    for seed in p["seeds"]:
        pair = [r for r in rows if r["seed"] == seed]
        assert len(pair) == 2 and {r["mode"] for r in pair} == set(MODES)
        assert pair[0]["initial_snapshot_hash"] == pair[1]["initial_snapshot_hash"]
        for r in pair:
            start = read(run / f"starts/{r['mode']}-{seed}.json")
            assert digest(start) == r["initial_snapshot_hash"]
            with np.load(run / f"actions-{r['mode']}-{seed}.npz") as data:
                actions = data["actions"]
                assert actions.shape == (r["control_steps"], 700)
                assert np.isfinite(actions).all() and np.max(abs(actions)) <= 1
            trace = read(run / f"trace-{r['mode']}-{seed}.json")
            assert len(trace) == r["control_steps"]
            assert r["physics_steps"] == r["control_steps"] * 10
            assert abs(r["elapsed_sim_s"] - r["physics_steps"] * 0.002) < 1e-8
            assert r["normalizer_stats_hash"] == oracle["normalizer_stats_hash"]
            assert r["amplitude"] == (None if r["mode"] == MODES[0] else 0.075)
            physics += r["physics_steps"]
    assert physics == result["physics_steps"]
    model = load_dynsyn(run / "inputs/model.zip")
    assert file_hash(run / "inputs/model.zip") == oracle["model_sha256"]
    assert canonical_actor_hash(model.actor) == oracle["canonical_actor_hash"]
    normalizer = FrozenNormalizer.load(run / "inputs/normalizer.pkl", 3601)
    assert file_hash(run / "inputs/normalizer.pkl") == oracle["normalizer_sha256"]
    assert normalizer.fingerprint == oracle["normalizer_stats_hash"]
    task = NativeTask(p["environment"])
    assert task.contract == read(run / "specification.json")
    findings = []
    for mode in MODES:
        record = read(run / f"replay-{mode}.json")
        set_decoder_mode(model, mode)
        if record["snapshot"] is None or not record["continuation_actions"]:
            findings.append(
                {"mode": mode, "status": "not_run", "reason": record["reason"]}
            )
            continue
        actuals = []
        for path in ("recorded_actions", "closed_loop", "closed_loop_repeat"):
            task.restore(record["snapshot"])
            if path == "recorded_actions":
                for action in record["continuation_actions"]:
                    task.step(action)
            else:
                while not task.done:
                    action, _ = model.predict(
                        normalizer.apply(task.last_obs), deterministic=True
                    )
                    task.step(action)
            snap = task.snapshot()
            error = float(
                np.max(
                    abs(np.asarray(snap["state"]) - record["final_snapshot"]["state"])
                )
            )
            obs_error = float(
                np.max(
                    abs(
                        np.asarray(snap["last_raw_observation"])
                        - record["final_snapshot"]["last_raw_observation"]
                    )
                )
            )
            actuals.append(snap)
            findings.append(
                {
                    "mode": mode,
                    "path": path,
                    "status": "completed",
                    "integration_state_max_error": error,
                    "observation_max_error": obs_error,
                    "done_matches": snap["done"] == record["final_snapshot"]["done"],
                    "sampled_failure_matches": snap["sampled_failure"]
                    == record["final_snapshot"]["sampled_failure"],
                }
            )
        assert np.array_equal(actuals[1]["state"], actuals[2]["state"])
    normalizer.check()
    audit_physics = task.physics_total
    task.close()
    reference_passed = len(findings) == 6 and all(
        r["status"] == "completed"
        and r["integration_state_max_error"] < 1e-9
        and r["observation_max_error"] < 1e-9
        and r["done_matches"]
        and r["sampled_failure_matches"]
        for r in findings
    )
    return manifest() | {
        "auditor_sha256": file_hash(Path(__file__)),
        "run_protocol_hash": prov["protocol_hash"],
        "source_run_commit": prov["code_commit"],
        "source_host": prov["host"],
        "accounting_passed": True,
        "attempts": len(rows),
        "invalid": summary["invalid"],
        "recomputed_physics_steps": physics,
        "audit_physics_steps": audit_physics,
        "findings": findings,
        "reference_replay_passed": reference_passed,
        "normalizer_unchanged": True,
        "native_task_completion_threshold_met": summary[
            "native_task_completion_threshold_met"
        ],
        "native_teacher_diagnostic_passed": reference_passed
        and summary["native_task_completion_threshold_met"],
        "passed": reference_passed,
        "scope": "Source-host qualification requires strict replay; native locomotion is separate from Joxing stance/boxing.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.run)
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
