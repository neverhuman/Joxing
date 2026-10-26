"""Independent receipt accounting plus evaluation snapshot/policy continuation audit."""

import argparse
import json
from pathlib import Path

import numpy as np

from joxing.provenance import digest, manifest, write_json
from joxing.stance_env import StanceEnv
from joxing.stance_train import file_hash, paired_summary


def read(path):
    return json.loads(path.read_text())


def main():
    import torch
    from stable_baselines3 import SAC

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = read(args.run / "protocol.json")
    result = read(args.run / "result.json")
    content = protocol.copy()
    content.pop("protocol_hash")
    assert digest(content) == protocol["protocol_hash"] == result["protocol_hash"]
    env = StanceEnv(training=False)
    assert env.specification() == protocol["specification"]
    rows = {
        name: read(args.run / f"{name}-evaluations.json")
        for name in ("initial", "trained")
    }
    for name, values in rows.items():
        assert [r["scenario_id"] for r in values] == protocol["eval_ids"]
        assert len({r["attempt_id"] for r in values}) == len(values)
        assert all(not r["invalid"] and r["status"] != "not_run" for r in values)
        assert all(
            r["reset_audit"]["projected_equality_error"] < 1e-8
            and r["reset_audit"]["max_penetration_m"] <= 0.002
            for r in values
        )
    assert [r["initial_state_hash"] for r in rows["initial"]] == [
        r["initial_state_hash"] for r in rows["trained"]
    ]
    assert paired_summary(rows["initial"], rows["trained"]) == result["summary"]
    assert result["evaluation_physics_substeps"] == sum(
        r["physics_steps"] for values in rows.values() for r in values
    )
    findings = []
    for name, checkpoint in (("initial", "initial"), ("trained", "final")):
        assert (
            file_hash(args.run / f"{checkpoint}.zip")
            == result["checkpoint_hashes"][checkpoint]
        )
        policy = SAC.load(args.run / f"{checkpoint}.zip", device="cpu")
        record = read(args.run / f"{name}-replay.json")
        env.restore(record["snapshot"])
        while not env.done:
            action, _ = policy.predict(env.observe(), deterministic=True)
            env.step(action)
        actual = np.asarray(env.snapshot()["state"])
        reference = np.asarray(record["final_state"])
        error = float(np.max(abs(actual - reference)))
        # INTEGRATION includes solver warmstart accelerations; report physical
        # coordinates separately without loosening the declared full-state test.
        qpos_error = float(
            np.max(abs(actual[1 : 1 + env.model.nq] - reference[1 : 1 + env.model.nq]))
        )
        qvel_start = 1 + env.model.nq
        qvel_error = float(
            np.max(
                abs(
                    actual[qvel_start : qvel_start + env.model.nv]
                    - reference[qvel_start : qvel_start + env.model.nv]
                )
            )
        )
        statuses = (
            env.result()["reason"] == record["result"]["reason"]
            and env.tick == record["result"]["physics_steps"]
        )
        # Also test exact same-host restoration with this host's policy arithmetic.
        env.restore(record["snapshot"])
        while not env.done:
            action, _ = policy.predict(env.observe(), deterministic=True)
            env.step(action)
        same_host = float(np.max(abs(np.asarray(env.snapshot()["state"]) - actual)))
        findings.append(
            {
                "policy": name,
                "cross_host_integration_state_max_error": error,
                "cross_host_qpos_max_error": qpos_error,
                "cross_host_qvel_max_error": qvel_error,
                "same_host_restore_max_error": same_host,
                "outcome_tick_matches": statuses,
                "passed": same_host < 1e-9 and statuses and error < 1e-5,
            }
        )
    report = manifest() | {
        "protocol_hash": protocol["protocol_hash"],
        "source_run_commit": result["code_commit"],
        "accounting_passed": True,
        "evaluation_attempts": result["evaluation_attempts"],
        "policies": findings,
        "passed": all(x["passed"] for x in findings),
        "tolerances": {"same_host_state": 1e-9, "cross_host_closed_loop_state": 1e-5},
        "scope": "closed-loop development-policy continuation; does not prove unassisted motor competence",
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
