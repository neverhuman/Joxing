"""Independently audit all D2 attempts; never launch, infer a teacher or step."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from joxing.motor_audit import evaluation, read, replay_tail, status_counts, worker
from joxing.motor_pilot import TRAINING_SEEDS, paired_conclusion, planned_rows
from joxing.motor_protocol import optimizer_hashes, validate
from joxing.provenance import ROOT, digest, write_json
from joxing.stance_train import file_hash
from joxing.wave import canonical_actor_hash


def audit(folder):
    r = read(folder / "result.json")
    p = read(folder / "protocol.json")
    assert r["protocol"] == p and digest(p) == r["protocol_sha256"]
    validate(p, r["protocol_sha256"], source=False)
    assert r["status"] in {"completed", "halted"}, "Running outcome cannot be concluded"
    supervision = folder.parent / "supervision-01"
    supervisor = read(supervision / "supervisor.json")
    assert supervisor["source_commit"] == r["code_commit"]
    assert supervisor["protocol_sha256"] == r["protocol_sha256"]
    tests = next(s for s in supervisor["stages"] if s["stage"] == "tests")
    stage = next(s for s in supervisor["stages"] if s["stage"] == "D2")
    assert tests["returncode"] == 0 and not tests["live_recorded_processes"]
    assert re.findall(r"(\d+) passed", (supervision / "tests.log").read_text()) == [
        str(p["qualification"]["linux_tests"])
    ]
    assert (
        stage["status"] in {"completed", "failed"}
        and not stage["live_recorded_processes"]
    )
    for process in stage["owned_processes"]:
        path = Path("/proc") / str(process["pid"]) / "stat"
        if path.exists():
            stat = path.read_text().rsplit(")", 1)[1].split()
            assert int(stat[19]) != process["start_ticks"] or stat[0] == "Z"
    committed = json.loads(
        subprocess.check_output(
            ["git", "show", r["code_commit"] + ":experiments/v0.0.10/protocol.json"],
            cwd=ROOT,
        )
    )
    assert committed == p
    for name, sha in p["source_files"].items():
        original = subprocess.check_output(
            ["git", "show", r["code_commit"] + ":" + name], cwd=ROOT
        )
        assert (
            hashlib.sha256(original).hexdigest() == sha
            and file_hash(ROOT / name) == sha
        ), name
    assert [
        (x["training_seed"], x["stage"], x["evaluation_seed"]) for x in r["evaluations"]
    ] == [
        (x["training_seed"], x["stage"], x["evaluation_seed"]) for x in planned_rows()
    ]
    assert [x["training_seed"] for x in r["training"]] == list(TRAINING_SEEDS)
    assert [x["training_seed"] for x in r["replays"]] == list(TRAINING_SEEDS)
    manifest = read(folder / "raw-manifest.json")
    actual = {
        str(path.relative_to(folder))
        for path in folder.rglob("*")
        if path.is_file()
        and path.name not in {"raw-manifest.json", "independent-audit.json"}
    }
    assert actual == set(manifest["files"]), "Raw manifest completeness"
    for name, row in manifest["files"].items():
        path = folder / name
        assert (
            path.stat().st_size == row["bytes"] and file_hash(path) == row["sha256"]
        ), name
    findings = []
    context_path = folder / "frozen-context.json"
    context = read(context_path) if context_path.exists() else None
    if context is not None:
        assert digest(context["bundle"]) == p["bundle_hash"]
        assert digest(context["groups"]) == context["bundle"]["groups_hash"]
        assert digest(context["normalizer"]) == context["bundle"]["normalizer_hash"]
        assert (
            digest(context["physical_specification"])
            == context["bundle"]["physical_contract"]
        )
        stats = context["normalizer"]
        assert not stats["training"] and not stats["norm_reward"] and stats["norm_obs"]
    eval_physics = training_lower = training_upper = replay_physics = 0
    for row in r["evaluations"]:
        path = (
            folder
            / f"seed-{row['training_seed']}"
            / (row["stage"] + "-" + str(row["evaluation_seed"]))
        )
        if row["status"] == "not_run":
            assert row.get("reason") and not path.exists()
            continue
        assert context is not None
        if row["status"] == "completed":
            assert read(path / "result.json") == row
            findings.append(evaluation(path, context["groups"], context))
        else:
            assert row["status"] in {"invalid", "incomplete"}
            assert path.exists() and row.get("reason")
            if (path / "fault-arrays.npz").exists():
                with np.load(path / "fault-arrays.npz", allow_pickle=False) as a:
                    assert "state" in a and "observation" in a
            findings.append(
                {
                    "status": row["status"],
                    "folder": str(path.relative_to(folder)),
                    "physics_steps": row.get("physics_steps", 0),
                    "scope": "Fault/partial receipts retained; no passing physical outcome inferred",
                }
            )
        eval_physics += row.get("physics_steps", 0)
    for index, row in enumerate(r["training"]):
        path = folder / f"seed-{row['training_seed']}"
        if row["status"] == "not_run":
            assert row.get("reason")
            continue
        assert read(path / "training.json") == row
        training_lower += row["physics_steps_lower_bound"]
        training_upper += row["physics_steps_upper_bound"]
        if row["status"] == "completed":
            assert (
                row["transitions"] == 8192
                and row["optimizer_updates"] == 448
                and row["warmup_transitions"] == 1024
            )
            workers = [
                worker(path / f"worker-{w}", index, w, context["groups"])
                for w in range(4)
            ]
            assert (
                sum(x["physics_steps"] for x in workers)
                == row["physics_steps_lower_bound"]
                == row["physics_steps_upper_bound"]
                == 81920
            )
            updates = row["update_receipts"]
            assert len(updates) == 448
            assert [u["transitions"] for u in updates] == list(range(1040, 8193, 16))
            assert [u["before"] for u in updates] == list(range(448)) and [
                u["after"] for u in updates
            ] == list(range(1, 449))
            assert all(
                np.isfinite(list(u["losses"].values())).all() and u["wall_s"] >= 0
                for u in updates
            )
            assert row["wall_s"] < 600 and not row["forced_worker_pids"]
            findings.append(
                {
                    "training_seed": row["training_seed"],
                    "workers": workers,
                    "updates": 448,
                }
            )
            from joxing.motor_run import MeasuredSAC

            identities = []
            for stage in ("initial", "final"):
                meta = read(path / (stage + "-bundle.json"))
                assert file_hash(path / (stage + ".zip")) == meta["checkpoint_sha256"]
                model = MeasuredSAC.load(path / (stage + ".zip"), device="cpu")
                assert optimizer_hashes(model) == meta["optimizer_state_hashes"]
                assert canonical_actor_hash(model.actor) == meta["actor_hash"]
                assert canonical_actor_hash(model.critic) == meta["critic_hash"]
                assert canonical_actor_hash(model.critic_target) == meta["target_hash"]
                assert (
                    model.num_timesteps
                    == meta["transitions"]
                    == (0 if stage == "initial" else 8192)
                )
                assert (
                    model._n_updates
                    == meta["optimizer_updates"]
                    == (0 if stage == "initial" else 448)
                )
                assert all(
                    np.isfinite(t.detach().cpu().numpy()).all()
                    for module in (model.actor, model.critic, model.critic_target)
                    for t in module.parameters()
                )
                identities.append(meta)
                del model
            for key in ("actor_hash", "critic_hash", "target_hash"):
                assert identities[0][key] != identities[1][key]
        else:
            assert row["status"] in {"invalid", "incomplete"} and row.get("reason")
            assert (
                row["physics_steps_lower_bound"]
                <= row["physics_steps_upper_bound"]
                <= 81920
            )
            # Every available private owner stays visible, including no-confirmation.
            assert len(row["workers"]) == 4
            findings.append(
                {
                    "training_seed": row["training_seed"],
                    "status": row["status"],
                    "workers": row["workers"],
                }
            )
    for row in r["replays"]:
        if row["status"] == "not_run":
            assert row.get("reason")
            continue
        path = folder / f"seed-{row['training_seed']}"
        if row["status"] == "completed":
            findings.append(replay_tail(path / "replay-83000", path / "final-83000"))
        else:
            assert row["status"] in {"invalid", "incomplete"} and row.get("reason")
        replay_physics += row.get("physics_steps", 0)
    lower = p["prior"]["physics_steps"] + eval_physics + training_lower
    upper = p["prior"]["physics_steps"] + eval_physics + training_upper
    assert (
        lower == r["main_physics_steps_lower_bound"]
        and upper == r["main_physics_steps_upper_bound"]
    )
    assert (
        upper <= p["caps"]["physics_steps"]
        and replay_physics
        == r["audit_physics_steps"]
        <= p["audit_caps"]["physics_steps"]
    )
    assert r["main_formal_wall_s"] < p["caps"]["wall_s"] or r["status"] == "halted"
    assert r["audit_wall_s"] < p["audit_caps"]["wall_s"] or r["status"] == "halted"
    assert r["formal_optimizer_updates"] == sum(
        x.get("optimizer_updates", 0) for x in r["training"]
    )
    assert r["boxing_games"] == 0
    conclusion = paired_conclusion(r["evaluations"])
    if r["status"] == "completed":
        assert all(
            x["status"] == "completed"
            for group in ("evaluations", "training", "replays")
            for x in r[group]
        )
        assert r["conclusion"] == conclusion and r["physics_accounting_exact"]
    raw = sum(v["bytes"] for v in manifest["files"].values())
    assert raw + p["prior"]["raw_charge_bytes"] <= p["caps"]["raw_bytes"]
    return {
        "passed": True,
        "status": r["status"],
        "case_counts": status_counts(r),
        "main_physics_lower": lower,
        "main_physics_upper": upper,
        "audit_replay_physics": replay_physics,
        "new_independent_audit_physics": 0,
        "raw_bytes_manifested": raw,
        "conclusion": conclusion,
        "findings": findings,
        "pilot_directional_win": bool(
            r["status"] == "completed"
            and conclusion["evaluation_directional_criteria_passed"]
        ),
        "scope": "All saved held-out commands/signed contact/load/posture intervals independently read; first failure persistence/reward/masks/survival reconstructed. Training reset/chunk/reward/update/checkpoint records checked. Final cached observation normalized from frozen stats; all other per-control normalized observation hashes are executed-source assertions checked by the two required exact replay tails, not independently recomputed without unsaved raw caches. Partial/fault outcomes retained and never promoted. No new teacher inference or physics.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    write_json(args.run / "independent-audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, indent=2))
