"""Frozen H33primary or fresh-process audit, with pre-entry durable step credit."""

import argparse
import gc
import json
import signal
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--stage", required=True, choices=("physical", "audit"))
    a = ap.parse_args()
    p = json.loads((ROOT / a.packet).read_text())
    from joxing.native_pair_motor_evidence import EvidenceBudget
    from joxing.native_pair_motor_rl import NativePairMotorEnv, PairMotorConfig
    from joxing.native_pair_motor_train import prove_native_replay
    from joxing.native_stance_probe import (
        CONDITIONS,
        SEEDS,
        audit_stance,
        condition_conclusion,
        evaluate_stance,
        storage_forecast,
    )
    from joxing.provenance import digest
    from joxing.wave import canonical_actor_hash

    if (
        p["schema"] != "joxing-native-stance-probe-v1"
        or [(c["id"], c["placement"], c["reference_profile"]) for c in p["conditions"]]
        != list(CONDITIONS)
        or p["evaluation_seeds"] != list(SEEDS)
        or p["horizon_ticks"] != 3000
    ):
        raise ValueError("Frozen finite stopping study required")
    out = ROOT / p["output"]
    if (
        storage_forecast() != p["storage_forecast"]
        or p["storage_forecast"]["maximum_raw_with_reserves"] > p["raw_bytes"]
    ):
        raise ValueError("Frozen prospective native evidence forecast")
    budget = EvidenceBudget(
        out, raw_limit=p["raw_bytes"], text_limit=p["receipt_bytes"]
    )
    target = out / a.stage
    target.mkdir(exist_ok=False)
    primary = (
        None
        if a.stage == "physical"
        else json.loads((out / "physical/result.json").read_text())
    )
    if primary is not None and primary["status"] != "completed":
        raise ValueError("Audit only a complete primary; failed prefixes retained")
    before = 0 if primary is None else primary["entered_physics"]
    entered = 0
    started = time.monotonic()
    ledger = (
        target / ("primary-entered.bin" if primary is None else "audit-entered.bin")
    ).open("xb+", buffering=0)
    ledger.write(struct.pack("<QQ", 0, (1 << 64) - 1))
    wall = p["native_wall_s"] if primary is None else p["audit_wall_s"]

    def guard(n):
        nonlocal entered
        if (
            before + entered + n + p["source_fixture_physics_reserve"]
            > p["physics_steps"]
        ):
            raise RuntimeError("Stance total entered physics first limit")
        if time.monotonic() - started > wall - 5:
            raise TimeoutError("Stance stage first limit")
        if n:
            entered += n
            ledger.seek(0)
            ledger.write(struct.pack("<QQ", entered, entered ^ ((1 << 64) - 1)))

    def interrupted(*_):
        raise TimeoutError("Owned stance stage interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    result = {
        "schema": p["schema"],
        "stage": a.stage,
        "status": "running",
        "protocol_sha256": digest(p),
        "rows": [
            {"id": c["id"] + "/" + str(s), "status": "not_run"}
            for c in p["conditions"]
            for s in SEEDS
        ],
        "proofs": [],
        "proof_rows": [{"id": c["id"], "status": "not_run"} for c in p["conditions"]],
        "trials": [],
        "conclusions": {},
        "entered_physics": 0,
        "optimizer_updates": 0,
        "boxing_qualified": False,
    }
    world = None

    def save():
        result.update(entered_physics=entered, wall_s=time.monotonic() - started)
        # Immutable trial/proof files retain the outcomes. Only this small
        # current progress receipt is atomically replaced, with both old and
        # temporary copies charged before writing.
        temporary = target / "result.json.tmp"
        budget.write_json(temporary, result)
        temporary.replace(target / "result.json")

    try:
        save()
        for c in p["conditions"]:
            world = NativePairMotorEnv.from_inputs(
                Path(p["teacher_inputs"]),
                config=PairMotorConfig(placement=c["placement"], horizon_ticks=3000),
                guard=guard,
                reference_profile=c["reference_profile"],
            )
            if world.bundle != p["expected_teacher_bundle"]:
                raise ValueError("Original actor/normalizer/reference changed")
            folder = target / c["id"]
            folder.mkdir()
            proof_row = next(r for r in result["proof_rows"] if r["id"] == c["id"])
            proof_row["status"] = "unconfirmed"
            save()
            proof = {
                "condition": c["id"],
                **prove_native_replay(world, seed=SEEDS[0], output=folder),
            }
            if primary is not None:
                old_proof = next(
                    r for r in primary["proofs"] if r["condition"] == c["id"]
                )
                if digest(proof) != digest(old_proof):
                    raise ValueError("Independent actual held-command proof mismatch")
            result["proofs"].append(proof)
            proof_row["status"] = "completed"
            save()
            trials = []
            for seed in SEEDS:
                row = next(
                    r for r in result["rows"] if r["id"] == c["id"] + "/" + str(seed)
                )
                row["status"] = "unconfirmed"
                save()
                if primary is None:
                    item = evaluate_stance(
                        world,
                        seed=seed,
                        output=folder / f"seed-{seed}",
                        budget=budget,
                        guard=guard,
                    )
                    item["condition"] = c["id"]
                else:
                    old = next(
                        r
                        for r in primary["trials"]
                        if r["condition"] == c["id"] and r["seed"] == seed
                    )
                    item = audit_stance(
                        world,
                        folder=out / "physical" / c["id"] / f"seed-{seed}",
                        expected={k: v for k, v in old.items() if k != "condition"},
                        guard=guard,
                    )
                    item["condition"] = c["id"]
                trials.append(item)
                result["trials"].append(item)
                row["status"] = "completed"
                save()
            if primary is None:
                result["conclusions"][c["id"]] = condition_conclusion(trials)
            if (
                canonical_actor_hash(world.teacher.model.actor)
                != p["expected_teacher_bundle"]["actor"]
            ):
                raise ValueError("Frozen pretrained actor changed")
            world.close()
            world = None
            gc.collect()
        if primary is not None:
            expected_steps = sum(r["final_tick"] for r in primary["trials"])
            expected_steps += sum(r["physics"] for r in primary["proofs"])
            if entered != expected_steps:
                raise ValueError("Complete independent replay accounting")
            result["conclusions"] = primary["conclusions"]
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        for r in [*result["rows"], *result["proof_rows"]]:
            if r["status"] == "unconfirmed":
                r["status"] = "invalid"
        raise
    finally:
        if world is not None:
            world.close()
        ledger.close()
        save()
    print(
        json.dumps(
            {
                "stage": a.stage,
                "status": result["status"],
                "entered_physics": entered,
                "conclusions": result["conclusions"],
            }
        )
    )


if __name__ == "__main__":
    main()
