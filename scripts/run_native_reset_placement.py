"""H34finite fixed-start trials or independent replay, with entered-step ledgers."""

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
    from joxing.native_pair_motor_train import prove_native_replay
    from joxing.native_reset_placement import (
        MODES,
        SEEDS,
        ResetPlacementFactory,
        reset_geometry,
    )
    from joxing.native_stance_probe import (
        audit_stance,
        evaluate_stance,
        storage_forecast,
    )
    from joxing.provenance import digest
    from joxing.wave import canonical_actor_hash

    if (
        p["schema"] != "joxing-native-reset-placement-study-v1"
        or [c["id"] for c in p["conditions"]] != list(MODES)
        or p["evaluation_seeds"] != list(SEEDS)
        or p["horizon_ticks"] != 3000
    ):
        raise ValueError("Frozen finite reset study required")
    forecast = storage_forecast()
    if (
        p["storage_forecast"]["frame_archive_bytes"] != forecast["frame_archive_bytes"]
        or p["storage_forecast"]["trace_archive_bytes"]
        != forecast["trace_archive_bytes"]
        or p["storage_forecast"]["maximum_raw_with_reserves"] > p["raw_bytes"]
    ):
        raise ValueError("Prospective archive/reservation forecast")
    out = ROOT / p["output"]
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
        raise ValueError("Complete primary prerequisite")
    before = 0 if primary is None else primary["entered_physics"]
    if primary is not None:
        count, complement = struct.unpack(
            "<QQ", (out / "physical/primary-entered.bin").read_bytes()
        )
        if count != before or complement != count ^ ((1 << 64) - 1):
            raise ValueError("Sealed primary physics ledger")
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
            raise RuntimeError("Reset total physics first limit")
        if time.monotonic() - started > wall - 5:
            raise TimeoutError("Reset stage first limit")
        if n:
            entered += n
            ledger.seek(0)
            ledger.write(struct.pack("<QQ", entered, entered ^ ((1 << 64) - 1)))

    def interrupted(*_):
        raise TimeoutError("Owned reset stage interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    result = {
        "schema": p["schema"],
        "stage": a.stage,
        "status": "running",
        "protocol_sha256": digest(p),
        "rows": [
            {"id": mode + "/" + str(seed), "status": "not_run"}
            for mode in MODES
            for seed in SEEDS
        ],
        "proof_rows": [{"id": mode, "status": "not_run"} for mode in MODES],
        "proofs": [],
        "trials": [],
        "conclusions": {},
        "entered_physics": 0,
        "scientific_optimizer_updates": 0,
        "boxing_qualified": False,
    }
    factory = world = None

    def save():
        result.update(entered_physics=entered, wall_s=time.monotonic() - started)
        temp = target / "result.json.tmp"
        budget.write_json(temp, result)
        temp.replace(target / "result.json")

    try:
        save()
        factory = ResetPlacementFactory(p["teacher_inputs"])
        if factory.bundle != p["expected_teacher_bundle"]:
            raise ValueError("Frozen original teacher/normalizer/reference drift")
        for mode in MODES:
            folder = target / mode
            folder.mkdir()
            trials = []
            for seed in SEEDS:
                row = next(
                    r for r in result["rows"] if r["id"] == mode + "/" + str(seed)
                )
                row["status"] = "unconfirmed"
                save()
                world = factory.make(seed=seed, mode=mode, guard=guard)
                world.reset(seed=seed)
                world._scratch(full=True)
                geometry = reset_geometry(world.scene.model, world.scratch)
                if digest(geometry) != digest(
                    world.scene.report["reset_placement"]["initial_geometry"]
                ):
                    raise ValueError("Actual live reset/scratch geometry mismatch")
                if seed == SEEDS[0]:
                    proof_row = next(r for r in result["proof_rows"] if r["id"] == mode)
                    proof_row["status"] = "unconfirmed"
                    save()
                    proof = {
                        "condition": mode,
                        **prove_native_replay(world, seed=seed, output=folder),
                    }
                    if primary is not None:
                        old_proof = next(
                            r for r in primary["proofs"] if r["condition"] == mode
                        )
                        if digest(proof) != digest(old_proof):
                            raise ValueError(
                                "Independent held native reset proof mismatch"
                            )
                    result["proofs"].append(proof)
                    proof_row["status"] = "completed"
                    save()
                if primary is None:
                    item = evaluate_stance(
                        world,
                        seed=seed,
                        output=folder / f"seed-{seed}",
                        budget=budget,
                        guard=guard,
                    )
                    item["condition"] = mode
                    item["reset_geometry"] = geometry
                    item["reset_placement"] = world.scene.report["reset_placement"]
                    contact = item["metrics"]["first_interactor_contact"]
                    item["first100ms_no_interactor_force"] = (
                        contact is None or contact["tick"] > 100
                    )
                else:
                    old = next(
                        r
                        for r in primary["trials"]
                        if r["condition"] == mode and r["seed"] == seed
                    )
                    # The immutable recorder uses the already qualified H33format;
                    # reset-specific results live in the wrapper, not a rewritten file.
                    expected = {
                        k: v
                        for k, v in old.items()
                        if k
                        not in {
                            "condition",
                            "reset_geometry",
                            "reset_placement",
                            "first100ms_no_interactor_force",
                        }
                    }
                    item = audit_stance(
                        world,
                        folder=out / "physical" / mode / f"seed-{seed}",
                        expected=expected,
                        guard=guard,
                    )
                    if digest(geometry) != digest(old["reset_geometry"]):
                        raise ValueError("Independent reset geometry mismatch")
                    item.update(
                        condition=mode,
                        reset_geometry=geometry,
                        first100ms_no_interactor_force=old[
                            "first100ms_no_interactor_force"
                        ],
                    )
                trials.append(item)
                result["trials"].append(item)
                row["status"] = "completed"
                world.close()
                world = None
                gc.collect()
                save()
            if primary is None:
                result["conclusions"][mode] = {
                    "all_four_geometry_qualified": all(
                        t["reset_geometry"]["clearance_qualified"] for t in trials
                    ),
                    "all_four_first100ms_clear": all(
                        t["first100ms_no_interactor_force"] for t in trials
                    ),
                    "paired_three_second_completions": sum(
                        t["final_tick"] == 3000
                        and not any(t["motor_failures"].values())
                        for t in trials
                    ),
                    "denominator": 4,
                }
            save()
        expected_steps = sum(
            t["final_tick"]
            for t in (result["trials"] if primary is None else primary["trials"])
        ) + sum(t["physics"] for t in result["proofs"])
        if entered != expected_steps:
            raise ValueError("Complete primary/audit physics denominator")
        if canonical_actor_hash(factory.model.actor) != factory.bundle["actor"]:
            raise ValueError("Frozen teacher changed")
        if primary is not None:
            result["conclusions"] = primary["conclusions"]
        result["initialization_qualified"] = all(
            c["all_four_geometry_qualified"] and c["all_four_first100ms_clear"]
            for c in result["conclusions"].values()
        )
        result["walking_control_qualified"] = (
            result["conclusions"][MODES[1]]["paired_three_second_completions"] == 4
        )
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
        if factory is not None:
            factory.close()
        ledger.close()
        save()
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in [
                    "stage",
                    "status",
                    "entered_physics",
                    "conclusions",
                    "initialization_qualified",
                    "walking_control_qualified",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
