"""Qualify a new CPU host against retained v8 native state/actions/contact replay."""

import argparse
import fcntl
import json
import time
import traceback
from pathlib import Path

import numpy as np

from joxing.native_transfer import load_controller, predict
from joxing.physical_motor import PhysicalMotorTask
from joxing.provenance import ROOT, digest, manifest, write_json
from joxing.stance_train import file_hash

PROTOCOL_SHA = "66caf5eb8e90b3775d6894c7d3e86ca54e411eebd1ff514c5ea1797145a0bec2"


def compare_contact_tail(expected_path, actual_path, first):
    with (
        np.load(expected_path, allow_pickle=False) as expected,
        np.load(actual_path, allow_pickle=False) as actual,
    ):
        assert set(expected.files) == set(actual.files)
        start = int(expected["contact_offsets"][first])
        for key in expected.files:
            if key == "contact_offsets":
                reference = expected[key][first:] - start
            elif key in {"intervals_s", "loads_N", "tilt_drop"}:
                reference = expected[key][first:]
            else:
                reference = expected[key][start:]
            if (
                reference.dtype != actual[key].dtype
                or not np.array_equal(reference, actual[key])
                or not np.isfinite(actual[key]).all()
            ):
                raise RuntimeError(f"Retained substep contact/posture mismatch: {key}")
    return True


def run(protocol_path, inputs, out):
    p = json.loads(protocol_path.read_text())
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Registered host qualification changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".host-qualification.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    rows = [
        {"snapshot": boundary, "path": mode, "status": "not_run"}
        for boundary in ("boundary", "mid_control")
        for mode in ("recorded_actions", "closed_loop")
    ]
    report = manifest() | {
        "status": "running",
        "passed": False,
        "protocol_sha256": digest(p),
        "script_sha256": file_hash(Path(__file__)),
        "cases": rows,
        "physics_steps": 0,
        "boxing_games": 0,
    }
    write_json(out / "protocol.json", p)
    write_json(out / "result.json", report)
    task = active = None

    def budget():
        from joxing.wave import check_resources

        if time.perf_counter() - started >= p["qualification_caps"]["wall_s"]:
            raise TimeoutError("Host qualification wall cap")
        if (
            task is not None
            and task.physics_total + task.env.frame_skip
            > p["qualification_caps"]["physics_steps"]
        ):
            raise TimeoutError("Host qualification physics cap")
        if (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            >= p["qualification_caps"]["raw_bytes"]
        ):
            raise TimeoutError("Host qualification raw cap")
        check_resources(out, p["qualification_caps"])

    try:
        import torch

        from joxing.wave import canonical_actor_hash, check_resources

        check_resources(out, p["qualification_caps"], True)
        for path, expected in p["reference_files"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError("Retained v8 reference changed: " + path)
        if (
            file_hash(ROOT / "scripts/benchmark_physical_scaling.py")
            != p["scaling_script_sha256"]
        ):
            raise ValueError("Frozen H2 runner changed")
        config = json.loads((ROOT / "experiments/v0.0.8/protocol.json").read_text())
        base = ROOT / "artifacts/v0.0.8/physical/native_2ms"
        record = json.loads((base / "replay.json").read_text())
        with np.load(base / "actions-74000.npz", allow_pickle=False) as saved:
            actions = saved["actions"].copy()
        model, n = load_controller(inputs)
        model.policy.set_training_mode(False)
        actor_hash = canonical_actor_hash(model.actor)
        task = PhysicalMotorTask(config["environment"], config["variants"][0])
        if task.contract != json.loads((base / "specification.json").read_text()):
            raise RuntimeError("New host compiled contract differs from v8")
        with torch.inference_mode():
            for row in rows:
                active = row
                row["status"] = "running"
                budget()
                write_json(out / "result.json", report)
                snap = (
                    record["snapshot"]
                    if row["snapshot"] == "boundary"
                    else record["mid_snapshot"]
                )
                task.restore(snap)
                first = (
                    snap["physical_control"]["completed_controls"] * task.env.frame_skip
                    + snap["physical_control"]["substep"]
                )
                before = task.physics_total
                index = record["continuation_start"]
                try:
                    if task.pending:
                        if not np.array_equal(task.action, actions[index]):
                            raise RuntimeError("Retained held muscle action differs")
                        budget()
                        task.resume_control()
                        index += 1
                    while not task.done:
                        budget()
                        if index >= len(actions):
                            raise RuntimeError("New host ran beyond source action tape")
                        action = (
                            actions[index]
                            if row["path"] == "recorded_actions"
                            else predict(model, n, task.last_obs)[0]
                        )
                        if not np.array_equal(action, actions[index]):
                            raise RuntimeError(
                                "New host scalar actor differs from source tape"
                            )
                        task.step(action)
                        index += 1
                    final = task.snapshot()
                    final_path = out / (
                        row["snapshot"] + "-" + row["path"] + "-final.json"
                    )
                    write_json(final_path, final)
                    if index != len(actions) or digest(final) != digest(
                        record["final_snapshot"]
                    ):
                        raise RuntimeError(
                            "New host final engine/cache/application snapshot differs"
                        )
                    evidence_path = out / (row["snapshot"] + "-" + row["path"] + ".npz")
                    task.dump_substeps(evidence_path)
                    compare_contact_tail(
                        base / "substeps-74000.npz", evidence_path, first
                    )
                    n.check()
                    if canonical_actor_hash(model.actor) != actor_hash:
                        raise RuntimeError("Actor changed")
                    row.update(
                        status="completed",
                        final_snapshot_exact=True,
                        all_tail_contacts_posture_exact=True,
                        predicted_actions_exact=row["path"] == "closed_loop",
                        first_source_substep=first,
                        raw_substeps_sha256=file_hash(evidence_path),
                        final_snapshot_sha256=file_hash(final_path),
                    )
                finally:
                    row["physics_steps"] = task.physics_total - before
                    report["physics_steps"] = task.physics_total
                    write_json(out / "result.json", report)
        report.update(
            status="completed",
            passed=True,
            actor_hash=actor_hash,
            normalizer_stats_hash=n.fingerprint,
            threads=torch.get_num_threads(),
        )
    except Exception as error:  # noqa: BLE001 -- qualification faults block H2, retained
        if active is not None:
            active.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
            )
        report.update(
            status="halted", error=str(error), traceback=traceback.format_exc()
        )
        if task is not None:
            write_json(
                out / "fault.json", {"reason": str(error), "snapshot": task.snapshot()}
            )
    finally:
        if task is not None:
            report["physics_steps"] = task.physics_total
            task.close()
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "halted_before_case"
        report["wall_s"] = time.perf_counter() - started
        write_json(out / "result.json", report)
        lease.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("protocol", "inputs", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.inputs, args.out)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("status", "passed", "physics_steps", "wall_s", "error")
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)
