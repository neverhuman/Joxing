"""Five preregistered frozen-policy shared motor attempts; no optimization.

Formal source publication and the supervisor's complete Linux gate are required
before entry. The original native results are dependencies, never rerun here.
"""

import copy
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from .evidence_storage import read_arrays, read_json
from .motor_storage import RawCredit, append_json, write_json
from .provenance import digest, manifest
from .shared_motor_evidence import MotorTape
from .shared_motor_inputs import FrozenPairTeacher
from .shared_motor_saved_audit import audit_saved_tape, require
from .shared_motor_task import NativeReferenceDriver, SharedMotorTask, build_motor_scene
from .upstream_motor import NativeTask

PACKET = "experiments/v0.0.11/shared-motor-physical-protocol.json"
RUN = "runs/v0.0.11/shared-motor-proof-01"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def process_identity():
    stat = Path("/proc/self/stat")
    return {
        "pid": os.getpid(),
        "start_ticks": int(stat.read_text().rsplit(")", 1)[1].split()[19])
        if stat.exists()
        else None,
    }


def planned_rows():
    return [
        {
            "id": f"source-{seed}",
            "seed": seed,
            "kind": "source",
            "start_tick": 0,
            "end_tick": 3520,
            "planned_physics": 3520,
        }
        for seed in range(71000, 71004)
    ] + [
        {
            "id": "replay-71000",
            "seed": 71000,
            "kind": "replay",
            "start_tick": 203,
            "end_tick": 3520,
            "planned_physics": 3317,
        }
    ]


def validate_packet(p):
    require(p["schema"] == "joxing-shared-motor-physical-v1", "Motor proof schema")
    require(
        p["rows"] == planned_rows() and p["planned_physics"] == 17397,
        "Motor fixed five rows",
    )
    require(
        p["physics_cap"] == 60000
        and p["runner_wall_s"] == 180
        and p["post_gate_wall_s"] == 300,
        "Motor time/physics caps",
    )
    require(
        p["raw_bytes"] == 32 * 1024**2
        and p["formal_simulation_workers"] == p["threads_per_process"] == 1,
        "Motor raw/worker caps",
    )
    require(
        p["prior_v11_physics"] == 5390
        and p["prior_v11_raw_bytes"] == 202302252
        and p["prior_v11_post_gate_upper_s"] == 600,
        "Motor prior accounting",
    )
    require(
        p["prior_v11_physics"] + p["physics_cap"] <= 120000
        and p["prior_v11_raw_bytes"] + p["raw_bytes"] <= 256 * 1024**2
        and p["prior_v11_post_gate_upper_s"] + p["post_gate_wall_s"] <= 900,
        "Motor original whole-version caps",
    )
    require(
        p["decoder_amplitude"] == 0.075 and p["source_host"] == "xbabe1",
        "Motor source host/decoder",
    )


def validate_test_summary(p, text):
    require(
        p["expected_xfail_nodes"]
        == [
            "grokbot/tests/test_referee_fuzz.py::test_restore_does_not_alias_the_snapshot"
        ],
        "Motor fixed known referee failure",
    )
    require(
        re.findall(r"(\d+) passed", text) == [str(p["expected_tests"])],
        "Motor registered test pass count",
    )
    require(
        re.findall(r"(?m)^XFAIL ([^ ]+)", text) == p["expected_xfail_nodes"],
        "Motor exact known strict xfail set",
    )
    require(
        re.findall(r"(\d+) xfailed", text) == [str(len(p["expected_xfail_nodes"]))],
        "Motor known xfail count",
    )


def verify_frozen(p, root, source_sha):
    root = Path(root)
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        == source_sha,
        "Motor frozen HEAD",
    )
    require(
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=root, check=False
        ).returncode
        == 0,
        "Motor source diff",
    )
    for key in ("source_files", "upstream_files", "reference_files"):
        for name, sha in p[key].items():
            require(file_sha(root / name) == sha, "Motor frozen file: " + name)
    inputs = Path(p["inputs_root"])
    require(file_sha(inputs / "model.zip") == p["model_sha256"], "Motor frozen weights")
    require(
        file_sha(inputs / "normalizer.pkl") == p["normalizer_sha256"],
        "Motor frozen normalizer file",
    )


def read_packet(root, protocol_sha, source_sha):
    root = Path(root)
    p = json.loads((root / PACKET).read_text())
    validate_packet(p)
    require(digest(p) == protocol_sha, "Motor protocol checksum")
    require(
        json.loads(
            subprocess.check_output(
                ["git", "show", "HEAD:" + PACKET], cwd=root, text=True
            )
        )
        == p,
        "Motor uncommitted protocol",
    )
    verify_frozen(p, root, source_sha)
    return p


def new_task(p, teacher, scene, seed):
    drivers = {
        a: NativeReferenceDriver(NativeTask(p["environment"]), seed) for a in ("A", "B")
    }
    for d in drivers.values():
        require(d.task.physics_total == 0, "Native reset donor stepped")
        require(
            math.ceil(d.terminate_time * d.cycles / 0.02) * 20 == 3520,
            "Native short horizon changed",
        )
        old = next(r for r in p["native_dependencies"] if r["seed"] == seed)
        require(
            d.init_time == old["initial_phase_s"],
            "Native reset reference phase changed",
        )
    return SharedMotorTask(
        scene, drivers, teacher, horizon_ticks=3520, bundle_hash=p["bundle_sha256"]
    )


def exact_tail(source_trace, source, replay_trace, replay):
    """Compare every retained physical/decision/cache tail, no new physics."""
    start = replay_trace["initial"]["tick"]
    require(
        source_trace["mid"]["task"] == replay_trace["initial"],
        "Replay actual mid engine/cache/control",
    )
    require(
        source_trace["mid"]["support"] == replay_trace["support_prefix"],
        "Replay actual support prefix",
    )
    require(
        source_trace["last_valid"] == replay_trace["last_valid"],
        "Replay final application snapshot",
    )
    require(
        source_trace["state_sha256"][start:] == replay_trace["state_sha256"],
        "Replay every engine hash",
    )
    for k in ("ticks", "end_qpos", "posture", "loads"):
        require(
            np.array_equal(source[k][start:], replay[k]), "Replay physical tail: " + k
        )
    offset = int(source["contact_offsets"][start])
    require(
        np.array_equal(
            source["contact_offsets"][start:] - offset, replay["contact_offsets"]
        ),
        "Replay contact offsets",
    )
    for k in ("contacts", "contact_geoms"):
        require(
            np.array_equal(source[k][offset:], replay[k]),
            "Replay signed contacts: " + k,
        )
    di = int(np.searchsorted(source["decision_ticks"], start))
    for k in source:
        if k in (
            "decision_ticks",
            "decision_engines",
            "raw",
            "normalized",
            "commands",
            "excitation",
        ) or k.startswith("witness_"):
            require(
                np.array_equal(source[k][di:], replay[k]),
                "Replay actual decisions/cache: " + k,
            )
    require(
        source_trace["decisions"][di:] == replay_trace["decisions"],
        "Replay native reference tail",
    )
    return {
        "passed": True,
        "physical_steps": len(replay["ticks"]),
        "new_audit_physics": 0,
        "held_steps_without_new_decision": replay_trace["initial"]["remaining"],
    }


def run_proof(p, root, out, *, source_sha, protocol_sha):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(
        out, p["raw_bytes"] - 512 * 1024
    )  # Supervisor/log share stays inside stage cap.
    started, physics = time.monotonic(), 0
    rows = [
        r | {"status": "not_run", "reason": "pending", "physics_attempted": 0}
        for r in p["rows"]
    ]
    result = {
        "schema": "joxing-shared-motor-proof-result-v1",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "status": "running",
        "rows": rows,
        "physics_attempted": 0,
        "optimizer_updates": 0,
        "valid_native_boxing_games": 0,
        "command": list(__import__("sys").argv),
        "process": process_identity(),
    }
    write_json(out / "protocol.json", p)
    write_json(out / "result.json", result)  # Allocate denominator BEFORE model load.
    task = tape = current = None

    def progress():
        write_json(
            out / "progress.json",
            {
                "physics_attempted": physics,
                "wall_s": time.monotonic() - started,
                "row": None if current is None else current["id"],
                "row_physics_attempted": None
                if current is None
                else current["physics_attempted"],
                "tick": None if task is None else task.tick,
                "status": result["status"],
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
        )

    def guard(n):
        nonlocal physics
        require(n == 1, "Motor one physical entry")
        if (
            time.monotonic() - started >= p["runner_wall_s"]
            or physics + n > p["physics_cap"]
        ):
            raise TimeoutError("Motor first time/physics cap")
        credit.check(out / "progress.json", 2048)
        physics += n
        current["physics_attempted"] += n
        progress()  # Persist attempted entry BEFORE mj_step1/mj_step2, even on fault.

    try:
        verify_frozen(p, root, source_sha)
        write_json(out / "provenance.json", manifest())
        teacher = FrozenPairTeacher.from_inputs(p["inputs_root"])
        require(
            teacher.normalizer.fingerprint == p["normalizer_stats_sha256"],
            "Motor normalizer stats",
        )
        scene = build_motor_scene()
        write_json(out / "scene.json", scene.report)
        first_mid = None
        for current in rows:
            verify_frozen(p, root, source_sha)
            current.update(status="running", reason=None)
            write_json(out / "result.json", result)
            append_json(
                out / "events.jsonl",
                {"row": current["id"], "status": "running", "physics": physics},
            )
            task = new_task(p, teacher, scene, current["seed"])
            if current["kind"] == "replay":
                require(first_mid is not None, "Actual first-source checkpoint missing")
                task.restore(first_mid["task"])
            tape = MotorTape(
                task,
                recording=current["id"] == "source-71000",
                continuation=None
                if current["kind"] == "source"
                else first_mid["support"],
            )
            while not task.done:
                if not task.remaining:
                    tape.collect_decision(task)
                contacts = task.advance_substep(guard)
                tape.capture_substep(task, contacts)
                if current["id"] == "source-71000" and task.tick == 203:
                    first_mid = {
                        "task": task.snapshot(),
                        "support": tape.support_snapshot(task),
                    }
                    tape.mid = copy.deepcopy(first_mid)
                if task.tick % 20 == 0:
                    progress()
            tape.finish_recording(task)
            current["evidence"] = tape.save(task, out / current["id"])
            require(
                all(d.task.physics_total == 0 for d in task.drivers.values()),
                "Reset donor advanced",
            )
            current.update(
                status="completed",
                reason="registered_horizon",
                elapsed_sim_s=float(task.data.time),
            )
            append_json(
                out / "events.jsonl",
                {"row": current["id"], "status": "completed", "physics": physics},
            )
            write_json(out / "result.json", result)
            for d in task.drivers.values():
                d.task.close()
            task = tape = None
        verify_frozen(p, root, source_sha)
        require(physics == 17397, "Motor complete physics denominator")
        result["status"] = "completed"
    except Exception as error:  # noqa: BLE001 -- preserve every failed attempt and all later slots
        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        if current is not None:
            current.update(
                status="invalid" if task is not None and task.invalid else "incomplete",
                reason=result["error"],
            )
        with credit.emergency():
            write_json(
                out / "fault.json",
                {
                    "error": result["error"],
                    "traceback": traceback.format_exc(),
                    "physics_attempted": physics,
                    "last_valid": None
                    if tape is None
                    else {
                        "tick": tape.start_tick + len(tape.intervals),
                        "state_sha256": tape.hashes[-1] if tape.hashes else None,
                        "engine": tape.last_valid_engine.tolist(),
                        "initial": tape.initial,
                        "last_decision": None
                        if not tape.decisions
                        else {
                            "tick": tape.decisions[-1]["tick"],
                            "references": tape.decisions[-1]["references"],
                            "commands": tape.decisions[-1]["commands"].tolist(),
                            "raw": tape.decisions[-1]["raw"].tolist(),
                        },
                    },
                },
            )
            if (
                task is not None
                and tape is not None
                and not (out / current["id"]).exists()
            ):
                try:
                    tape.finish_recording(task)
                    tape.save(task, out / current["id"])
                except Exception as save_error:  # noqa: BLE001 -- retain failed temporary bytes
                    write_json(out / "fault-save.json", {"error": repr(save_error)})
    finally:
        if task is not None:
            for d in task.drivers.values():
                d.task.close()
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "stopped after earlier failure"
        result.update(
            physics_attempted=physics,
            wall_s=time.monotonic() - started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        with credit.emergency():
            write_json(out / "result.json", result)
            progress()
    return result


def audit_proof(p, root, out, *, source_sha, protocol_sha):
    out = Path(out)
    verify_frozen(p, root, source_sha)
    result = json.loads((out / "result.json").read_text())
    require(
        result["source_commit"] == source_sha
        and result["protocol_sha256"] == protocol_sha,
        "Motor result provenance",
    )
    require(
        [
            {k: r[k] for k in spec}
            for r, spec in zip(result["rows"], planned_rows(), strict=True)
        ]
        == planned_rows(),
        "Motor result row denominator",
    )
    require(
        sum(r["physics_attempted"] for r in result["rows"])
        == result["physics_attempted"],
        "Motor attempted physics accounting",
    )
    teacher = FrozenPairTeacher.from_inputs(p["inputs_root"])
    scene = build_motor_scene()
    findings, traces, arrays = [], {}, {}
    for row in result["rows"]:
        if row["status"] != "completed":
            findings.append(
                {
                    "id": row["id"],
                    "passed": False,
                    "status": row["status"],
                    "reason": row["reason"],
                }
            )
            continue
        task = new_task(p, teacher, scene, row["seed"])
        try:
            folder = out / row["id"]
            trace, data = (
                read_json(folder / "trace.json"),
                read_arrays(folder / "tape.npz"),
            )
            measurement = audit_saved_tape(task, folder)
            require(
                len(data["ticks"])
                == row["physics_attempted"]
                == row["planned_physics"],
                "Motor valid interval denominator",
            )
            # Independent scalar calls on actual saved normalized inputs; no stepping.
            for i in range(len(data["commands"])):
                for pi in range(2):
                    predicted, hidden = teacher.model.predict(
                        data["normalized"][i, pi], deterministic=True
                    )
                    require(
                        hidden is None
                        and np.array_equal(predicted, data["commands"][i, pi]),
                        "Motor saved policy command parity",
                    )
            teacher.normalizer.check()
            traces[row["id"]], arrays[row["id"]] = trace, data
            findings.append(
                {"id": row["id"], **measurement, "policy_inference_audited": True}
            )
        finally:
            for d in task.drivers.values():
                d.task.close()
    replay = (
        exact_tail(
            traces["source-71000"],
            arrays["source-71000"],
            traces["replay-71000"],
            arrays["replay-71000"],
        )
        if "replay-71000" in traces
        else {"passed": False, "reason": "replay not completed"}
    )
    complete = (
        result["status"] == "completed"
        and len(findings) == 5
        and all(f["passed"] for f in findings)
        and replay["passed"]
    )
    motor = complete and all(
        all(v is None for v in f["physical_failure"].values())
        and all(v is None for v in f["tracking_failure"].values())
        for f in findings[:4]
    )
    verify_frozen(p, root, source_sha)
    return {
        "schema": "joxing-shared-motor-independent-audit-v1",
        "passed": complete,
        "short_motor_gate_passed": motor,
        "findings": findings,
        "exact_replay": replay,
        "new_audit_physics": 0,
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "optimizer_updates": 0,
        "valid_native_boxing_games": 0,
        "scope": "Four correlated pair worlds; short shared locomotion transfer, not stationary guard or boxing",
    }
