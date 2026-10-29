"""Recorded-command qualification only; no new inference, dynamics or learning."""

import argparse
import hashlib
import json
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from .motor_storage import RawCredit, append_json, write_json
from .native_face_qualification import ROOT, digest, no_integration, rss_bytes

PACKET = "experiments/v0.0.11/upper-residual-protocol.json"
RUN = "runs/v0.0.11/upper-residual-01"
GAME = "offline-recorded-motor-source71000"


def packet(protocol_sha, source_sha):
    import sys

    p = json.loads((ROOT / PACKET).read_text())
    if (
        p["schema"] != "joxing-upper-residual-offline-v1"
        or digest(p) != protocol_sha
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != source_sha
        or json.loads(
            subprocess.check_output(["git", "show", "HEAD:" + PACKET], cwd=ROOT)
        )
        != p
    ):
        raise ValueError("Upper residual frozen source/protocol")
    for name, d in {**p["source_files"], **p["input_files"]}.items():
        path = Path(name) if name.startswith("/") else ROOT / name
        if (
            path.stat().st_size != d["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != d["sha256"]
        ):
            raise ValueError("Upper residual source/input identity: " + name)
    if (
        sys.version.split()[0] != p["python_version"]
        or np.__version__ != p["numpy_version"]
        or mujoco.__version__ != p["mujoco_version"]
    ):
        raise ValueError("Upper residual pinned runtime")
    return p


def inputs(p):
    from .upper_residual import UpperResidualSpec

    config = json.loads(Path(p["checkpoint_config"]).read_text())
    if config["env_name"] != "msgym/LocomotionFullEnv-v1":
        raise ValueError("Expected frozen full-body task config")
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "third_party/ms-human-700/MS-Human-700.xml")
    )
    groups = config["agent_kwargs"]["policy_kwargs"]["dynsyn"]
    spec = UpperResidualSpec.from_native_model(
        model, groups, ROOT / "third_party/ms-human-700"
    )
    if (
        len(spec.selected_ids),
        len(spec.protected_ids),
        len(spec.active_groups),
        len(spec.mixed_groups),
    ) != (122, 578, 46, 0):
        raise ValueError("Registered native arm-file ownership")
    with np.load(p["teacher_tape"], allow_pickle=False, max_header_size=4096) as z:
        commands, ticks = z["commands"], z["decision_ticks"]
    if (
        commands.dtype != np.float32
        or commands.shape != (176, 2, 700)
        or ticks.tolist() != list(range(0, 3520, 20))
        or not np.isfinite(commands).all()
        or np.any(abs(commands) > 1)
    ):
        raise ValueError("Original saved native teacher command extent")
    return spec, commands, ticks, groups


def cases(p, commands, ticks, dimension):
    base = {
        "zero": np.zeros(dimension, np.float32),
        "positive": np.ones(dimension, np.float32),
        "negative": -np.ones(dimension, np.float32),
        "alternating": np.array(
            [1 if i % 2 == 0 else -1 for i in range(dimension)], np.float32
        ),
        "ramp": np.linspace(-1, 1, dimension, dtype=np.float32),
        "fixed-uniform": np.random.default_rng(p["residual_seed"])
        .uniform(-1, 1, dimension)
        .astype(np.float32),
    }
    for i, tick in enumerate(ticks):
        for j, actor in enumerate("AB"):
            for name, residual in base.items():
                yield (
                    f"{int(tick)}/{actor}/{name}",
                    actor,
                    int(tick),
                    commands[i, j],
                    residual,
                )
    for j, actor in enumerate("AB"):
        for k in range(dimension):
            residual = np.zeros(dimension, np.float32)
            residual[k] = 1
            yield f"0/{actor}/basis-{k}", actor, 0, commands[0, j], residual


def prove(p, source_sha, protocol_sha):
    from .upper_residual import HeldUpperCommand, array_sha
    from .upper_residual_evidence import (
        descriptor_check,
        expected_command,
        inspect_snapshot,
    )

    out = ROOT / RUN
    out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(out, p["raw_bytes"] - p["supervision_reserve_bytes"])
    result = {
        "schema": "joxing-upper-residual-qualification-v1",
        "status": "running",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "new_physics": 0,
        "new_optimizer_updates": 0,
        "new_teacher_inferences": 0,
        "physical_balance_qualified": False,
        "rows": [{"id": key, "status": "not_run"} for key in p["case_ids"]],
    }
    began = time.monotonic()
    current = None
    write_json(out / "result.json", result)
    try:
        with no_integration():
            spec, commands, ticks, groups = inputs(p)
            descriptor = spec.descriptor()
            binding = descriptor_check(
                descriptor,
                source_root=ROOT / "third_party/ms-human-700",
                checkpoint_groups=groups,
            )
            write_json(
                out / "mask-binding.json",
                {"descriptor": descriptor, "sha256": spec.sha256, "ownership": binding},
            )
            for index, (key, actor, tick, teacher, residual) in enumerate(
                cases(p, commands, ticks, len(spec.active_groups))
            ):
                current = result["rows"][index]
                if key != current["id"]:
                    raise ValueError("Upper residual preregistered case order")
                if (
                    time.monotonic() - began > p["stage_wall_limits_s"]["proof"]
                    or rss_bytes() > p["maximum_rss_bytes"]
                ):
                    raise TimeoutError("Upper residual proof wall/RSS")
                current["status"] = "running"
                append_json(
                    out / "events.jsonl", {"case_index": index, "phase": "entered"}
                )
                record = HeldUpperCommand.create(
                    teacher,
                    residual,
                    spec,
                    actor=actor,
                    game_id=GAME,
                    sampled_tick=tick,
                )
                expected = expected_command(teacher, residual, descriptor)
                protected = list(spec.protected_ids)
                if (
                    record.command_bytes != expected.tobytes()
                    or expected[protected].tobytes() != teacher[protected].tobytes()
                    or (
                        not np.any(residual)
                        and record.command_bytes != teacher.tobytes()
                    )
                ):
                    raise ValueError("Upper residual command/protected/zero bit parity")
                saved = json.loads(json.dumps(record.snapshot(), allow_nan=False))
                inspect_snapshot(
                    saved,
                    descriptor,
                    teacher,
                    residual,
                    expected,
                    actor=actor,
                    tick=tick,
                    game_id=GAME,
                )
                restored = HeldUpperCommand.restore(
                    saved, spec, actor=actor, game_id=GAME, sampled_tick=tick
                )
                if (
                    restored.command_at(
                        actor=actor, game_id=GAME, engine_tick=tick + 3
                    ).tobytes()
                    != record.command_bytes
                    or restored.command_at(
                        actor=actor, game_id=GAME, engine_tick=tick + 19
                    ).tobytes()
                    != record.command_bytes
                ):
                    raise ValueError("Upper residual held-record roundtrip")
                try:
                    restored.command_at(
                        actor=actor, game_id=GAME, engine_tick=tick + 20
                    )
                except ValueError:
                    pass
                else:
                    raise ValueError("Upper residual stale hold accepted")
                append_json(
                    out / "cases.jsonl",
                    {
                        "case_index": index,
                        "teacher_sha256": array_sha(teacher),
                        "residual_sha256": array_sha(residual),
                        "command_sha256": hashlib.sha256(
                            record.command_bytes
                        ).hexdigest(),
                        "status": "passed",
                    },
                )
                append_json(
                    out / "events.jsonl", {"case_index": index, "phase": "completed"}
                )
                current["status"] = "passed"
                if index == 0:
                    write_json(out / "held-example.json", saved)
                if (index + 1) % 64 == 0:
                    write_json(out / "result.json", result)
            if index + 1 != len(p["case_ids"]):
                raise ValueError("Upper residual complete denominator")
        result.update(
            status="passed",
            ownership=binding,
            spec_sha256=spec.sha256,
            case_count=len(p["case_ids"]),
            zero_cases=352,
            physical_replay_qualified=False,
        )
    except Exception as error:  # noqa: BLE001 -- retain every unsuccessful attempt
        import traceback

        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        if current is not None and current["status"] == "running":
            current.update(status="failed", reason=result["error"])
        with credit.emergency():
            write_json(
                out / "fault.json",
                {"error": result["error"], "traceback": traceback.format_exc()},
            )
    finally:
        result.update(
            wall_s=time.monotonic() - began,
            max_rss_bytes=rss_bytes(),
            counts=dict(Counter(r["status"] for r in result["rows"])),
        )
        with credit.emergency():
            write_json(out / "result.json", result)
    return result


def audit(p, source_sha, protocol_sha):
    from .upper_residual_evidence import (
        descriptor_check,
        expected_command,
        inspect_snapshot,
    )

    began = time.monotonic()
    out = ROOT / RUN
    result = json.loads((out / "result.json").read_text())
    binding = json.loads((out / "mask-binding.json").read_text())
    descriptor = binding["descriptor"]
    if (
        result["status"] != "passed"
        or result["source_commit"] != source_sha
        or result["protocol_sha256"] != protocol_sha
        or digest(descriptor) != binding["sha256"]
    ):
        raise ValueError("Upper residual saved run binding")
    # Reader does not instantiate the candidate spec or invoke its compose/restore.
    groups = json.loads(Path(p["checkpoint_config"]).read_text())["agent_kwargs"][
        "policy_kwargs"
    ]["dynsyn"]
    ownership = descriptor_check(
        descriptor,
        source_root=ROOT / "third_party/ms-human-700",
        checkpoint_groups=groups,
    )
    with np.load(p["teacher_tape"], allow_pickle=False, max_header_size=4096) as z:
        commands, ticks = z["commands"], z["decision_ticks"]
    rows = [json.loads(s) for s in (out / "cases.jsonl").read_text().splitlines()]
    events = [json.loads(s) for s in (out / "events.jsonl").read_text().splitlines()]
    expected_events = [
        {"case_index": i, "phase": phase}
        for i in range(len(p["case_ids"]))
        for phase in ("entered", "completed")
    ]
    if (
        events != expected_events
        or len(rows) != len(p["case_ids"])
        or result["rows"] != [{"id": key, "status": "passed"} for key in p["case_ids"]]
    ):
        raise ValueError("Upper residual saved case/event denominator")
    with no_integration():
        for i, (key, actor, tick, teacher, residual) in enumerate(
            cases(p, commands, ticks, ownership["action_dimension"])
        ):
            expected = expected_command(teacher, residual, descriptor)
            record = {
                "case_index": i,
                "teacher_sha256": hashlib.sha256(teacher.tobytes()).hexdigest(),
                "residual_sha256": hashlib.sha256(residual.tobytes()).hexdigest(),
                "command_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                "status": "passed",
            }
            if rows[i] != record or key != p["case_ids"][i]:
                raise ValueError("Independent upper residual saved output mismatch")
            if i == 0:
                inspect_snapshot(
                    json.loads((out / "held-example.json").read_text()),
                    descriptor,
                    teacher,
                    residual,
                    expected,
                    actor=actor,
                    tick=tick,
                    game_id=GAME,
                )
    if (
        time.monotonic() - began > p["stage_wall_limits_s"]["audit"]
        or rss_bytes() > p["maximum_rss_bytes"]
    ):
        raise TimeoutError("Upper residual independent audit wall/RSS")
    report = {
        "schema": "joxing-upper-residual-independent-v1",
        "status": "passed",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "all_planned_rows_audited": True,
        "rows": len(rows),
        "zero_cases": 352,
        "ownership": ownership,
        "candidate_compose_or_restore_called": False,
        "wall_s": time.monotonic() - began,
        "max_rss_bytes": rss_bytes(),
        "new_physics": 0,
        "new_optimizer_updates": 0,
        "physical_balance_qualified": False,
    }
    RawCredit(out, p["raw_bytes"] - p["supervision_reserve_bytes"])
    write_json(out / "independent-audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("proof", "audit"), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise TimeoutError("Owned upper residual task interrupted: " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    p = packet(args.protocol_sha256, args.source_sha)
    result = (prove if args.stage == "proof" else audit)(
        p, args.source_sha, args.protocol_sha256
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
