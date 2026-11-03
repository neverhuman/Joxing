"""Finite unassisted stopping probe and exact replay evidence, never training.

Freeze the protocol and source publicly before running with real checkpoints.
Synthetic-teacher source fixtures establish software invariants only.
"""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .native_pair_motor_pilot import MotorMetrics
from .native_stance_reference import PROFILE
from .native_tape_storage import load, plan, store
from .provenance import digest

CONDITIONS = (
    ("lanes-walk", "opposed-separated-lanes-v1", None),
    ("lanes-stop400ms", "opposed-separated-lanes-v1", PROFILE),
    ("opposed-stop400ms", "opposed", PROFILE),
)
SEEDS = (71000, 71016, 71032, 71048)
METRIC_COLUMNS = (
    "tilt_rad",
    "pelvis_drop_m",
    "up_world_z",
    "floor_right_N",
    "floor_left_N",
    "floor_nonfoot_N",
    "root_world_x_m",
    "root_world_y_m",
    "root_drift_xy_m",
    "root_speed_xy_mps",
    "right_glove_target_m",
    "left_glove_target_m",
    "right_face_cos_positive",
    "root_yaw_cos_world_x",
    "root_yaw_sin_world_x",
)


def storage_forecast():
    """Exact maximum archive sizes plus prospective non-tape reservations."""
    frames = {
        k: np.zeros(shape, np.float64)
        for k, shape in {
            "qpos": (151, 170),
            "qvel": (151, 170),
            "activation": (151, 1400),
            "control": (151, 1400),
            "time": (151,),
        }.items()
    }
    trace = {
        "teacher_inputs": np.zeros((150, 2, 3601), np.float32),
        "commands": np.zeros((150, 2, 700), np.float32),
        "engine_sha256": np.zeros((150, 32), np.uint8),
        "metrics_1ms": np.zeros((3000, 2, len(METRIC_COLUMNS)), np.float64),
        "contact_sha256_1ms": np.zeros((3000, 32), np.uint8),
    }
    f, t = plan(frames)["archive_bytes"], plan(trace)["archive_bytes"]
    reserves = {
        "source_fixtures": 64 * 1024**2,
        "all_text_including_snapshots_progress_audit_faults": 16 * 1024**2,
        "three_media_stages_including_encoder_temporary_files": 48 * 1024**2,
        "other_temporary_and_fault": 16 * 1024**2,
    }
    return {
        "trials": 12,
        "maximum_frames_per_trial": 151,
        "frame_archive_bytes": f,
        "trace_archive_bytes": t,
        "all_trial_archives_bytes": 12 * (f + t),
        "non_tape_raw_reserves": reserves,
        "maximum_raw_with_reserves": 12 * (f + t) + sum(reserves.values()),
        "text_reserve": reserves["all_text_including_snapshots_progress_audit_faults"],
    }


def contact_digest(contacts):
    """All staged contacts in solver order, with every field and signed zero."""
    h = hashlib.sha256()
    h.update(np.array([len(contacts)], dtype="<i8").tobytes())
    for c in contacts:
        h.update(np.array([c.geom1, c.geom2], dtype="<i8").tobytes())
        h.update(
            np.asarray(
                np.r_[
                    c.point,
                    c.normal,
                    c.force_on_geom2,
                    c.normal_force_N,
                    c.approach_mps,
                    c.penetration_m,
                    c.velocity1_at_contact,
                    c.velocity2_at_contact,
                ],
                dtype="<f8",
            ).tobytes()
        )
    return h.digest()


def stance_gate(*, final_tick, failed, maximum_drift_m, last_second_mean_speed_mps):
    if (
        type(final_tick) is not int
        or not 0 < final_tick <= 3000
        or type(failed) is not bool
        or not np.isfinite(maximum_drift_m)
        or maximum_drift_m < 0
    ):
        raise ValueError("Malformed stance gate measurements")
    complete = final_tick == 3000 and not failed
    if complete and (
        last_second_mean_speed_mps is None
        or not np.isfinite(last_second_mean_speed_mps)
        or last_second_mean_speed_mps < 0
    ):
        raise ValueError("Complete stance requires final full second speed")
    return bool(
        complete and maximum_drift_m <= 0.30 and last_second_mean_speed_mps <= 0.15
    )


class StanceMetrics:
    """Every completed1ms: unchanged support, world-XY root excursion and speed."""

    def __init__(self, world):
        if world.core.tick != 0:
            raise ValueError("Measure stance from actual reset")
        self.rows, self.contact_hashes = [], []
        self.first_interactor_contact = None
        self.motor = MotorMetrics()
        self.pelvis = [world.core.model.body(p + "_pelvis").id for p in "AB"]
        self.initial_xy = world.scratch.xpos[self.pelvis, :2].copy()
        self.geom_owners = {
            i: world.core.model.geom(i).name[0]
            if world.core.model.geom(i).name.startswith(("A_", "B_"))
            else None
            for i in range(world.core.model.ngeom)
        }

    def observe(self, world, contacts):
        if world.core.tick != len(self.rows) + 1:
            raise ValueError("Missing/duplicate1ms stance measurement")
        self.motor.observe(world)
        model, data = world.core.model, world.scratch
        row = []
        for j, p in enumerate("AB"):
            m = world.last_metrics[p]
            pos = data.xpos[self.pelvis[j]]
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            # scratch kinematics/qvel reflect the completed live state; no extra
            # live mj_forward or evaluated state writes to refresh velocity.
            mujoco.mj_jac(model, data, jacp, jacr, pos, self.pelvis[j])
            velocity = jacp @ data.qvel
            forward = data.xmat[self.pelvis[j]].reshape(3, 3)[:2, 0]
            forward = forward / np.linalg.norm(forward)
            loads = m["floor_loads_N"]
            row.append(
                [
                    m["tilt_rad"],
                    m["drop_m"],
                    m["up_world_z"],
                    loads["right"],
                    loads["left"],
                    loads["nonfoot"],
                    *pos[:2],
                    float(np.linalg.norm(pos[:2] - self.initial_xy[j])),
                    float(np.linalg.norm(velocity[:2])),
                    *m["hand_target_distance_m"],
                    m["face_cos_positive"],
                    *forward,
                ]
            )
        if not np.isfinite(row).all():
            raise FloatingPointError("Nonfinite stance measurements")
        self.rows.append(row)
        self.contact_hashes.append(contact_digest(contacts))
        if self.first_interactor_contact is None:
            for c in contacts:
                owners = {self.geom_owners[c.geom1], self.geom_owners[c.geom2]}
                if owners == {"A", "B"} and c.normal_force_N > 0:
                    self.first_interactor_contact = {
                        "tick": world.core.tick,
                        "interval_start_tick": world.core.tick - 1,
                        "surfaces": [
                            model.geom(c.geom1).name,
                            model.geom(c.geom2).name,
                        ],
                        "normal_force_N": c.normal_force_N,
                        "point_world_m": c.point.tolist(),
                        "approach_mps": c.approach_mps,
                    }
                    break

    def summary(self, world):
        if len(self.rows) != world.core.tick or not self.rows:
            raise ValueError("Incomplete stance measurement denominator")
        a = np.array(self.rows, np.float64)
        output = {}
        for j, p in enumerate("AB"):
            drift = float(np.max(a[:, j, METRIC_COLUMNS.index("root_drift_xy_m")]))
            speed = (
                float(a[2000:3000, j, METRIC_COLUMNS.index("root_speed_xy_mps")].mean())
                if world.core.tick == 3000
                else None
            )
            output[p] = {
                "maximum_root_xy_excursion_m": drift,
                "last_full_second_mean_speed_xy_mps": speed,
                "stance_candidate": stance_gate(
                    final_tick=world.core.tick,
                    failed=world.failures[p] is not None,
                    maximum_drift_m=drift,
                    last_second_mean_speed_mps=speed,
                ),
                "first_failure_tick": None
                if world.failures[p] is None
                else world.failures[p]["tick"],
                "motor": self.motor.summary(world)[p],
            }
        return {
            "actors": output,
            "first_interactor_contact": self.first_interactor_contact,
        }


def evaluate_stance(world, *, seed, output, budget, guard):
    """One whole fixed motor trial; both zero residuals, no checkpoint selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    world.reset(seed=seed)
    snapshot = world.snapshot()
    budget.write_json(output / "initial-snapshot.json", snapshot)
    metrics = StanceMetrics(world)
    frames, commands, inputs, hashes = [], [], [], []

    def frame():
        d = world.core.data
        frames.append(
            {
                "qpos": d.qpos.copy(),
                "qvel": d.qvel.copy(),
                "activation": d.act.copy(),
                "control": d.ctrl.copy(),
                "time": np.array(float(d.time)),
            }
        )

    frame()
    while not world.done:
        inputs.append(np.stack([world.core.inputs[p].values for p in "AB"]))
        world.begin_control({p: np.zeros(138, np.float32) for p in "AB"})
        commands.append(np.stack([world.core.commands[p] for p in "AB"]))
        while world.core.remaining:
            contacts = world.advance_substep()
            metrics.observe(world, contacts)
        hashes.append(hashlib.sha256(world.core.state().tobytes()).digest())
        frame()
        guard(0)
    arrays = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    trace = {
        "teacher_inputs": np.stack(inputs),
        "commands": np.stack(commands),
        "engine_sha256": np.array([np.frombuffer(h, np.uint8) for h in hashes]),
        "metrics_1ms": np.array(metrics.rows, np.float64),
        "contact_sha256_1ms": np.array(
            [np.frombuffer(h, np.uint8) for h in metrics.contact_hashes]
        ),
    }
    budget.check(plan(arrays)["archive_bytes"] + plan(trace)["archive_bytes"])
    result = {
        "schema": "joxing-native-stance-probe-v1",
        "seed": seed,
        "placement": world.config.placement,
        "reference_profile": world.reference_profile,
        "contract": world.contract,
        "contract_hash": world.contract_hash,
        "core_contract": world.core.contract,
        "core_contract_hash": world.core.contract_hash,
        "final_tick": world.core.tick,
        "frames": len(frames),
        "candidate_points": world.referee.points,
        "motor_failures": world.failures,
        "metrics": metrics.summary(world),
        "metric_columns": list(METRIC_COLUMNS),
        "frame_tape": store(output / "frames.npz", arrays),
        "trace_tape": store(output / "stance-trace.npz", trace),
        "initial_snapshot_sha256": snapshot["sha256"],
        "sporting_qualified": False,
        "optimizer_updates": 0,
    }
    budget.write_json(output / "evaluation.json", result)
    return result


def condition_conclusion(rows):
    """Retain all four fixed starts; missing/invalid evidence never qualifies."""
    if len(rows) != 4 or {r["seed"] for r in rows} != set(SEEDS):
        raise ValueError("All four fixed stance starts required")
    counts = {
        p: sum(r["metrics"]["actors"][p]["stance_candidate"] for r in rows)
        for p in "AB"
    }
    return {
        "starts": 4,
        "qualifying_starts": counts,
        "stance_candidate_only": all(n >= 3 for n in counts.values()),
        "boxing_qualified": False,
    }


def audit_stance(world, *, folder, expected, guard):
    """A fresh reset/restore and real physical replay; never write archived poses."""
    folder = Path(folder)
    if digest(json.loads((folder / "evaluation.json").read_text())) != digest(expected):
        raise ValueError("Pinned stance result differs from file")
    if (
        world.contract_hash != expected["contract_hash"]
        or digest(expected["contract"]) != world.contract_hash
    ):
        raise ValueError("Foreign stance contract")
    frames = load(folder / "frames.npz", expected["frame_tape"])
    trace = load(folder / "stance-trace.npz", expected["trace_tape"])
    snapshot = json.loads((folder / "initial-snapshot.json").read_text())
    if snapshot["sha256"] != expected["initial_snapshot_sha256"]:
        raise ValueError("Foreign stance reset")
    world.reset(seed=expected["seed"])
    world.restore(snapshot)
    world._scratch()
    metrics = StanceMetrics(world)
    count = expected["final_tick"] // 20
    if (
        not 1 <= count <= 150
        or expected["final_tick"] != 20 * count
        or expected["frames"] != count + 1
        or trace["teacher_inputs"].shape != (count, 2, 3601)
        or trace["commands"].shape != (count, 2, 700)
        or trace["metrics_1ms"].shape != (20 * count, 2, len(METRIC_COLUMNS))
        or trace["contact_sha256_1ms"].shape != (20 * count, 32)
    ):
        raise ValueError("Stance replay complete denominator/shape")

    def check_frame(i):
        d = world.core.data
        actual = {
            "qpos": d.qpos,
            "qvel": d.qvel,
            "activation": d.act,
            "control": d.ctrl,
            "time": np.array(float(d.time)),
        }
        if set(frames) != set(actual) or any(
            a.tobytes() != frames[k][i].tobytes() for k, a in actual.items()
        ):
            raise ValueError("Stance frame replay mismatch")

    check_frame(0)
    before = world.physics_total
    for i in range(count):
        if world.done:
            raise ValueError("Stance replay ends before archived denominator")
        actual = np.stack([world.core.inputs[p].values for p in "AB"])
        if actual.tobytes() != trace["teacher_inputs"][i].tobytes():
            raise ValueError("Stance teacher input replay mismatch")
        world.begin_control({p: np.zeros(138, np.float32) for p in "AB"})
        if (
            np.stack([world.core.commands[p] for p in "AB"]).tobytes()
            != trace["commands"][i].tobytes()
        ):
            raise ValueError("Stance all1400teacher commands replay mismatch")
        for _ in range(20):
            contacts = world.advance_substep()
            metrics.observe(world, contacts)
            k = world.core.tick - 1
            if metrics.contact_hashes[-1] != trace["contact_sha256_1ms"][k].tobytes():
                raise ValueError("Stance1ms contact replay mismatch")
            if (
                np.array(metrics.rows[-1], np.float64).tobytes()
                != trace["metrics_1ms"][k].tobytes()
            ):
                raise ValueError("Stance1ms metric replay mismatch")
        if (
            hashlib.sha256(world.core.state().tobytes()).digest()
            != trace["engine_sha256"][i].tobytes()
        ):
            raise ValueError("Stance native engine replay mismatch")
        check_frame(i + 1)
        guard(0)
    if (
        not world.done
        or metrics.summary(world) != expected["metrics"]
        or world.referee.points != tuple(expected["candidate_points"])
        or world.failures != expected["motor_failures"]
    ):
        raise ValueError("Stance ending/summary replay mismatch")
    return {
        "status": "pass",
        "seed": expected["seed"],
        "new_replay_physics": world.physics_total - before,
        "all1400commands_exact": True,
        "all1ms_contacts_metrics_exact": True,
        "all_frames_engine_exact": True,
    }
