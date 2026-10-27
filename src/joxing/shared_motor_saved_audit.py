"""Saved motor measurements audited without stepping or calling the live writer.

Checks semantic feature assembly, reference timeline, muscle transformation,
signed contact algebra, independent scratch posture and support persistence.
Policy inference and exact trajectory continuation remain separate gates.
"""

import copy

import mujoco
import numpy as np

from .boxing_evidence import CONTACT_COLUMNS
from .env import INTEGRATION
from .evidence_storage import read_arrays, read_json
from .provenance import digest
from .shared_motor import MotorReferenceState
from .shared_motor_audit import CACHE_FIELDS, PLAYERS, SemanticInputAudit, byte_sha
from .shared_motor_inputs import MotorPolicyInput, reference_sha
from .support import SupportMonitor


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_measurements(task, trace, arrays):
    """Also accepts decoded data, so corruption tests can bypass container hashes."""
    m = task.model
    require(trace["schema"] == "joxing-shared-motor-evidence-v1", "Motor tape schema")
    require(trace["contract"] == task.contract, "Motor tape contract")
    require(
        digest(trace["contact_columns"]) == digest(CONTACT_COLUMNS),
        "Motor contact columns",
    )
    require(
        trace["competition"] is False and trace["optimizer_updates"] == 0, "Motor scope"
    )
    initial = trace["initial"]
    initial_payload = {k: v for k, v in initial.items() if k != "sha256"}
    require(initial["sha256"] == digest(initial_payload), "Motor initial checksum")
    require(initial["contract_hash"] == task.contract_hash, "Motor initial contract")
    start = initial["tick"]
    ticks = arrays["ticks"]
    count, decisions, nc = (
        len(ticks),
        len(arrays["decision_ticks"]),
        len(arrays["contacts"]),
    )
    width = mujoco.mj_stateSize(m, INTEGRATION)
    shapes = {
        "decision_ticks": (decisions,),
        "decision_engines": (decisions, width),
        "raw": (decisions, 2, 3601),
        "normalized": (decisions, 2, 3601),
        "commands": (decisions, 2, 700),
        "excitation": (decisions, 2, 700),
        "ticks": (count,),
        "end_qpos": (count, m.nq),
        "posture": (count, 2, 2),
        "loads": (count, 2, 3),
        "contact_offsets": (count + 1,),
        "contact_geoms": (nc, 2),
        "contacts": (nc, 18),
        "video_ticks": (len(arrays["video_ticks"]),),
        "video_states": (len(arrays["video_ticks"]), width),
    }
    for k, shape in {
        "qpos": (170,),
        "qvel": (170,),
        "act": (1400,),
        "qacc": (170,),
        "actuator_force": (1400,),
        "actuator_length": (1400,),
        "actuator_velocity": (1400,),
        "key_xpos": (2, 7, 3),
    }.items():
        shapes["witness_" + k] = (decisions, *shape)
    integer = {
        "ticks",
        "decision_ticks",
        "contact_offsets",
        "contact_geoms",
        "video_ticks",
    }
    single = {"raw", "normalized", "commands"}
    require(set(arrays) == set(shapes), "Motor arrays denominator")
    for k, shape in shapes.items():
        dtype = np.dtype(
            np.int64 if k in integer else np.float32 if k in single else np.float64
        )
        require(
            arrays[k].shape == shape
            and arrays[k].dtype == dtype
            and np.isfinite(arrays[k]).all(),
            "Motor array precision/shape: " + k,
        )
    require(
        np.array_equal(ticks, np.arange(start + 1, start + count + 1)),
        "Motor interval clock",
    )
    final_tick = start + count
    require(final_tick <= task.horizon_ticks, "Motor horizon overflow")
    expected_decisions = np.arange(
        start if not initial["remaining"] else start + initial["remaining"],
        final_tick,
        20,
    )
    require(
        np.array_equal(arrays["decision_ticks"], expected_decisions),
        "Motor decision clock",
    )
    require(
        len(trace["decisions"]) == decisions and len(trace["state_sha256"]) == count,
        "Motor metadata denominator",
    )
    require(
        all(
            isinstance(h, str)
            and len(h) == 64
            and all(c in "0123456789abcdef" for c in h)
            for h in trace["state_sha256"]
        ),
        "Motor engine digest format",
    )
    # Independent action arithmetic, not the production excitation() helper.
    command = arrays["commands"]
    require(np.all(abs(command) <= 1), "Motor normalized command range")
    transformed = (1 / (1 + np.exp(-5 * (command - np.float32(0.5))))).astype(
        np.float64
    )
    require(
        np.array_equal(transformed, arrays["excitation"]), "Motor sigmoid applied once"
    )
    norm = trace["normalizer"]
    require(
        digest(norm) == task.teacher.normalizer.fingerprint,
        "Motor frozen normalizer identity",
    )
    require(
        norm["training"] is False
        and norm["norm_reward"] is False
        and norm["norm_obs"] is True,
        "Motor frozen normalizer mode",
    )
    normalized = np.clip(
        (arrays["raw"] - np.asarray(norm["mean"]))
        / np.sqrt(np.asarray(norm["var"]) + norm["epsilon"]),
        -norm["clip"],
        norm["clip"],
    ).astype(np.float32)
    require(
        np.array_equal(normalized, arrays["normalized"]), "Motor saved normalized input"
    )
    semantic = SemanticInputAudit(m, task.drivers["A"].task.model)
    for i, tick in enumerate(arrays["decision_ticks"]):
        row = trace["decisions"][i]
        require(row["tick"] == int(tick), "Motor decision metadata clock")
        refs, samples = {}, {}
        for pi, p in enumerate(PLAYERS):
            saved = row["references"][p]
            driver = copy.copy(task.drivers[p])
            driver.tick = 0
            driver.sample(int(tick), saved["sample_time"])
            require(
                digest(saved) == digest(driver.snapshot()),
                "Motor native reference timeline",
            )
            refs[p] = MotorReferenceState(
                saved["qpos"], saved["future_qpos"], saved["keypoints_relative_pelvis"]
            )
            samples[p] = MotorPolicyInput(
                p, task.game_id, int(tick), arrays["raw"][i, pi], reference_sha(refs[p])
            )
        witness = {
            k: arrays["witness_" + k][i]
            for k in ("qpos", "qvel", "act", *CACHE_FIELDS, "key_xpos")
        }
        witness["tick"] = int(tick)
        semantic.verify(
            witness,
            arrays["decision_engines"][i],
            samples,
            refs,
            game_id=task.game_id,
            tick=int(tick),
        )
    offsets = arrays["contact_offsets"]
    require(
        offsets[0] == 0 and offsets[-1] == nc and np.all(np.diff(offsets) >= 0),
        "Motor contact offsets",
    )
    geoms, contacts = arrays["contact_geoms"], arrays["contacts"]
    require(np.all((geoms >= 0) & (geoms < m.ngeom)), "Motor contact identity")
    if nc:
        require(np.all(geoms[:, 0] != geoms[:, 1]), "Motor self geom contact")
        require(
            np.allclose(
                np.linalg.norm(contacts[:, 3:6], axis=1), 1, rtol=0, atol=1e-10
            ),
            "Motor contact normal",
        )
        approach = np.sum(
            (contacts[:, 9:12] - contacts[:, 12:15]) * contacts[:, 3:6], axis=1
        )
        require(
            np.allclose(approach, contacts[:, 16], rtol=0, atol=1e-10),
            "Motor contact approach",
        )
        force = np.sum(contacts[:, 6:9] * contacts[:, 3:6], axis=1)
        require(
            np.allclose(force, contacts[:, 15], rtol=1e-10, atol=1e-8),
            "Motor signed normal force",
        )
        require(
            np.all(contacts[:, 15] >= 0) and np.all(contacts[:, 17] >= 0),
            "Motor force/penetration sign",
        )
    # Resolve all floor roles and body weights from compiled model, not saved labels.
    floor = m.geom("floor").id
    roles, weights = {}, {}
    scratch = mujoco.MjData(m)
    mujoco.mj_setState(
        m, scratch, np.asarray(initial["engine"], np.float64), INTEGRATION
    )
    mujoco.mj_kinematics(m, scratch)
    for p in PLAYERS:
        bodies = {i for i in range(m.nbody) if m.body(i).name.startswith(p + "_")}
        weights[p] = float(sum(m.body_mass[list(bodies)]) * 9.81)
        roles[p] = {}
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) in bodies:
                name = m.body(int(m.geom_bodyid[g])).name
                roles[p][g] = (
                    "right"
                    if name in (p + "_calcn_r", p + "_toes_r")
                    else "left"
                    if name in (p + "_calcn_l", p + "_toes_l")
                    else "nonfoot"
                )
    require(
        trace["floor_roles"]
        == {p: {str(g): r for g, r in roles[p].items()} for p in PLAYERS},
        "Motor saved floor role",
    )
    require(trace["weight_N"] == weights, "Motor saved body weight")
    prefix = trace["support_prefix"]
    require(
        prefix["tick"] == start and prefix["initial_height"] == trace["initial_height"],
        "Motor support prefix clock/height",
    )
    if start == 0:
        for p in PLAYERS:
            h = float(scratch.xpos[m.body(p + "_pelvis").id, 2])
            require(
                h == trace["initial_height"][p]
                and prefix["monitors"][p] == SupportMonitor().snapshot()
                and prefix["physical_failure"][p] is None
                and prefix["tracking_failure"][p] is None,
                "Motor initial support state",
            )
    monitors = {p: SupportMonitor(**prefix["monitors"][p]) for p in PLAYERS}
    failures = copy.deepcopy(prefix["physical_failure"])
    tracking = copy.deepcopy(prefix["tracking_failure"])
    for i, tick in enumerate(ticks):
        scratch.qpos[:] = arrays["end_qpos"][i]
        mujoco.mj_kinematics(m, scratch)
        for pi, p in enumerate(PLAYERS):
            load = np.zeros(3)
            for k in range(int(offsets[i]), int(offsets[i + 1])):
                g1, g2 = geoms[k]
                if g1 == floor and g2 in roles[p]:
                    other, f = g2, contacts[k, 6:9]
                elif g2 == floor and g1 in roles[p]:
                    other, f = g1, -contacts[k, 6:9]
                else:
                    continue
                load[("right", "left", "nonfoot").index(roles[p][other])] += max(
                    0.0, float(f[2])
                )
            require(
                np.array_equal(load, arrays["loads"][i, pi]),
                "Motor support force attribution",
            )
            pelvis = m.body(p + "_pelvis").id
            tilt = float(
                np.arccos(np.clip(scratch.xmat[pelvis].reshape(3, 3)[2, 1], -1, 1))
            )
            drop = trace["initial_height"][p] - float(scratch.xpos[pelvis, 2])
            require(
                np.array_equal([tilt, drop], arrays["posture"][i, pi]),
                "Motor scratch posture",
            )
            reason = monitors[p].update(
                dict(zip(("right", "left", "nonfoot"), load, strict=True)),
                weights[p],
                tilt,
                drop,
                0.001,
            )
            if reason and failures[p] is None:
                failures[p] = {"tick": int(tick), "reason": reason}
            if tick % 20 == 0:
                prev = np.searchsorted(arrays["decision_ticks"], tick, side="left") - 1
                oldref = (
                    trace["decisions"][prev]["references"][p]
                    if prev >= 0
                    else initial["references"][p]
                )
                qp, _ = semantic.joints[p]
                error = float(
                    np.mean(
                        abs(
                            arrays["end_qpos"][i, qp][3:]
                            - np.asarray(oldref["qpos"])[3:]
                        )
                    )
                )
                if (
                    error > task.drivers[p].task.env.qpos_diff_th
                    and tracking[p] is None
                ):
                    tracking[p] = {"tick": int(tick), "error": error}
        if trace["mid"] is not None and tick == trace["mid"]["task"]["tick"]:
            saved = trace["mid"]
            require(
                saved["support"]
                == {
                    "tick": int(tick),
                    "initial_height": trace["initial_height"],
                    "monitors": {p: v.snapshot() for p, v in monitors.items()},
                    "physical_failure": failures,
                    "tracking_failure": tracking,
                },
                "Motor actual mid support",
            )
    require(failures == trace["physical_failure"], "Motor physical failure first latch")
    require(tracking == trace["tracking_failure"], "Motor tracking first latch")
    for label, snapshot in [
        ("final", trace["last_valid"]),
        ("mid", None if trace["mid"] is None else trace["mid"]["task"]),
    ]:
        if snapshot is None:
            continue
        require(
            snapshot["sha256"]
            == digest({k: v for k, v in snapshot.items() if k != "sha256"})
            and snapshot["contract_hash"] == task.contract_hash,
            "Motor saved snapshot checksum/contract",
        )
        tick = snapshot["tick"]
        require(
            start <= tick <= final_tick and (label != "final" or tick == final_tick),
            "Motor saved snapshot clock",
        )
        expected_sha = (
            byte_sha(np.asarray(initial["engine"], np.float64))
            if tick == start
            else trace["state_sha256"][tick - start - 1]
        )
        require(
            byte_sha(np.asarray(snapshot["engine"], np.float64)) == expected_sha,
            "Motor saved snapshot engine digest",
        )
    video_ticks = arrays["video_ticks"]
    if len(video_ticks):
        expected = sorted(
            {start, final_tick, *range(((start // 40) + 1) * 40, final_tick + 1, 40)}
        )
        require(video_ticks.tolist() == expected, "Motor fixed recording schedule")
        for tick, state in zip(video_ticks, arrays["video_states"], strict=True):
            require(abs(state[0] - tick * 0.001) <= 1e-9, "Motor video state time")
            if tick == start:
                require(
                    state.tolist() == initial["engine"], "Motor video initial engine"
                )
            else:
                require(
                    byte_sha(state) == trace["state_sha256"][int(tick) - start - 1],
                    "Motor video engine digest",
                )
    return {
        "passed": True,
        "physics_steps": 0,
        "measured_intervals": count,
        "decisions": decisions,
        "contacts": nc,
        "physical_failure": failures,
        "tracking_failure": tracking,
        "video_frames": len(video_ticks),
        "policy_inference_audited": False,
        "trajectory_continuation_audited": False,
    }


def audit_saved_tape(task, folder):
    return audit_measurements(
        task, read_json(folder / "trace.json"), read_arrays(folder / "tape.npz")
    )
