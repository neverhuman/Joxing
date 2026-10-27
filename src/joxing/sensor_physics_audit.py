"""Saved physical/contact audit for sensed tapes, without controller assumptions.

Signed force/approach, ownership, glove arithmetic and ledger checks are the
frozen D0 array-audit method, copied explicitly to avoid changing that version.
MatchReferee/HitTracker and saved geometric measurements remain reused.
No physics steps are performed. This is not independent physical calibration.
"""

import json
from pathlib import Path

import numpy as np

from .boxing_audit import require
from .boxing_evidence import CONTACT_COLUMNS, array_sha
from .referee import GloveInterval, MatchReferee
from .rules import Touch
from .sensed_env import SensedBoxingEnv
from .sensor_audit import audit_sensor_transport, read_arrays, typed


def audit_sensor_physics(folder, scene, *, expected_physics, delays):
    folder = Path(folder)
    result = audit_sensor_transport(
        folder, scene, expected_physics=expected_physics, delays=delays
    )
    initial = json.loads((folder / "sensor-initial.json").read_text())["base"]
    outer_final = json.loads((folder / "sensor-final.json").read_text())
    final = outer_final["base"]
    trace = json.loads((folder / "trace.json").read_text())
    contract = json.loads((folder / "contract.json").read_text())
    sensor = json.loads((folder / "sensor-trace.json").read_text())
    samples = read_arrays(folder / "sensor-tape.npz")
    tape = read_arrays(folder / "tape.npz")
    n, start = expected_physics, initial["application"]["tick"]
    require(trace["schema"] == "joxing-shared-evidence-v1", "Physical tape schema")
    require(
        trace["contact_columns"] == {k: list(v) for k, v in CONTACT_COLUMNS.items()},
        "Contact column contract",
    )
    typed(tape["contact_offsets"], (n + 1,), np.int64, "contact_offsets")
    require(
        tape["contact_offsets"][0] == 0
        and np.all(np.diff(tape["contact_offsets"]) >= 0),
        "Contact offsets",
    )
    m = int(tape["contact_offsets"][-1])
    require(0 <= m <= 150000, "Contact allocation bound")
    typed(tape["contacts"], (m, 18), np.float64, "contacts")
    typed(tape["contact_geoms"], (m, 2), np.int64, "contact_geoms")
    referee = MatchReferee.restore(scene, initial["application"]["referee"])
    peak, impulse, touch_count = 0.0, 0.0, 0
    points_by_tick = {}
    for i, row in enumerate(trace["intervals"]):
        tick = start + i
        require(row["tick"] == tick, "Missing/duplicate interval clock")
        points_by_tick[tick] = list(referee.points)
        lo, hi = tape["contact_offsets"][i : i + 2]
        contacts, ids = tape["contacts"][lo:hi], tape["contact_geoms"][lo:hi]
        require(
            np.all(contacts[:, 15] >= 0) and np.all(contacts[:, 17] >= 0),
            "Negative force/penetration magnitude",
        )
        require(
            np.allclose(
                np.linalg.norm(contacts[:, 3:6], axis=1), 1, rtol=0, atol=1e-10
            ),
            "Contact normal not unit",
        )
        require(
            np.allclose(
                np.sum(contacts[:, 3:6] * contacts[:, 6:9], axis=1),
                contacts[:, 15],
                rtol=1e-10,
                atol=1e-8,
            ),
            "Signed normal force mismatch",
        )
        require(
            np.allclose(
                np.sum(
                    (contacts[:, 9:12] - contacts[:, 12:15]) * contacts[:, 3:6], axis=1
                ),
                contacts[:, 16],
                rtol=1e-12,
                atol=1e-10,
            ),
            "Contact approach/velocity sign mismatch",
        )
        for pair in ids:
            require(
                all(int(g) in referee.registry for g in pair), "Unknown contact surface"
            )
        intervals = {}
        require(set(row["gloves"]) == set(referee.gloves), "Missing glove evidence")
        for name, glove in referee.gloves.items():
            g = row["gloves"][name]
            evidence = g["physical_evidence"]
            matches = []
            for k, pair in enumerate(ids):
                if glove not in pair:
                    continue
                other = int(pair[1] if pair[0] == glove else pair[0])
                surface = referee.registry[other]
                if surface["actor"] in [None, name[0]]:
                    continue
                matches.append((k, pair, other))
            require(
                len(matches)
                == len(g["touches"])
                == len(evidence["trunk_point_velocities_world"]),
                "Glove/contact denominator",
            )
            axis = np.array(evidence["axis_world"])
            position = np.array(evidence["glove_position_world"])
            require(
                np.isfinite(axis).all() and abs(np.linalg.norm(axis) - 1) < 1e-10,
                "Glove face axis",
            )
            extension = (
                np.array(evidence["trunk_rotation_world"]).T
                @ (position - np.array(evidence["trunk_position_world"]))
            )[0]
            require(
                abs(extension - g["extension_m"]) < 1e-12, "Recovery extension mismatch"
            )
            for h, (k, pair, other) in enumerate(matches):
                c = contacts[k]
                t = g["touches"][h]
                normal = c[3:6] if pair[0] == glove else -c[3:6]
                velocity = c[9:12] if pair[0] == glove else c[12:15]
                role = referee.registry[other]["role"]
                zone = (
                    "guard"
                    if role in ["guard", "glove"]
                    else role
                    if role in ["head", "torso"]
                    else "native_non_scoring"
                )
                require(
                    t["sample_id"] == f"{k}:{pair[0]}:{pair[1]}" and t["zone"] == zone,
                    "Glove surface/onset identity",
                )
                require(
                    t["normal_force_N"] == c[15] and t["approach_mps"] == c[16],
                    "Glove force/approach mismatch",
                )
                require(
                    abs(
                        t["active_speed_mps"]
                        - np.dot(
                            velocity
                            - np.array(evidence["trunk_point_velocities_world"][h]),
                            normal,
                        )
                    )
                    < 1e-10,
                    "Active glove speed mismatch",
                )
                require(
                    abs(t["face_cos"] - np.dot(axis, normal)) < 1e-12
                    and t["striking_face"] == bool(np.dot(c[:3] - position, axis) > 0),
                    "Glove striking face mismatch",
                )
                if c[15] > 0:
                    touch_count += 1
                    peak = max(peak, float(c[15]))
                    impulse += float(c[15]) * 0.001
            intervals[name] = GloveInterval(
                tuple(Touch(**t) for t in g["touches"]),
                g["clearance_m"],
                g["extension_m"],
                evidence,
            )
        events = referee.commit(
            intervals, tick, bell=tick + 1 == contract["horizon_ticks"]
        )
        require(events == row["events"], "Candidate event/impulse ledger differs")
    for row in trace["decisions"]:
        require(
            row["points"] == points_by_tick[row["tick"]],
            "Decision points differ from saved contacts",
        )
    require(
        referee.snapshot() == final["application"]["referee"], "Final referee differs"
    )
    points_by_tick[start + n] = list(referee.points)
    for t, points in zip(
        samples["sample_ticks"].tolist(), sensor["sample_points"], strict=True
    ):
        require(
            points == points_by_tick[t], "Sample points differ from physical ledger"
        )
    env = SensedBoxingEnv(
        scene,
        horizon_ticks=contract["horizon_ticks"],
        bundles=contract["bundles"],
        scenario=contract["reset_scenario"],
        sensor_schedule=contract["sensor_schedule"],
    )
    env.restore(outer_final)
    transition = {k: v for k, v in env.transition().items() if k != "observations"}
    require(
        json.loads(json.dumps(transition)) == trace["transition"],
        "Physical ending/reward/ledger transition differs",
    )
    require(
        all(isinstance(h, str) and len(h) == 64 for h in trace["state_sha256"]),
        "Malformed interval engine hashes",
    )
    require(
        array_sha(np.array(final["engine"], np.float64)) == trace["state_sha256"][-1],
        "Final engine hash",
    )
    require(env.physics_total == 0, "Physical auditor stepped physics")
    return result | {
        "status": "passed",
        "physical_referee_audited": True,
        "formal_sensor_proof": False,
        "positive_glove_touches": touch_count,
        "peak_glove_force_N": peak,
        "summed_glove_normal_impulse_Ns": impulse,
        "limitations": [
            "Reuses frozen FK/controller/HitTracker and saved clearance/trunk frames; no physical calibration",
            "A per-case audit is not the complete eight-row formal proof or motor qualification",
        ],
    }
