"""Saved-data score, signed-contact and frozen-controller checks; no simulation."""

from collections import Counter
from types import SimpleNamespace

import mujoco
import numpy as np

from .boxing_evidence import array_sha
from .env import INTEGRATION
from .evidence_storage import read_arrays, read_json
from .provenance import digest
from .score_diagnostic import audit_saved_trace


def audit_source(folder, env, controllers):
    initial, final, trace, contract = [
        read_json(folder / (n + ".json"))
        for n in ("initial", "final", "trace", "contract")
    ]
    tape = read_arrays(folder / "tape.npz")
    for state in (initial, final):
        if digest({k: v for k, v in state.items() if k != "sha256"}) != state[
            "sha256"
        ] or state["contract_hash"] != digest(contract):
            raise ValueError("Native strike snapshot/contract identity")
    n = len(trace["intervals"])
    if trace["quarantined"] or n != contract["horizon_ticks"] or n > 3000:
        raise ValueError("Native source interval denominator")
    if contract != env.contract or env.model.nu != 1400 or env.model.na != 1400:
        raise ValueError("Native source scene/controller/anatomy contract")
    if (
        tape["contact_offsets"].shape != (n + 1,)
        or tape["contact_offsets"][0] != 0
        or np.any(np.diff(tape["contact_offsets"]) < 0)
        or tape["contact_offsets"][-1] != len(tape["contacts"])
        or len(tape["contacts"]) > 150000
    ):
        raise ValueError("Native signed-contact extent")
    semantic = {g["id"]: g for g in env.scene.report["physical_geoms"]}
    gloves = {g["name"]: g["id"] for g in semantic.values() if g["role"] == "glove"}
    if len(gloves) != 4:
        raise ValueError("Native glove denominator")
    for i, row in enumerate(trace["intervals"]):
        if row["tick"] != i:
            raise ValueError("Native physical clock gap")
        lo, hi = tape["contact_offsets"][i : i + 2]
        values, ids = tape["contacts"][lo:hi], tape["contact_geoms"][lo:hi]
        if (
            array_sha(values) != row["contacts_sha256"]
            or array_sha(ids) != row["contact_geoms_sha256"]
        ):
            raise ValueError("Native saved contact bytes differ")
        if (
            not np.isfinite(values).all()
            or np.any(values[:, 15] < 0)
            or np.any(values[:, 17] < 0)
            or not np.allclose(
                np.linalg.norm(values[:, 3:6], axis=1), 1, rtol=0, atol=1e-10
            )
            or not np.allclose(
                np.sum(values[:, 3:6] * values[:, 6:9], axis=1),
                values[:, 15],
                rtol=1e-10,
                atol=1e-8,
            )
            or not np.allclose(
                np.sum((values[:, 9:12] - values[:, 12:15]) * values[:, 3:6], axis=1),
                values[:, 16],
                rtol=1e-12,
                atol=1e-10,
            )
        ):
            raise ValueError("Native signed force/normal/approach invariant")
        if any(int(g) not in semantic for g in ids.ravel()):
            raise ValueError("Unknown native contact surface")
        for name, glove in gloves.items():
            g = row["gloves"][name]
            e = g["physical_evidence"]
            matches = [
                (k, pair, int(pair[1] if pair[0] == glove else pair[0]))
                for k, pair in enumerate(ids)
                if glove in pair
                and semantic[int(pair[1] if pair[0] == glove else pair[0])]["actor"]
                not in (None, name[0])
            ]
            if len(matches) != len(g["touches"]) or len(matches) != len(
                e["trunk_point_velocities_world"]
            ):
                raise ValueError("Native staged glove/contact denominator")
            axis, position = (
                np.array(e["axis_world"]),
                np.array(e["glove_position_world"]),
            )
            if abs(np.linalg.norm(axis) - 1) > 1e-10:
                raise ValueError("Native staged glove axis")
            extension = (
                np.array(e["trunk_rotation_world"]).T
                @ (position - np.array(e["trunk_position_world"]))
            )[0]
            if abs(extension - g["extension_m"]) > 1e-12:
                raise ValueError("Native staged recovery extension")
            for index, (k, pair, target) in enumerate(matches):
                c, t = values[k], g["touches"][index]
                normal = c[3:6] if pair[0] == glove else -c[3:6]
                velocity = c[9:12] if pair[0] == glove else c[12:15]
                role = semantic[target]["role"]
                zone = (
                    "guard"
                    if role in ("guard", "glove")
                    else role
                    if role in ("head", "torso")
                    else "native_non_scoring"
                )
                active = float(
                    np.dot(
                        velocity - np.array(e["trunk_point_velocities_world"][index]),
                        normal,
                    )
                )
                if (
                    t["sample_id"] != f"{k}:{pair[0]}:{pair[1]}"
                    or t["zone"] != zone
                    or t["normal_force_N"] != c[15]
                    or t["approach_mps"] != c[16]
                    or abs(t["active_speed_mps"] - active) > 1e-10
                    or abs(t["face_cos"] - np.dot(axis, normal)) > 1e-12
                    or t["striking_face"] != bool(np.dot(c[:3] - position, axis) > 0)
                ):
                    raise ValueError("Native contact-to-score measurement binding")
    scores = audit_saved_trace(
        initial["application"]["referee"],
        final["application"]["referee"],
        trace["intervals"],
        n,
        maximum_intervals=3000,
    )
    env.restore(initial)
    for actor in "AB":
        controllers[actor].restore(initial["application"]["policy_states"][actor])
    histories = {a: {} for a in "AB"}
    scratch = mujoco.MjData(env.model)
    events = [e for row in trace["intervals"] for e in row["events"]]
    for j, d in enumerate(trace["decisions"]):
        tick = d["tick"]
        if (
            tick != 20 * j
            or d["rng_before"] != {a: env.rngs[a].bit_generator.state for a in "AB"}
            or d["policy_before"] != {a: controllers[a].snapshot() for a in "AB"}
        ):
            raise ValueError("Native decision/controller/RNG clock")
        mujoco.mj_setState(env.model, scratch, tape["decision_engines"][j], INTEGRATION)
        if np.any(scratch.xfrc_applied) or np.any(scratch.qfrc_applied):
            raise ValueError("Native decision external assistance")
        mujoco.mj_forward(env.model, scratch)
        points = tuple(
            sum(
                e["points"]
                for e in events
                if e["event_id"].split("/")[1].startswith(a + "_")
                and e["finalized_tick"] < tick
            )
            for a in "AB"
        )
        commands = {
            a: np.zeros(700, np.float32) if j == 0 else tape["commands"][j - 1, k]
            for k, a in enumerate("AB")
        }
        observed = env._make_observations(scratch, tick, points, commands)
        for k, a in enumerate("AB"):
            if observed[a].values.tobytes() != tape["observations"][j, k].tobytes():
                raise ValueError(
                    "Native observation differs from saved decision engine"
                )
            histories[a][tick] = observed[a].values
            history = np.zeros((8, 1970), np.float32)
            times = np.full(8, -1, np.int64)
            keep = sorted(histories[a])[-8:]
            for index, t in enumerate(keep, 8 - len(keep)):
                history[index], times[index] = histories[a][t], t
            if (
                array_sha(history) != d["history_sha256"][a]
                or times.tolist() != d["history_ticks"][a]
            ):
                raise ValueError("Native causal history selection")
            command = controllers[a](
                SimpleNamespace(values=observed[a].values, sampled_tick=tick),
                None,
                env.rngs[a],
            )
            if command.tobytes() != tape["commands"][j, k].tobytes():
                raise ValueError("Native frozen controller command differs")
        if {a: controllers[a].snapshot() for a in "AB"} != d["origin_after"]:
            raise ValueError("Native controller origin transition")
    if {a: env.rngs[a].bit_generator.state for a in "AB"} != final["application"][
        "rngs"
    ] or {a: controllers[a].snapshot() for a in "AB"} != final["application"][
        "policy_states"
    ]:
        raise ValueError("Native final RNG/controller ownership")
    onsets = []
    causes = Counter()
    for row in scores["explanations"]:
        for sample in row["samples"]:
            if sample["first_zone_onset"]:
                onsets.append({"glove": row["glove"], "tick": row["tick"], **sample})
                causes.update(
                    k
                    for k in ("approach", "active_speed", "face_cos", "striking_face")
                    if not sample["instantaneous_predicates"][k]
                )
    return {
        "status": "passed",
        "new_physics": 0,
        "intervals": n,
        "contacts": len(tape["contacts"]),
        "decisions": len(trace["decisions"]),
        "points": [
            sum(
                e["points"]
                for e in events
                if e["event_id"].split("/")[1].startswith(a + "_")
            )
            for a in "AB"
        ],
        "first_direct_zone_onsets": onsets,
        "ineligible_onset_predicates": dict(causes),
        "effective_sample_rejections": scores["rejections"],
        "source_final_sha256": final["sha256"],
        "whole_boxing_proof_passed": False,
        "limitations": [
            "Diagnostic range reset; no trained opponent awareness or tactical history",
            "Saved staged trunk/face geometry is reused, not independent force calibration",
        ],
    }
