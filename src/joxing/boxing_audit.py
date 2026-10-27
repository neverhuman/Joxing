"""Read saved match tapes, without stepping physics or trusting a success flag.

Contact sign/approach, action application and history selection are independent
array arithmetic. FK/actor assembly, the diagnostic controller and HitTracker
are reused frozen implementations, explicitly not independent physics models.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from .boxing_evidence import CONTACT_COLUMNS, PLAYERS, array_sha, diagnostic_command
from .env import INTEGRATION, BoxingEnv
from .referee import GloveInterval, MatchReferee
from .rules import Touch


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit_case(folder, scene, *, expected_physics, delays):
    folder = Path(folder)
    initial = json.loads((folder / "initial.json").read_text())
    final = json.loads((folder / "final.json").read_text())
    trace = json.loads((folder / "trace.json").read_text())
    contract = json.loads((folder / "contract.json").read_text())
    env = BoxingEnv(
        scene,
        horizon_ticks=contract["horizon_ticks"],
        bundles=contract["bundles"],
        scenario=contract["reset_scenario"],
    )
    require(env.contract == contract, "Foreign case contract")
    env.restore(initial)  # Frozen semantic snapshot validation, no mj_step.
    with np.load(folder / "tape.npz", allow_pickle=False) as z:
        tape = {k: z[k] for k in z.files}
    require(trace["schema"] == "joxing-shared-evidence-v1", "Foreign evidence schema")
    require(
        trace["contact_columns"] == {k: list(v) for k, v in CONTACT_COLUMNS.items()},
        "Contact column contract",
    )
    n, start = expected_physics, initial["application"]["tick"]
    require(
        len(trace["intervals"]) == n == final["application"]["tick"] - start,
        "Incomplete physical interval denominator",
    )
    shapes = {
        "applied": (n, 2, 700),
        "excitation": (n, 2, 700),
        "packets": (n, 2, 2),
        "contact_offsets": (n + 1,),
    }
    d = len(tape["decision_ticks"])
    shapes |= {
        "observations": (d, 2, 1970),
        "histories": (d, 2, 8, 1970),
        "history_ticks": (d, 2, 8),
        "history_received": (d, 2, 8),
        "history_valid": (d, 2, 8),
        "chosen": (d, 2, 700),
        "decision_states": (d, len(initial["engine"])),
    }
    for k, shape in shapes.items():
        require(
            tape[k].shape == shape and np.isfinite(tape[k]).all(),
            "Invalid typed tape: " + k,
        )
    for k in ["applied", "chosen", "observations", "histories"]:
        require(tape[k].dtype == np.float32, "Actor/command precision: " + k)
    for k in [
        "packets",
        "contact_offsets",
        "contact_geoms",
        "history_ticks",
        "history_received",
        "decision_ticks",
    ]:
        require(tape[k].dtype == np.int64, "Clock/contact integer precision: " + k)
    require(tape["history_valid"].dtype == bool, "History validity precision")
    require(
        len(trace["decisions"]) == d == len(trace["rngs_before"]),
        "Decision denominator",
    )
    require(
        tape["contact_offsets"][0] == 0
        and np.all(np.diff(tape["contact_offsets"]) >= 0),
        "Contact offsets",
    )
    m = tape["contact_offsets"][-1]
    require(
        tape["contacts"].shape == (m, 18) and tape["contact_geoms"].shape == (m, 2),
        "Contact denominator",
    )
    require(
        tape["contacts"].dtype == np.float64 and np.isfinite(tape["contacts"]).all(),
        "Contact precision/nonfinite",
    )
    require(
        np.all(np.abs(tape["applied"]) <= 1) and np.all(np.abs(tape["chosen"]) <= 1),
        "Command range",
    )
    decoded = (1 / (1 + np.exp(-5 * (tape["applied"] - np.float32(0.5))))).astype(
        np.float64
    )
    require(
        np.array_equal(decoded, tape["excitation"]),
        "Muscle decoding differs or applied twice",
    )
    expected_ticks = np.arange(
        start if not env.remaining_control else env.control_start + 20,
        start + n,
        20,
        dtype=np.int64,
    )
    require(
        np.array_equal(tape["decision_ticks"], expected_ticks),
        "Missing/reordered decision or new mid-command decision",
    )
    histories = {
        a: {
            r["sampled_tick"]: np.array(r["values"], np.float32)
            for r in initial["application"]["histories"][a]["samples"]
        }
        for a in PLAYERS
    }
    chosen_by_tick = {}
    rngs = {a: np.random.default_rng() for a in PLAYERS}
    for a in PLAYERS:
        rngs[a].bit_generator.state = initial["application"]["rngs"][a]
    for i, tick in enumerate(tape["decision_ticks"].tolist()):
        row = trace["decisions"][i]
        require(
            row["tick"] == tick and row["game_id"] == env.game_id,
            "Foreign decision owner/clock",
        )
        raw = tape["decision_states"][i]
        require(abs(raw[0] - tick * 0.001) < 1e-9, "Decision engine clock")
        scratch = mujoco.MjData(env.model)
        mujoco.mj_setState(env.model, scratch, raw, INTEGRATION)
        require(
            not np.any(scratch.qfrc_applied) and not np.any(scratch.xfrc_applied),
            "Decision assistance",
        )
        mujoco.mj_forward(env.model, scratch)
        previous = (
            np.stack([initial["application"]["commands"][a] for a in PLAYERS]).astype(
                np.float32
            )
            if tick == start
            else tape["applied"][tick - start - 1]
        )
        observations = env._make_observations(
            scratch, tick, row["points"], dict(zip(PLAYERS, previous, strict=True))
        )
        for j, a in enumerate(PLAYERS):
            require(
                np.array_equal(observations[a].values, tape["observations"][i, j]),
                "Actor observation differs from decision engine",
            )
            histories[a][tick] = tape["observations"][i, j]
            keep = sorted(histories[a])[-8:]
            expected_history = np.zeros((8, 1970), np.float32)
            expected_times = np.full(8, -1, np.int64)
            for k, t in enumerate(keep, 8 - len(keep)):
                expected_history[k], expected_times[k] = histories[a][t], t
            require(
                np.array_equal(expected_history, tape["histories"][i, j])
                and np.array_equal(expected_times, tape["history_ticks"][i, j])
                and np.array_equal(expected_times, tape["history_received"][i, j])
                and np.array_equal(expected_times >= 0, tape["history_valid"][i, j]),
                "Causal history selection/padding/arrival mismatch",
            )
            require(
                rngs[a].bit_generator.state == trace["rngs_before"][i][a],
                "Controller RNG discontinuity",
            )
            h = SimpleNamespace(values=expected_history, valid=expected_times >= 0)
            require(
                np.array_equal(
                    diagnostic_command(observations[a], h, rngs[a]),
                    tape["chosen"][i, j],
                ),
                "Chosen command differs from frozen causal controller",
            )
        chosen_by_tick[tick] = tape["chosen"][i]
    for a in PLAYERS:
        require(
            rngs[a].bit_generator.state == final["application"]["rngs"][a],
            "Final RNG differs",
        )
    # Reconstruct delivery independently, seeded with the actual mid-command queue.
    for j, a in enumerate(PLAYERS):
        q = initial["application"]["queues"][a]
        current = np.array(initial["application"]["commands"][a], np.float32)
        last = q["last_applied"]
        receipt = (
            [-1, -1] if last is None else [last["decision_tick"], last["apply_tick"]]
        )
        pending = [
            (p["decision_tick"], p["apply_tick"], np.array(p["command"], np.float32))
            for p in q["pending"]
        ]
        for k in range(n):
            tick = start + k
            if tick in chosen_by_tick:
                pending.append((tick, tick + delays[a], chosen_by_tick[tick][j]))
            due = [p for p in pending if p[1] <= tick]
            pending = [p for p in pending if p[1] > tick]
            if due:
                winner = max(due, key=lambda p: p[0])
                if winner[0] > receipt[0]:
                    receipt, current = list(winner[:2]), winner[2]
            require(
                np.array_equal(current, tape["applied"][k, j])
                and receipt == tape["packets"][k, j].tolist(),
                "Command delivery/hold mismatch",
            )
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
    env.restore(final)
    transition = {k: v for k, v in env.transition().items() if k != "observations"}
    require(
        json.loads(json.dumps(transition)) == trace["transition"],
        "Termination/truncation/ledger transition differs",
    )
    require(
        len(trace["state_sha256"]) == n
        and all(len(x) == 64 for x in trace["state_sha256"]),
        "Missing engine replay hashes",
    )
    if not final["application"]["truncated"]:
        require(
            array_sha(np.array(final["engine"], np.float64))
            == trace["state_sha256"][-1],
            "Final engine differs from saved interval hash",
        )
    return {
        "status": "passed",
        "physics": n,
        "positive_glove_touches": touch_count,
        "peak_glove_force_N": peak,
        "summed_glove_normal_impulse_Ns": impulse,
        "audit_new_physics": 0,
        "limitations": [
            "Reuses frozen FK/actor/controller/HitTracker implementations",
            "Geometric clearance and saved trunk/face frames are source measurements, not independent physics verification",
            "No stable motor, realistic punch, legal native bout or learning claim",
        ],
    }


def compare_replay(source, replay):
    source, replay = Path(source), Path(replay)
    st = json.loads((source / "trace.json").read_text())
    rt = json.loads((replay / "trace.json").read_text())
    start = json.loads((replay / "initial.json").read_text())["application"]["tick"]
    base = json.loads((source / "initial.json").read_text())["application"]["tick"]
    offset = start - base
    require(
        json.loads((source / "final.json").read_text())
        == json.loads((replay / "final.json").read_text()),
        "Replay final application/engine mismatch",
    )
    require(
        st["intervals"][offset:] == rt["intervals"]
        and st["state_sha256"][offset:] == rt["state_sha256"],
        "Replay contact/event/engine tail mismatch",
    )
    with (
        np.load(source / "tape.npz", allow_pickle=False) as a,
        np.load(replay / "tape.npz", allow_pickle=False) as b,
    ):
        for key in ["applied", "excitation", "packets"]:
            require(
                np.array_equal(a[key][offset:], b[key]), "Replay command tail: " + key
            )
        first = int(a["contact_offsets"][offset])
        for key in ["contacts", "contact_geoms"]:
            require(
                np.array_equal(a[key][first:], b[key]), "Replay contact tail: " + key
            )
        require(
            np.array_equal(a["contact_offsets"][offset:] - first, b["contact_offsets"]),
            "Replay contact offsets",
        )
        selected = a["decision_ticks"] >= start
        for key in [
            "decision_ticks",
            "chosen",
            "decision_states",
            "observations",
            "histories",
            "history_ticks",
            "history_received",
            "history_valid",
        ]:
            require(
                np.array_equal(a[key][selected], b[key]),
                "Replay decision/history tail: " + key,
            )
    return {
        "status": "passed",
        "source_tick": start,
        "tail_physics": len(rt["intervals"]),
        "same_source_exact": True,
    }
