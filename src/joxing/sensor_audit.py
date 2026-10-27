"""Received-input audit using saved arrays, with zero new physics steps.

Transport/history/receipt selection is independent of SensorQueue/TimedHistory.
FK, the frozen diagnostic controller, and semantic snapshot validation are
explicitly reused. This component does NOT replace the physical referee audit,
the D0 source-host oracle comparison, or the formal eight-case proof runner.
"""

import copy
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from .boxing_audit import compare_replay, require
from .boxing_evidence import PLAYERS, array_sha, diagnostic_command
from .env import INTEGRATION, excitation
from .observations import SCHEMA_HASH
from .provenance import digest
from .sensed_env import SensedBoxingEnv


def read_arrays(path):
    with zipfile.ZipFile(path) as archive:
        require(
            sum(i.file_size for i in archive.infolist()) <= 128 * 1024**2,
            "Uncompressed sensor evidence cap",
        )
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def typed(value, shape, dtype, name):
    require(
        value.shape == shape
        and value.dtype == np.dtype(dtype)
        and np.isfinite(value).all(),
        "Malformed typed sensor evidence: " + name,
    )


def audit_transport(
    initial, final, trace, sensor_trace, tape, samples, schedule, delays
):
    """Audit transport only, given physically validated sampled vectors.

    No SensorQueue or TimedHistory calls are made here. Initial queues are a
    separately validated checkpoint dependency; compare replay starts against
    their source case in the formal runner, not a freshly invented checkpoint.
    """
    base, finish = initial["base"], final["base"]
    app = base["application"]
    start, end = app["tick"], finish["application"]["tick"]
    n = end - start
    require(0 < n <= 1000, "Sensor interval denominator")
    remaining = app["remaining_control"]
    first = start if not remaining else start + remaining
    decisions = np.arange(first, end, 20, dtype=np.int64)
    typed(tape["decision_ticks"], decisions.shape, np.int64, "decision_ticks")
    require(
        np.array_equal(tape["decision_ticks"], decisions), "Decision clock/hold tail"
    )
    d = len(decisions)
    for name, shape, dtype in [
        ("observations", (d, 2, 1970), np.float32),
        ("histories", (d, 2, 8, 1970), np.float32),
        ("history_ticks", (d, 2, 8), np.int64),
        ("history_received", (d, 2, 8), np.int64),
        ("history_valid", (d, 2, 8), bool),
        ("history_sample_ages", (d, 2, 8), np.int64),
        ("history_receipt_ages", (d, 2, 8), np.int64),
        ("chosen", (d, 2, 700), np.float32),
        ("applied", (n, 2, 700), np.float32),
        ("excitation", (n, 2, 700), np.float64),
        ("packets", (n, 2, 2), np.int64),
    ]:
        typed(tape[name], shape, dtype, name)
    require(
        len(trace["intervals"]) == n == len(trace["state_sha256"])
        and len(trace["decisions"]) == d == len(trace["rngs_before"])
        and len(sensor_trace["frames"]) == d == len(sensor_trace["rngs_after"]),
        "Sensor trace denominator",
    )
    require(sensor_trace["schema"] == "joxing-sensor-evidence-v1", "Sensor schema")
    require(set(schedule) == set(delays) == set(PLAYERS), "Both timing owners required")
    require(
        all(
            isinstance(s, (list, tuple))
            and 1 <= len(s) <= 16
            and all(x is None or type(x) is int and 0 <= x <= 40 for x in s)
            for s in schedule.values()
        )
        and all(type(x) is int and 0 <= x <= 40 for x in delays.values()),
        "Timing contract bounds",
    )
    require(np.all(np.abs(tape["chosen"]) <= 1), "Normalized command range")
    require(
        np.array_equal(excitation(tape["applied"]), tape["excitation"]),
        "Once-only muscle decoding",
    )
    queues = copy.deepcopy(initial["sensors"])
    history, pending, records = {}, {}, {}
    for a in PLAYERS:
        history[a] = {r["sampled_tick"]: r for r in queues[a]["history"]["samples"]}
        pending[a] = {p["sampled_tick"]: p for p in queues[a]["pending"]}
        records[a] = {r["sampled_tick"]: r for r in queues[a]["records"]}
    emitted = []
    rngs = {a: np.random.default_rng() for a in PLAYERS}
    for a in PLAYERS:
        rngs[a].bit_generator.state = app["rngs"][a]
    calls = {a: 0 for a in PLAYERS}
    index_by_tick = {int(t): i for i, t in enumerate(decisions)}
    for t in range(start, end + 1):
        if t > start:
            if t % 20 == 0:
                require(t in samples, "Missing true sampled state")
                for j, a in enumerate(PLAYERS):
                    delay = schedule[a][(t // 20) % len(schedule[a])]
                    packet = {
                        "game_id": app["game_id"],
                        "player": a,
                        "schema_hash": SCHEMA_HASH,
                        "sampled_tick": t,
                        "delivery_tick": t + (delay or 0),
                        "values": samples[t][j].tolist(),
                    }
                    record = {
                        "sampled_tick": t,
                        "delivery_tick": packet["delivery_tick"],
                        "sha256": digest(packet),
                        "received_tick": None,
                        "disposition": "dropped" if delay is None else "pending",
                        "dropped_at_source": delay is None,
                    }
                    records[a][t] = record
                    emitted.append({"player": a, "event": "sampled", **record})
                    if delay is not None:
                        pending[a][t] = packet
            for a in PLAYERS:
                due = sorted(
                    [p for p in pending[a].values() if p["delivery_tick"] <= t],
                    key=lambda p: (p["delivery_tick"], p["sampled_tick"]),
                )
                for p in due:
                    sampled = p["sampled_tick"]
                    latest = max(history[a], default=-1)
                    history[a][sampled] = {
                        "sampled_tick": sampled,
                        "received_tick": t,
                        "values": p["values"],
                    }
                    for old in sorted(history[a])[:-8]:
                        del history[a][old]
                    record = records[a][sampled]
                    record.update(
                        received_tick=t,
                        disposition="delivered" if sampled > latest else "late_history",
                    )
                    emitted.append({"player": a, "event": "received", **record})
                    del pending[a][sampled]
        if t not in index_by_tick:
            continue
        i = index_by_tick[t]
        require(trace["decisions"][i]["tick"] == t, "Decision trace clock")
        for j, a in enumerate(PLAYERS):
            keep = sorted(history[a])[-8:]
            values = np.zeros((8, 1970), np.float32)
            times = np.full(8, -1, np.int64)
            arrivals = times.copy()
            for k, sampled in enumerate(keep, 8 - len(keep)):
                h = history[a][sampled]
                values[k], times[k], arrivals[k] = (
                    h["values"],
                    sampled,
                    h["received_tick"],
                )
            for name, expected in [
                ("histories", values),
                ("history_ticks", times),
                ("history_received", arrivals),
                ("history_valid", times >= 0),
                ("history_sample_ages", np.where(times >= 0, t - times, -1)),
                ("history_receipt_ages", np.where(times >= 0, t - arrivals, -1)),
                ("observations", values[-1]),
            ]:
                require(np.array_equal(tape[name][i, j], expected), "Received " + name)
            available = bool(keep)
            expected_frame = {
                "decision_tick": t,
                "available": available,
                "actor_called": available,
                "sample_age_ticks": t - int(times[-1]) if available else -1,
                "receipt_age_ticks": t - int(arrivals[-1]) if available else -1,
                "observation_owner": {
                    "game_id": "actor-visible",
                    "player": a,
                    "sampled_tick": int(times[-1]),
                    "schema_hash": SCHEMA_HASH,
                }
                if available
                else None,
            }
            require(
                digest(sensor_trace["frames"][i][a]) == digest(expected_frame),
                "Actor boundary/privacy",
            )
            require(
                rngs[a].bit_generator.state == trace["rngs_before"][i][a],
                "Policy RNG before decision",
            )
            if available:
                command = diagnostic_command(
                    SimpleNamespace(values=values[-1]),
                    SimpleNamespace(values=values, valid=times >= 0),
                    rngs[a],
                )
                calls[a] += 1
            else:
                command = (
                    np.array(app["commands"][a], np.float32)
                    if t == start
                    else tape["applied"][t - start - 1, j]
                )
            require(
                np.array_equal(command, tape["chosen"][i, j]), "Received-input command"
            )
            require(
                rngs[a].bit_generator.state == sensor_trace["rngs_after"][i][a],
                "Policy RNG after decision/hold",
            )
    require(
        emitted == sensor_trace["events"], "Missing/reordered/wrong-time sensor receipt"
    )
    for a in PLAYERS:
        queues[a]["last_polled_tick"] = end
        queues[a]["pending"] = [p for _, p in sorted(pending[a].items())]
        queues[a]["records"] = list(records[a].values())
        queues[a]["history"]["samples"] = [h for _, h in sorted(history[a].items())]
        require(queues[a] == final["sensors"][a], "Final pending/history sensor state")
        require(
            rngs[a].bit_generator.state == finish["application"]["rngs"][a], "Final RNG"
        )
    # Command queues are reconstructed separately, including pending pre-checkpoint
    # commands and the17held substeps after a +3mid-control checkpoint.
    for j, a in enumerate(PLAYERS):
        q = app["queues"][a]
        current = np.array(app["commands"][a], np.float32)
        last = q["last_applied"]
        receipt = (
            [-1, -1] if last is None else [last["decision_tick"], last["apply_tick"]]
        )
        commands = [
            (p["decision_tick"], p["apply_tick"], np.array(p["command"], np.float32))
            for p in q["pending"]
        ]
        for k in range(n):
            t = start + k
            if t in index_by_tick:
                commands.append((t, t + delays[a], tape["chosen"][index_by_tick[t], j]))
            due = [p for p in commands if p[1] <= t]
            commands = [p for p in commands if p[1] > t]
            if due:
                winner = max(due, key=lambda p: p[0])
                if winner[0] > receipt[0]:
                    receipt, current = list(winner[:2]), winner[2]
            require(
                np.array_equal(tape["applied"][k, j], current)
                and tape["packets"][k, j].tolist() == receipt,
                "Command transport/held tail",
            )
        require(
            np.array_equal(
                np.array(finish["application"]["commands"][a], np.float32), current
            ),
            "Final held command",
        )
    return {"status": "transport_passed", "physics_observed": n, "actor_calls": calls}


def audit_sensor_transport(folder, scene, *, expected_physics, delays):
    folder = Path(folder)
    initial = json.loads((folder / "sensor-initial.json").read_text())
    final = json.loads((folder / "sensor-final.json").read_text())
    trace = json.loads((folder / "trace.json").read_text())
    sensor_trace = json.loads((folder / "sensor-trace.json").read_text())
    contract = json.loads((folder / "contract.json").read_text())
    env = SensedBoxingEnv(
        scene,
        horizon_ticks=contract["horizon_ticks"],
        bundles=contract["bundles"],
        scenario=contract["reset_scenario"],
        sensor_schedule=contract["sensor_schedule"],
    )
    require(env.contract == contract, "Foreign sensed contract")
    env.restore(initial)  # Semantic validation/FK, no physics.
    for name, snapshot in [("initial", initial), ("final", final)]:
        require(
            json.loads((folder / (name + ".json")).read_text()) == snapshot["base"],
            "Sensor/base snapshot disagreement",
        )
    tape, sampled = (
        read_arrays(folder / "tape.npz"),
        read_arrays(folder / "sensor-tape.npz"),
    )
    for name in ["history_sample_ages", "history_receipt_ages"]:
        tape[name] = sampled[name]
    start, end = env.tick, final["base"]["application"]["tick"]
    require(end - start == expected_physics, "Expected physics denominator")
    ticks = np.arange((start + 19) // 20 * 20, end + 1, 20, dtype=np.int64)
    width = len(initial["base"]["engine"])
    typed(
        tape["decision_states"],
        (len(tape["decision_ticks"]), width),
        np.float64,
        "decision_states",
    )
    for name, shape, dtype in [
        ("sample_ticks", ticks.shape, np.int64),
        ("sample_engines", (len(ticks), width), np.float64),
        ("sample_commands", (len(ticks), 2, 700), np.float32),
        ("sample_values", (len(ticks), 2, 1970), np.float32),
    ]:
        typed(sampled[name], shape, dtype, name)
    require(np.array_equal(sampled["sample_ticks"], ticks), "Actual sample cadence")
    require(
        len(sensor_trace["sample_points"]) == len(ticks), "Sample points denominator"
    )
    sample_values = {}
    for i, t in enumerate(ticks.tolist()):
        raw = sampled["sample_engines"][i]
        expected_sha = (
            array_sha(np.array(initial["base"]["engine"], np.float64))
            if t == start
            else trace["state_sha256"][t - start - 1]
        )
        require(
            array_sha(raw) == expected_sha, "Sample engine differs from physical tape"
        )
        require(abs(raw[0] - t * 0.001) < 1e-9, "Sample engine clock")
        scratch = mujoco.MjData(env.model)
        mujoco.mj_setState(env.model, scratch, raw, INTEGRATION)
        require(
            not np.any(scratch.qfrc_applied) and not np.any(scratch.xfrc_applied),
            "Sample assistance",
        )
        mujoco.mj_forward(env.model, scratch)
        require(not any(w.number for w in scratch.warning), "Sample engine warning")
        commands = dict(zip(PLAYERS, sampled["sample_commands"][i], strict=True))
        expected_commands = (
            np.stack([env.commands[a] for a in PLAYERS])
            if t == start
            else tape["applied"][t - start - 1]
        )
        require(
            np.array_equal(sampled["sample_commands"][i], expected_commands),
            "Sample command clock",
        )
        for a in PLAYERS:
            require(
                np.array_equal(
                    scratch.ctrl[env.bodies[a].actuator_ids], excitation(commands[a])
                ),
                "Sample engine controls",
            )
        observations = env._make_observations(
            scratch, t, sensor_trace["sample_points"][i], commands
        )
        require(
            np.array_equal(
                np.stack([observations[a].values for a in PLAYERS]),
                sampled["sample_values"][i],
            ),
            "Sample features differ from actual sampled engine",
        )
        sample_values[t] = sampled["sample_values"][i]
        at_decision = np.flatnonzero(tape["decision_ticks"] == t)
        if len(at_decision):
            require(
                np.array_equal(tape["decision_states"][at_decision[0]], raw),
                "Decision engine differs from actual sample",
            )
    result = audit_transport(
        initial,
        final,
        trace,
        sensor_trace,
        tape,
        sample_values,
        contract["sensor_schedule"],
        delays,
    )
    env.restore(final)
    require(env.physics_total == 0, "Audit stepped physics")
    return result | {
        "audit_new_physics": 0,
        "sampled_engines_checked": len(ticks),
        "physical_referee_audited": False,
        "formal_sensor_proof": False,
        "limitations": [
            "Transport component only; physical referee, source/replay linkage and D0 oracle gates remain mandatory",
            "Reuses frozen FK/controller and semantic snapshot validation",
        ],
    }


def compare_sensor_replay(source, replay, *, source_checkpoint):
    """Compare all received/physical tails against their actual source tape."""
    source, replay = Path(source), Path(replay)
    physical = compare_replay(source, replay)
    initial = json.loads((replay / "sensor-initial.json").read_text())
    require(
        initial == json.loads(Path(source_checkpoint).read_text()),
        "Replay actual source checkpoint",
    )
    start = initial["base"]["application"]["tick"]
    s = json.loads((source / "sensor-trace.json").read_text())
    r = json.loads((replay / "sensor-trace.json").read_text())
    require(
        json.loads((source / "sensor-final.json").read_text())
        == json.loads((replay / "sensor-final.json").read_text()),
        "Replay final sensor state",
    )
    require(
        [f for f in s["frames"] if f["A"]["decision_tick"] >= start] == r["frames"],
        "Replay actor input metadata tail",
    )
    source_tape = read_arrays(source / "tape.npz")
    selected = source_tape["decision_ticks"] >= start
    require(
        [rng for rng, keep in zip(s["rngs_after"], selected, strict=True) if keep]
        == r["rngs_after"],
        "Replay post-policy RNG tail",
    )

    def event_clock(event):
        return (
            event["sampled_tick"]
            if event["event"] == "sampled"
            else event["received_tick"]
        )

    require(
        [e for e in s["events"] if event_clock(e) > start] == r["events"],
        "Replay sensor receipt tail",
    )
    a, b = (
        read_arrays(source / "sensor-tape.npz"),
        read_arrays(replay / "sensor-tape.npz"),
    )
    selected = a["sample_ticks"] >= start
    require(set(a) == set(b), "Replay sample evidence schema")
    for k in a:
        keep = (
            source_tape["decision_ticks"] >= start
            if k.startswith("history_")
            else selected
        )
        require(np.array_equal(a[k][keep], b[k]), "Replay actual sample tail: " + k)
    require(
        [p for p, keep in zip(s["sample_points"], selected, strict=True) if keep]
        == r["sample_points"],
        "Replay sample points tail",
    )
    return physical | {"sensor_exact": True, "audit_new_physics": 0}


def compare_zero_delay_oracle(oracle, sensed):
    """Exact saved-array comparison; caller must qualify source-host provenance.

    Intentionally distinct environment/snapshot contract hashes are excluded;
    all underlying application state, engines, numeric arrays and physical trace
    must match. This function never rebuilds a scene or reruns the old oracle.
    """
    oracle, sensed = Path(oracle), Path(sensed)
    old_contract = json.loads((oracle / "contract.json").read_text())
    new_contract = json.loads((sensed / "contract.json").read_text())
    require(
        new_contract["sensor_schedule"] == {a: [0] for a in PLAYERS},
        "Oracle requires zero sensor delay",
    )
    require(
        all(new_contract.get(k) == v for k, v in old_contract.items() if k != "schema"),
        "Oracle physical/controller contract differs",
    )
    for name in ["initial", "final"]:
        old = json.loads((oracle / (name + ".json")).read_text())
        new = json.loads((sensed / (name + ".json")).read_text())
        require(old["engine"] == new["engine"], "Oracle engine: " + name)
        require(
            old["application"] == new["application"], "Oracle base application: " + name
        )
    old_trace = json.loads((oracle / "trace.json").read_text())
    new_trace = json.loads((sensed / "trace.json").read_text())
    for trace, contract in [(old_trace, old_contract), (new_trace, new_contract)]:
        require(
            trace["transition"]["info"].pop("contract_hash") == digest(contract),
            "Oracle transition contract identity",
        )
    require(old_trace == new_trace, "Oracle physical/decision trace")
    a, b = read_arrays(oracle / "tape.npz"), read_arrays(sensed / "tape.npz")
    require(set(a) == set(b), "Oracle array schema")
    for k in a:
        require(
            a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]),
            "Oracle arrays: " + k,
        )
    return {"status": "passed", "exact_saved_oracle": True, "audit_new_physics": 0}
