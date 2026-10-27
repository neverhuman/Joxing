"""Independent D2 raw-evidence arithmetic; never steps or predicts a teacher."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .provenance import digest

ROLES = ("right", "left", "nonfoot")


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    assert all(np.isfinite(v).all() for v in result.values()), str(path)
    return result


def decoded(tape, groups):
    n = len(tape["residuals"])
    assert tape["residuals"].shape == (n, 138)
    assert tape["teachers"].shape == tape["commands"].shape == (n, 700)
    for key in ("residuals", "teachers", "commands"):
        assert tape[key].dtype == np.float32 and np.max(abs(tape[key]), initial=0) <= 1
    assert len(groups) == 138 and all(groups)
    assert sorted(i for group in groups for i in group) == list(range(700))
    expanded = np.empty((n, 700), np.float32)
    for group, indices in enumerate(groups):
        expanded[:, indices] = tape["residuals"][:, group, None]
    expected = np.clip(tape["teachers"] + np.float32(0.1) * expanded, -1, 1)
    assert np.array_equal(expected, tape["commands"]), "decoded700 command"


def contacts(data, count, floor, roles):
    assert set(data) == {
        "intervals_s",
        "loads_N",
        "tilt_drop",
        "contact_offsets",
        "contact_geom_ids",
        "contact_role",
        "contact_force_N",
        "contact_position_m",
    }
    offsets = data["contact_offsets"]
    assert (
        offsets.shape == (count + 1,)
        and offsets[0] == 0
        and np.all(np.diff(offsets) >= 0)
    )
    assert np.issubdtype(offsets.dtype, np.integer)
    n = int(offsets[-1])
    assert (
        data["intervals_s"].shape == (count, 2)
        and data["intervals_s"].dtype == np.float64
    )
    assert data["loads_N"].shape == data["tilt_drop"].shape[:1] + (3,) == (count, 3)
    assert data["tilt_drop"].shape == (count, 2)
    assert (
        data["contact_geom_ids"].shape == (n, 2)
        and data["contact_geom_ids"].dtype == np.int32
    )
    assert data["contact_role"].shape == (n,) and data["contact_role"].dtype == np.int8
    assert data["contact_force_N"].shape == data["contact_position_m"].shape == (n, 3)
    assert all(
        data[k].dtype == np.float64
        for k in ["loads_N", "tilt_drop", "contact_force_N", "contact_position_m"]
    )
    for i in range(count):
        loads = np.zeros(3)
        for j in range(int(offsets[i]), int(offsets[i + 1])):
            g1, g2 = data["contact_geom_ids"][j]
            role = int(data["contact_role"][j])
            assert (g1 == floor) != (g2 == floor)
            other = int(g2 if g1 == floor else g1)
            assert role in range(3) and roles[str(other)] == ROLES[role]
            loads[role] += max(0.0, float(data["contact_force_N"][j, 2]))
        assert np.array_equal(loads, data["loads_N"][i]), ("signed floor load", i)
    return n


def first_latch(data, weight, initial=None):
    accum = (
        [0.0, 0.0, 0.0]
        if initial is None
        else [initial[k] for k in ("nonfoot_s", "tilt_s", "drop_s")]
    )
    first = reason = None
    for i, (start, end) in enumerate(data["intervals_s"]):
        assert end > start and abs(end - start - 0.002) < 1e-9
        if i:
            assert start == data["intervals_s"][i - 1, 1]
        tests = [
            data["loads_N"][i, 2] >= 0.1 * weight,
            data["tilt_drop"][i, 0] >= np.pi / 3,
            data["tilt_drop"][i, 1] >= 0.3,
        ]
        accum = [
            v + 0.002 if condition else 0.0
            for v, condition in zip(accum, tests, strict=True)
        ]
        for name, elapsed, threshold in zip(
            ("nonfoot_support", "excess_tilt", "pelvis_drop"),
            accum,
            (0.05, 0.1, 0.1),
            strict=True,
        ):
            if first is None and elapsed + 1e-12 >= threshold:
                first, reason = float(end), name
    return (
        first,
        reason,
        dict(zip(("nonfoot_s", "tilt_s", "drop_s"), accum, strict=True)),
    )


def reward_row(row, begin, reward_key="reward"):
    end = row["telemetry"]["time_s"]
    failure = row["telemetry"]["physical_failure_s"]
    assert failure is None or begin <= failure <= end
    alive = max(0.0, min(end, end if failure is None else failure) - begin)
    assert row[reward_key] == alive - float(failure is not None)
    assert row["terminated"] == (failure is not None)
    assert row["truncated"] == bool(failure is None and end + 1e-9 >= 20.0)
    assert row["bootstrap_mask"] == float(not row["terminated"])
    assert row["trace_mask"] == float(not (row["terminated"] or row["truncated"]))
    assert np.isfinite(row["native_reward"])
    return end


def normalized(raw, stats):
    # VecNormalize's frozen observation arithmetic, independently reconstructed.
    raw = np.asarray(raw, np.float32)
    return np.clip(
        (raw - np.asarray(stats["mean"]))
        / np.sqrt(np.asarray(stats["var"]) + stats["epsilon"]),
        -stats["clip"],
        stats["clip"],
    ).astype(np.float32)


def evaluation(folder, groups, context):
    folder = Path(folder)
    row = read(folder / "result.json")
    assert row["status"] == "completed" and row["physics_accounting_exact"]
    trace = read(folder / "trace.json")
    tape = arrays(folder / "actions.npz")
    substeps = arrays(folder / "substeps.npz")
    n = len(trace)
    assert n == row["controls"] == len(tape["commands"]) and n > 0
    decoded(tape, groups)
    if row["stage"] == "initial":
        assert np.array_equal(tape["residuals"], np.zeros((n, 138), np.float32))
    initial = read(folder / "initial.json")
    final = read(folder / "final.json")
    for snapshot in (initial, final):
        assert (
            snapshot["bundle_hash"] == row["bundle_hash"]
            and snapshot["episode_seed"] == row["evaluation_seed"]
        )
        assert snapshot["reference_dtypes"] == {
            "qpos_ref": "<f4",
            "xpos_ref": "<f4",
            "qpos_ref_future": "<f8",
        }
        for key, dtype in snapshot["reference_dtypes"].items():
            values = snapshot["physical"]["application"][key]
            assert np.array_equal(
                np.asarray(values, dtype=dtype).astype(np.float64), values
            )
    assert initial["held"] is None and not initial["physical"]["done"]
    assert initial["physical"]["physical_control"]["completed_controls"] == 0
    assert final["held"] is None and final["physical"]["done"]
    assert final["physical"]["physical_control"]["completed_controls"] == n
    assert row["physics_steps"] == 10 * n == len(substeps["intervals_s"])
    count = contacts(
        substeps,
        10 * n,
        context["floor_geom_id"],
        context["physical_specification"]["floor_roles"],
    )
    assert substeps["intervals_s"][0, 0] == 0.0
    weight = float(sum(context["physical_specification"]["body_mass"]) * 9.81)
    failure, reason, monitor = first_latch(substeps, weight)
    assert row["physical_failure"] == reason
    assert final["physical"]["physical_control"]["physical_failure_s"] == failure
    assert final["physical"]["physical_control"]["physical_failure"] == reason
    assert final["physical"]["monitor"] == monitor
    assert row["survival_s"] == min(20.0, 20.0 if failure is None else failure)
    begin = 0.0
    for i, t in enumerate(trace):
        assert t["control_index"] == i and 0 <= t["teacher_clipped_fraction"] <= 1
        assert len(t["observation_hash"]) == 64
        prior = begin
        begin = reward_row(t, begin)
        failure_in_control = t["telemetry"]["physical_failure_s"]
        assert t["alive_seconds"] == max(
            0.0,
            min(begin, begin if failure_in_control is None else failure_in_control)
            - prior,
        )
        last = (i + 1) * 10 - 1
        assert begin == substeps["intervals_s"][last, 1]
        assert np.array_equal(
            [t["telemetry"]["floor_loads_N"][k] for k in ROLES],
            substeps["loads_N"][last],
        )
        assert t["telemetry"]["tilt_rad"] == substeps["tilt_drop"][last, 0]
        assert t["telemetry"]["drop_m"] == substeps["tilt_drop"][last, 1]
        assert not (t["terminated"] or t["truncated"]) or i == n - 1
    assert (
        row["terminated"] == trace[-1]["terminated"]
        and row["truncated"] == trace[-1]["truncated"]
    )
    assert row["motor_return"] == sum(t["reward"] for t in trace)
    if "normalizer" in context:
        stats = context["normalizer"]
        end_obs = normalized(final["physical"]["last_raw_observation"], stats)
        assert digest(end_obs.tolist()) == trace[-1]["observation_hash"]
    if row["mid_snapshot_available"]:
        boundary = read(folder / "boundary.json")
        mid = read(folder / "mid.json")
        assert boundary["physical"]["physical_control"]["completed_controls"] == 10
        assert boundary["held"] is None
        assert (
            mid["physical"]["physical_control"]["completed_controls"] == 10
            and mid["physical"]["physical_control"]["substep"] == 3
        )
        assert mid["physical"]["physical_control"]["pending"]
        assert np.array_equal(mid["held"]["residual"], tape["residuals"][10])
        assert np.array_equal(mid["held"]["teacher"], tape["teachers"][10])
        assert np.array_equal(mid["held"]["command"], tape["commands"][10])
    return {
        "evaluation_seed": row["evaluation_seed"],
        "stage": row["stage"],
        "controls": n,
        "physics_steps": row["physics_steps"],
        "survival_s": row["survival_s"],
        "contacts": count,
        "raw_verified": True,
    }


def replay_tail(folder, source):
    folder, source = Path(folder), Path(source)
    r = read(folder / "result.json")
    s = read(source / "result.json")
    assert r["status"] == "completed" and r["exact"] and r["physics_accounting_exact"]
    assert read(folder / "start.json") == read(source / "mid.json")
    assert read(folder / "final.json") == read(source / "final.json")
    assert read(folder / "trace.json") == read(source / "trace.json")[10:]
    a, b = arrays(source / "actions.npz"), arrays(folder / "actions.npz")
    assert set(a) == set(b)
    for key in a:
        assert a[key].dtype == b[key].dtype and np.array_equal(a[key][10:], b[key])
    a, b = arrays(source / "substeps.npz"), arrays(folder / "substeps.npz")
    first = 103
    offset = int(a["contact_offsets"][first])
    assert set(a) == set(b)
    for key in a:
        expected = (
            a[key][first:] - offset
            if key == "contact_offsets"
            else a[key][first:]
            if key in {"intervals_s", "loads_N", "tilt_drop"}
            else a[key][offset:]
        )
        assert expected.dtype == b[key].dtype and np.array_equal(expected, b[key]), key
    assert r["physics_steps"] == s["physics_steps"] - 103
    assert r["controls"] == s["controls"] - 10
    return {
        "passed": True,
        "physics_steps": r["physics_steps"],
        "new_audit_physics_steps": 0,
    }


def worker(folder, index, owner, groups):
    from .motor_pilot import reset_seed

    folder = Path(folder)
    progress = read(folder / "progress.json")
    assert progress["status"] == "closed" and progress["transitions"] == 2048
    assert progress["physics_steps"] == 20480 and progress["unflushed_transitions"] == 0
    assert not (folder / "fault.json").exists()
    lines = [
        json.loads(line)
        for line in (folder / "transitions.jsonl").read_text().splitlines()
    ]
    episodes = [
        json.loads(line)
        for line in (folder / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(lines) == 2048 and [r["episode_index"] for r in episodes] == list(
        range(len(episodes))
    )
    for r in episodes:
        assert r["seed"] == reset_seed(index, owner, r["episode_index"])
    files = sorted(folder.glob("commands-*.npz"))
    assert len(files) == progress["chunks"]
    chunks = [arrays(p) for p in files]
    tape = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    decoded(tape, groups)
    assert tape["rewards"].shape == (2048,) and tape["rewards"].dtype == np.float64
    episode = 0
    begin = 0.0
    completed = 0
    for i, r in enumerate(lines):
        assert r["transition"] == i + 1 and r["physics_steps"] == 10 * (i + 1)
        assert r["episode_index"] == episode and r["episode_seed"] == reset_seed(
            index, owner, episode
        )
        assert r["reward"] == tape["rewards"][i]
        begin = reward_row(r, begin)
        if r["terminated"] or r["truncated"]:
            episode += 1
            completed += 1
            begin = 0.0
    assert completed == progress["completed_motor_episodes"]
    assert len(episodes) == completed + 1 and progress["episode_index"] == completed
    assert (
        progress["bundle_hash"]
        == read(folder / "last-valid-snapshot.json")["bundle_hash"]
    )
    return {
        "worker_index": owner,
        "transitions": 2048,
        "physics_steps": 20480,
        "started_motor_episodes": len(episodes),
        "completed_motor_episodes": completed,
        "open_started_episodes": 1,
        "zero_step_open_episode": bool(
            lines[-1]["terminated"] or lines[-1]["truncated"]
        ),
    }


def status_counts(report):
    return {
        group: dict(Counter(r["status"] for r in report[group]))
        for group in ("evaluations", "training", "replays")
    }
