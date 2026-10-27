"""Read every H3 tape/contact array independently; never run new physics."""

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, write_json
from joxing.stance_train import file_hash

PROTOCOL = "b4ad191f93a6645557ebeb23e8a5cd80e643370195cc0a8423de4a150cc41373"
ACTOR = "963d2f56c5887f79e094ddf998316730475f979efb100073237cda81817e272e"
NORMALIZER = "34f38e6e87544d77b87b9a155fe6b9996629fd671f235b773c27f004d6fd5df8"


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    assert all(np.isfinite(a).all() for a in result.values()), path
    return result


def same_arrays(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].dtype == right[key].dtype, key
        assert np.array_equal(left[key], right[key]), key


def check_contacts(data, count, floor, roles):
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
    assert offsets.shape == (count + 1,) and offsets[0] == 0
    assert np.issubdtype(offsets.dtype, np.integer) and np.all(np.diff(offsets) >= 0)
    contacts = int(offsets[-1])
    assert data["intervals_s"].shape == (count, 2)
    assert data["loads_N"].shape == (count, 3)
    assert data["tilt_drop"].shape == (count, 2)
    assert data["contact_geom_ids"].shape == (contacts, 2)
    assert data["contact_geom_ids"].dtype == np.int32
    assert data["contact_role"].shape == (contacts,)
    assert data["contact_role"].dtype == np.int8
    assert data["contact_force_N"].shape == (contacts, 3)
    assert data["contact_position_m"].shape == (contacts, 3)
    assert np.all(data["intervals_s"][:, 1] > data["intervals_s"][:, 0])
    names = ["right", "left", "nonfoot"]
    for i, (start, end) in enumerate(pairwise(offsets)):
        loads = np.zeros(3)
        for j in range(int(start), int(end)):
            g1, g2 = data["contact_geom_ids"][j]
            assert (g1 == floor) != (g2 == floor)
            other = g2 if g1 == floor else g1
            role = int(data["contact_role"][j])
            assert role in range(3) and roles[str(other)] == names[role]
            loads[role] += max(0.0, float(data["contact_force_N"][j, 2]))
        assert np.array_equal(loads, data["loads_N"][i]), i
    return contacts


def audit(folder, receipts):
    r, p = read(folder / "result.json"), read(folder / "protocol.json")
    assert r["protocol_sha256"] == digest(p) == PROTOCOL
    assert p == read(ROOT / "experiments/v0.0.9/collector-protocol.json")
    source = subprocess.check_output(
        ["git", "show", r["code_commit"] + ":scripts/benchmark_collector.py"], cwd=ROOT
    )
    assert hashlib.sha256(source).hexdigest() == r["script_sha256"]
    paths = subprocess.check_output(
        ["git", "ls-tree", "--name-only", r["code_commit"], "src/joxing/"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    contents = {
        path: subprocess.check_output(
            ["git", "show", r["code_commit"] + ":" + path], cwd=ROOT
        ).decode()
        for path in paths
        if path.endswith(".py")
    }
    assert contents and digest(contents) == r["source_hash"]
    for path, expected in p["source_files"].items():
        assert file_hash(ROOT / path) == expected, path
    assert r["status"] == "completed" and r["physics_accounting_exact"]
    assert r["boxing_games"] == 0 and r["wall_s"] < p["caps"]["wall_s"]
    assert r["maximum_allocated_physics_steps"] == p["planned_physics_steps"] == 82394
    assert r["numpy"] == "2.4.2" and r["mujoco"] == "3.8.0"
    test_log = (receipts / "linux-tests.txt").read_text()
    assert "165 passed" in test_log and "FAILED" not in test_log
    exit_receipt = read(receipts / "process-exit.json")
    assert exit_receipt["head"] == r["code_commit"]
    assert not exit_receipt["owned_processes"] and not exit_receipt["source_diff"]
    exited = {x["pid"] for x in exit_receipt["recorded_processes"] if not x["exists"]}
    assert len(exited) == len(exit_receipt["recorded_processes"])
    supervisor = read(receipts / "supervisor.json")
    assert supervisor["status"] == "completed"
    assert all(
        s["status"] == "completed" and s["returncode"] == 0
        for s in supervisor["stages"]
    )
    assert supervisor["source_commit"] == r["code_commit"]
    base = ROOT / "artifacts/v0.0.8/physical/native_2ms"
    reference = read(base / "replay.json")
    specification = read(base / "specification.json")
    roles = specification["floor_roles"]
    source_contacts = arrays(base / "substeps-74000.npz")
    # Independently infer the unique shared geom from the hash-pinned floor-only source tape.
    shared = set.intersection(
        *(
            set(map(int, pair))
            for pair in np.unique(source_contacts["contact_geom_ids"], axis=0)
        )
    )
    assert len(shared) == 1
    floor = shared.pop()
    expected_qualification = [
        (s, m)
        for s in ("boundary", "mid_control")
        for m in ("recorded_actions", "closed_loop")
    ]
    assert [
        (x["snapshot"], x["path"]) for x in r["qualification"]
    ] == expected_qualification
    assert read(folder / "qualification/result.json")["cases"] == r["qualification"]
    for row in r["qualification"]:
        assert row["status"] == "completed" and row["source_state_contact_action_exact"]
        first = 100 if row["snapshot"] == "boundary" else 103
        assert (
            row["first_source_substep"] == first
            and row["physics_steps"] == 2700 - first
        )
        stem = row["snapshot"] + "-" + row["path"]
        assert (
            read(folder / "qualification" / (stem + "-final.json"))
            == reference["final_snapshot"]
        )
        offset = int(source_contacts["contact_offsets"][first])
        expected = {
            k: a[first:] - offset
            if k == "contact_offsets"
            else a[first:]
            if k in {"intervals_s", "loads_N", "tilt_drop"}
            else a[offset:]
            for k, a in source_contacts.items()
        }
        same_arrays(expected, arrays(folder / "qualification" / (stem + ".npz")))
    expected_waves = [
        (repeat, arm)
        for repeat in range(3)
        for arm in (p["arms"] if repeat % 2 == 0 else p["arms"][::-1])
    ]
    assert [(w["repeat"], w["arm"]) for w in r["waves"]] == expected_waves
    findings, anchors = [], {}
    for wave in r["waves"]:
        assert wave["status"] == "completed" and len(wave["cases"]) == 4
        assert [x["seed"] for x in wave["cases"]] == p["seeds"]
        capacity = wave["capacity_before"]
        assert (
            capacity["available_memory_bytes"]
            >= 4 * p["worker_memory_reservation_bytes"]
            + p["caps"]["minimum_memory_bytes"]
        )
        assert capacity["disk_free_bytes"] >= p["caps"]["minimum_disk_bytes"]
        assert 1200 / wave["steady_parent_s"] == wave["steady_control_steps_per_s"]
        assert (
            1200 / wave["startup_inclusive_s"]
            == wave["startup_inclusive_control_steps_per_s"]
        )
        assert wave["startup_inclusive_s"] >= wave["steady_parent_s"] > 0
        records = {x["case"]["case_id"]: x for x in wave["worker_receipts"]}
        assert len(records) == 4
        for row in wave["cases"]:
            case_id = f"{wave['repeat']}:{wave['arm']}:{row['seed']}"
            assert row["case_id"] == case_id and row["arm"] == wave["arm"]
            assert row["status"] == "completed" and row["paired_exact"]
            assert (
                row["controls"] == 300
                and row["physics_steps"] == row["substeps"] == 3000
            )
            case_dir = folder / case_id.replace(":", "-")
            private = read(case_dir / "progress.json")
            record = records[case_id]
            assert private == record
            assert private["case"] == {
                k: v for k, v in row.items() if k != "paired_exact"
            }
            assert record["phase"] == "completed" and record["threads"] == 1
            assert (
                record["actor_hash"] == ACTOR
                and record["normalizer_stats_hash"] == NORMALIZER
            )
            assert record["pid"] in exited
            assert record["peak_rss_KiB"] * 1024 < p["worker_memory_reservation_bytes"]
            initial, final = (
                read(case_dir / "initial.json"),
                read(case_dir / "final.json"),
            )
            assert (
                digest(initial) == row["initial_hash"]
                and digest(final) == row["final_hash"]
            )
            assert (
                len(row["pre_snapshot_hashes"]) == 300
                and row["pre_snapshot_hashes"][0] == row["initial_hash"]
            )
            for filename, field in [
                ("trajectory.npz", "arrays_sha256"),
                ("substeps.npz", "substeps_sha256"),
                ("trace.json", "trace_sha256"),
            ]:
                assert file_hash(case_dir / filename) == row[field]
            tape, contacts = (
                arrays(case_dir / "trajectory.npz"),
                arrays(case_dir / "substeps.npz"),
            )
            assert set(tape) == {"actions", "observations", "states"}
            assert (
                tape["actions"].shape == (300, 700)
                and tape["actions"].dtype == np.float32
            )
            assert (
                tape["observations"].shape == (300, 3601)
                and tape["observations"].dtype == np.float32
            )
            assert (
                tape["states"].shape == (300, len(initial["state"]))
                and tape["states"].dtype == np.float64
            )
            assert np.array_equal(tape["states"][0], initial["state"])
            assert np.array_equal(
                tape["observations"][0],
                np.asarray(initial["last_raw_observation"], dtype=np.float32),
            )
            contact_count = check_contacts(contacts, 3000, floor, roles)
            trace = read(case_dir / "trace.json")
            assert len(trace) == 300
            expected_resets = [{"control_index": 0, "seed": row["seed"]}]
            for i in range(300):
                end = 10 * (i + 1) - 1
                t = trace[i]
                assert t["time_s"] == contacts["intervals_s"][end, 1]
                assert [
                    t["floor_loads_N"][k] for k in ("right", "left", "nonfoot")
                ] == contacts["loads_N"][end].tolist()
                assert [t["tilt_rad"], t["drop_m"]] == contacts["tilt_drop"][
                    end
                ].tolist()
                if i < 299 and (t["terminated"] or t["truncated"]):
                    expected_resets.append(
                        {
                            "control_index": i + 1,
                            "seed": row["seed"] + 1000000 * len(expected_resets),
                        }
                    )
            assert row["resets"] == expected_resets
            starts = [x["control_index"] for x in row["resets"]] + [300]
            for start, end in pairwise(starts):
                intervals = contacts["intervals_s"][10 * start : 10 * end]
                assert intervals[0, 0] == 0 and np.array_equal(
                    intervals[1:, 0], intervals[:-1, 1]
                )
                assert np.allclose(
                    intervals[:, 1] - intervals[:, 0], 0.002, rtol=0, atol=1e-15
                )
            identity = {
                k: row[k]
                for k in [
                    "resets",
                    "initial_hash",
                    "final_hash",
                    "pre_snapshot_hashes",
                    "arrays_sha256",
                    "substeps_sha256",
                    "trace_sha256",
                ]
            }
            if row["seed"] not in anchors:
                anchors[row["seed"]] = (identity, tape, contacts, trace, initial, final)
            else:
                a = anchors[row["seed"]]
                assert (
                    identity == a[0]
                    and trace == a[3]
                    and initial == a[4]
                    and final == a[5]
                )
                same_arrays(a[1], tape)
                same_arrays(a[2], contacts)
            findings.append(
                {
                    "case_id": case_id,
                    "physics_steps": 3000,
                    "reset_count": len(row["resets"]) - 1,
                    "contacts": contact_count,
                    "raw_exact": True,
                }
            )
    physics = sum(x["physics_steps"] for x in findings) + sum(
        x["physics_steps"] for x in r["qualification"]
    )
    assert physics == r["physics_steps"] == 82394 < p["caps"]["physics_steps"]
    raw_bytes = sum(
        f.stat().st_size
        for f in folder.rglob("*")
        if f.is_file() and f.name != "audit.json"
    )
    assert raw_bytes < p["caps"]["raw_bytes"]
    medians = {
        arm: statistics.median(
            w["steady_control_steps_per_s"] for w in r["waves"] if w["arm"] == arm
        )
        for arm in p["arms"]
    }
    startup = {
        arm: statistics.median(
            w["startup_inclusive_control_steps_per_s"]
            for w in r["waves"]
            if w["arm"] == arm
        )
        for arm in p["arms"]
    }
    ratio, startup_ratio = (
        medians["allocation_reduced"] / medians["baseline"],
        startup["allocation_reduced"] / startup["baseline"],
    )
    gate = (
        ratio >= p["criteria"]["minimum_steady_speedup"]
        and startup_ratio >= p["criteria"]["minimum_startup_ratio"]
    )
    assert (
        medians == r["median_steady_control_steps_per_s"]
        and startup == r["median_startup_inclusive_control_steps_per_s"]
    )
    assert ratio == r["steady_speedup"] and gate == r["performance_gate_met"]
    return {
        "passed": True,
        "auditor_sha256": file_hash(Path(__file__)),
        "source_commit": r["code_commit"],
        "scope": "Every saved raw array/dtype, trace, initial/final snapshot and receipt checked; four source-v8 tails exact. Pre-snapshot hash lists compared, not independently regenerated: full intermediate application snapshots were not saved. Frozen source verifies closed-loop action assertions and in-flight resource/deadline guards; no new physics/actor execution or independent peak sampling.",
        "qualification_counts": dict(Counter(x["status"] for x in r["qualification"])),
        "timed_counts": dict(
            Counter(x["status"] for w in r["waves"] for x in w["cases"])
        ),
        "invalid": 0,
        "incomplete": 0,
        "not_run": 0,
        "physics_steps": physics,
        "new_audit_physics_steps": 0,
        "raw_bytes": raw_bytes,
        "steady_medians": medians,
        "startup_inclusive_medians": startup,
        "steady_speedup": ratio,
        "startup_ratio": startup_ratio,
        "performance_gate_met": gate,
        "decision": "qualifies" if gate else "decline; reference default retained",
        "findings": findings,
        "result_sha256": file_hash(folder / "result.json"),
        "linux_tests_sha256": file_hash(receipts / "linux-tests.txt"),
        "recorded_exited_pids": len(exited),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder, args.receipts)
    write_json(args.folder / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, indent=2))
