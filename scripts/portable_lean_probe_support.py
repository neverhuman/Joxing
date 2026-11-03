"""Explicit H30 fake-only packet boundary and saved guard checks; stdlib only."""

import hashlib
import json
from pathlib import Path

SCHEMA = "joxing-portable-lean-worker-qualification-v1"
INNER_SCHEMA = "joxing-lean-rollout-worker-probe-v1"
SLOTS = [
    ("sb3A", "sb3", 1097),
    ("leanA", "lean", 1097),
    ("sb3B", "sb3", 1098),
    ("leanB", "lean", 1098),
]
CONTROLLER_RESERVE = 128 * 1024**2


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def inner_view(p):
    keys = (
        "whole_wall_s",
        "per_slot_wall_s",
        "control_wall_s",
        "physics_limit",
        "fixture_bytes",
        "receipt_bytes",
        "receipt_reserve_bytes",
        "simulation_workers",
        "threads_per_process",
        "process_ceiling",
        "planned_stop_rss_bytes",
        "owned_rss_bytes",
        "max_monitor_gap_s",
        "close_deadline_s",
        "ipc_bytes",
        "rss_reader",
        "watchdog_output_backend",
        "watchdog_sampling_profile",
        "slots",
    )
    return {"schema": INNER_SCHEMA, **{k: p[k] for k in keys}}


def validate(p):
    expected = {
        "schema": SCHEMA,
        "whole_wall_s": 300,
        "setup_wall_s": 60,
        "control_wall_s": 30,
        "per_slot_wall_s": 30,
        "audit_wall_s": 30,
        "task_wall_s": 270,
        "physics_limit": 0,
        "simulation_workers": 1,
        "threads_per_process": 1,
        "process_ceiling": 12,
        "planned_stop_rss_bytes": 7 * 1024**3,
        "owned_rss_bytes": 8 * 1024**3,
        "fixture_bytes": 16 * 1024**2,
        "receipt_bytes": 1024**2,
        "receipt_reserve_bytes": 768 * 1024,
        "ipc_bytes": 1024**2,
        "max_monitor_gap_s": 0.1,
        "close_deadline_s": 2,
        "controller_reserve_bytes": CONTROLLER_RESERVE,
        "rss_reader": "terminal-mm-reserve-v3",
        "watchdog_output_backend": "private-tmpfs-v1",
        "watchdog_sampling_profile": "lean-probe-1ms-v1",
        "receipt_wire_encoding": "gzip-main-stdout-bootstrap-v2",
        "audit_decoded_bytes": 1024**2,
    }
    if (
        any(p.get(k) != v for k, v in expected.items())
        or [(s["id"], s["profile"], s["seed"]) for s in p.get("slots", [])] != SLOTS
    ):
        raise ValueError("Explicit finite H30 fake-only outer packet required")
    inner = inner_view(p)
    if p.get("inner_packet") != inner or p.get("inner_packet_sha256") != digest(inner):
        raise ValueError("Frozen explicit inner probe packet drift")
    return inner


def bound_equal(a, b):
    return all(a.get(k) == b.get(k) for k in ("pid", "start_ticks", "parent_pid"))


def headroom(p, usage, projection):
    if (
        projection.get("scope")
        != "prospective stored-evidence sizing; zero native work"
        or projection["projected_peak_receipt_bytes"] > p["receipt_bytes"]
        or projection["remaining_before_probes_bytes"] > p["receipt_reserve_bytes"]
        or usage["receipt_bytes"] + p["receipt_reserve_bytes"] > p["receipt_bytes"]
        or usage["fixture_bytes"] + 2 * p["ipc_bytes"] > p["fixture_bytes"]
    ):
        raise ValueError("Prospective whole-study receipt/fixture first limit")
    return {
        "used": usage,
        "reserved_receipts": p["receipt_reserve_bytes"],
        "projection": projection,
    }


def guard_limits(p):
    return {
        "planned_rss": p["planned_stop_rss_bytes"] - p["controller_reserve_bytes"],
        "hard_rss": p["owned_rss_bytes"] - p["controller_reserve_bytes"],
        "process_cap": p["process_ceiling"] - 1,
    }


def check_guard(w, p, *, complete):
    expected = guard_limits(p)
    if any(w["limits"][k] != v for k, v in expected.items()):
        raise ValueError("Guard reserved-controller limits drift")
    if (
        w["limits"]["poll_s"] != 0.001
        or w["final_live"]
        or w["peak_capacity_charge_bytes"] + p["controller_reserve_bytes"]
        > p["owned_rss_bytes"]
    ):
        raise ValueError("Guard capacity/profile/cleanup")
    if complete and (
        w["status"] != "exited"
        or w["pending_retirements"]
        or max(w["max_probe_s"], w["max_sample_gap_s"]) > 0.1
    ):
        raise ValueError("Guard100ms gap/exit/pending gate")
    for rec in w["retirement_records"].values():
        e = rec["evidence"]
        if (
            e["tasks"] != e["after_tasks"]
            or e["tasks"] != [str(rec["pid"])]
            or e["statm_pages"] != [0] * 7
            or not e["vmrss_absent"]
            or not e["vmsize_absent"]
            or e["status_threads"] != 1
            or not all(
                x["start_ticks"] == rec["start_ticks"] and x["flags"] & 4
                for x in (e["before"], e["after"], e["task"])
            )
        ):
            raise ValueError("Raw terminal-mm evidence")
    return {
        "status": w["status"],
        "aggregate_capacity_upper_bound": w["peak_capacity_charge_bytes"]
        + p["controller_reserve_bytes"],
        "max_probe_s": w["max_probe_s"],
        "max_sample_gap_s": w["max_sample_gap_s"],
    }


def positive_retirement(w, worker):
    r = w["retirement_records"].get(str(worker["pid"]) + ":" + worker["start"])
    if (
        not r
        or not r["released"]
        or r["highwater_charge_bytes"] <= 0
        or r["max_hold_s"] > 0.1
    ):
        raise ValueError("Actual positive <=100ms fake-worker retirement required")
    return r


def dispatch(root, p, slot, folder, ready, source, protocol, run_probe):
    inner = validate(p)
    if digest(p) != protocol or slot not in [s[0] for s in SLOTS]:
        raise ValueError("Outer packet/slot binding")
    folder = Path(folder)
    if (
        folder.resolve() != (Path(root) / p["output"] / "fixtures" / slot).resolve()
        or folder.exists()
        or folder.is_symlink()
    ):
        raise ValueError("Fresh owned fake probe folder required")
    binding = {
        "source_commit": source,
        "outer_protocol_sha256": protocol,
        "inner_packet_sha256": digest(inner),
        "inner_packet": inner,
        "slot": slot,
        "native_physics": 0,
    }
    with folder.with_name(slot + "-outer-binding.json").open("x") as stream:
        json.dump(binding, stream, sort_keys=True)
    return run_probe(folder, slot, inner, ready)
