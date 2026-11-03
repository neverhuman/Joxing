"""H28 explicit entry: zero-native paired IPC/import proof, no default changes."""

import argparse
import json
import os

from qualify_deferred_trial import ROOT
from qualify_deferred_trial import main as supervise

from joxing.lean_worker_probe import SCHEMA, SLOTS


def validate(packet):
    expected = {
        "schema": SCHEMA,
        "whole_wall_s": 300,
        "per_slot_wall_s": 30,
        "control_wall_s": 30,
        "physics_limit": 0,
        "fixture_bytes": 16 * 1024**2,
        "receipt_bytes": 1024**2,
        "receipt_reserve_bytes": 768 * 1024,
        "simulation_workers": 1,
        "threads_per_process": 1,
        "process_ceiling": 12,
        "planned_stop_rss_bytes": 7 * 1024**3,
        "owned_rss_bytes": 8 * 1024**3,
        "max_monitor_gap_s": 0.1,
        "close_deadline_s": 2,
        "ipc_bytes": 1024**2,
        "rss_reader": "terminal-mm-reserve-v3",
        "watchdog_output_backend": "private-tmpfs-v1",
        "watchdog_sampling_profile": "lean-probe-1ms-v1",
    }
    if (
        any(packet.get(k) != v for k, v in expected.items())
        or [(s["id"], s["profile"], s["seed"]) for s in packet["slots"]] != SLOTS
    ):
        raise ValueError("Registered zero-native H28 profile/caps required")


def receipt_headroom(packet, usage):
    projection = json.loads((ROOT / packet["receipt_projection_file"]).read_text())
    reserve = packet["receipt_reserve_bytes"]
    if (
        projection["scope"] != "synthetic stored-evidence sizing; zero native work"
        or projection["projected_peak_receipt_bytes"] > reserve
        or reserve != 768 * 1024
        or usage["receipt_bytes"] + reserve > packet["receipt_bytes"]
    ):
        raise ValueError("H28 prospective receipt first limit before probe")
    return {
        "used": usage["receipt_bytes"],
        "reserved": reserve,
        "projection": projection,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("packet", "source-sha", "protocol-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--epoch-start-monotonic", type=float, required=True)
    args = parser.parse_args()
    validate(json.loads((ROOT / args.packet).read_text()))
    packet = json.loads((ROOT / args.packet).read_text())
    if os.uname().release != packet["execution_kernel_release"]:
        raise ValueError("Registered execution kernel required")
    supervise()
