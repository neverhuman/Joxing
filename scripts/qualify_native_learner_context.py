"""Explicit H27 entry point; all old supervisor routes/defaults are retained."""

import argparse
import json

from qualify_deferred_trial import ROOT
from qualify_deferred_trial import main as supervise

from joxing.training_context import PROFILE, SCHEMA


def receipt_headroom(packet, usage):
    projection = json.loads((ROOT / packet["receipt_projection_file"]).read_text())
    reserve = packet["receipt_reserve_bytes"]
    if (
        projection["scope"] != "synthetic stored-evidence sizing; zero native work"
        or projection["projected_peak_receipt_bytes"] > reserve
        or reserve < 1024**2
        or usage["receipt_bytes"] + reserve > packet["receipt_bytes"]
    ):
        raise ValueError("Prospective receipt first limit before native work")
    return {
        "used": usage["receipt_bytes"],
        "reserved": reserve,
        "projection": projection,
    }


def validate(packet, original):
    if packet["schema"] != SCHEMA or packet["execution_profile"] != PROFILE:
        raise ValueError("Named learner execution profile required")
    for key in ("trial_config", "seed", "evaluation_ids", "slots", "balance_config"):
        if packet[key] != original[key]:
            raise ValueError("Original learner fixture changed: " + key)
    expected = {
        "simulation_workers": 4,
        "threads_per_process": 1,
        "process_ceiling": 16,
        "owned_rss_bytes": 8 * 1024**3,
        "planned_stop_rss_bytes": 7 * 1024**3,
        "max_monitor_gap_s": 0.1,
        "close_deadline_s": 2,
        "whole_wall_s": 300,
        "per_slot_wall_s": 60,
        "physics_limit": 240000,
        "receipt_bytes": 2 * 1024**2,
        "fixture_bytes": 1024**3,
        "control_wall_s": 30,
    }
    if any(packet[k] != v for k, v in expected.items()):
        raise ValueError("Registered H27 first-limit envelope changed")
    if (
        packet["rss_reader"] != "terminal-mm-reserve-v3"
        or packet["watchdog_output_backend"] != "private-tmpfs-v1"
    ):
        raise ValueError("Qualified independent watcher required")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("packet", "source-sha", "protocol-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--epoch-start-monotonic", type=float, required=True)
    args = parser.parse_args()
    packet = json.loads((ROOT / args.packet).read_text())
    original = json.loads(
        (
            ROOT / "experiments/v0.0.11/deferred-native-terminal-v3-protocol.json"
        ).read_text()
    )
    validate(packet, original)
    supervise()
