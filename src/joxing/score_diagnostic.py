"""Read-only contact explanations and a scorer independent of HitTracker.update.

Consumes already staged measurements. It cannot validate missing physical inputs
or supply a reward to a controller. Rules and production snapshots are read-only.
"""

import copy
import math
from collections import Counter
from dataclasses import asdict, is_dataclass

from .provenance import digest


def plain(value):
    return asdict(value) if is_dataclass(value) else copy.deepcopy(value)


def new_tracker(episode, glove, rules):
    return {
        "rules": plain(rules),
        "episode_id": episode,
        "glove_id": glove,
        "last_tick": -1,
        "sequence": 0,
        "clear_since": None,
        "current": None,
        "ledger": {},
    }


def explain_interval(interval, tracker_before, tick, rules, *, bell=False):
    """Return independent next snapshot, events and effective rejection evidence.

    Current low approach/speed does not reject an earlier eligible onset. A saved
    false onset latch without the onset sample has unknown *cause*, not a cause
    guessed from the current contact. Impulse eligibility is separate from award.
    """
    row, s, r = plain(interval), copy.deepcopy(tracker_before), plain(rules)
    if type(tick) is not int or tick < 0 or type(bell) is not bool:
        raise ValueError("Malformed diagnostic clock")
    if s["rules"] != r:
        raise ValueError("Diagnostic rules differ from saved tracker")
    details = {"tick": tick, "bell": bell, "samples": [], "duplicate_tick": False}
    if tick == s["last_tick"]:
        details["duplicate_tick"] = True
        return {"after": s, "events": [], "explanation": details}
    if tick < s["last_tick"] or (s["last_tick"] >= 0 and tick != s["last_tick"] + 1):
        raise ValueError("Missing or reversed diagnostic tick")
    if not all(math.isfinite(row[k]) for k in ("extension_m", "clearance_m")):
        raise ValueError("Nonfinite geometric recovery evidence")
    unique = {}
    for t in row["touches"]:
        if (
            not all(
                math.isfinite(t[k])
                for k in (
                    "normal_force_N",
                    "approach_mps",
                    "active_speed_mps",
                    "face_cos",
                )
            )
            or t["normal_force_N"] < 0
        ):
            raise ValueError("Nonfinite or negative staged contact")
        key = t["sample_id"]
        if key in unique and unique[key] != t:
            raise ValueError("Conflicting staged sample identity")
        unique[key] = t
    s["last_tick"] = tick
    positive = [t for t in unique.values() if t["normal_force_N"] > 0]
    if s["current"] is None and positive:
        s["sequence"] += 1
        s["current"] = {
            "sequence": s["sequence"],
            "onset_tick": tick,
            "peak_extension_m": row["extension_m"],
            "last_contact_tick": tick,
            "last_direct_tick": None,
            "direct_started": False,
            "onset_eligible": {},
            "impulses": {},
            "window_impulse_Ns": 0.0,
            "guard_window_impulse_Ns": 0.0,
            "peak_force_N": 0.0,
            "finalized": False,
        }
    e = s["current"]
    details["unique_samples"] = len(unique)
    details["duplicate_samples"] = len(row["touches"]) - len(unique)
    if e is None:
        return {"after": s, "events": [], "explanation": details}
    e["peak_extension_m"] = max(e["peak_extension_m"], row["extension_m"])
    if positive:
        e["last_contact_tick"], s["clear_since"] = tick, None
    elif row["clearance_m"] < r["clearance_m"]:
        s["clear_since"] = None
    elif s["clear_since"] is None:
        s["clear_since"] = tick
    clearance_ticks = 0 if s["clear_since"] is None else tick - s["clear_since"] + 1
    retraction = e["peak_extension_m"] - row["extension_m"]
    recovered = (
        s["clear_since"] is not None
        and clearance_ticks >= round(r["clear_s"] / r["dt_s"])
        and retraction >= r["retraction_m"]
    )
    window_ticks = round(r["impact_window_s"] / r["dt_s"])
    in_window = not bell and tick - e["onset_tick"] < window_ticks
    collecting = in_window and not e["finalized"]
    details.update(
        event_id=f"{s['episode_id']}/{s['glove_id']}/{e['sequence']}",
        onset_tick=e["onset_tick"],
        in_window=in_window,
        already_finalized=e["finalized"],
        clearance_ticks=clearance_ticks,
        retraction_m=retraction,
        recovered=recovered,
    )
    if collecting:
        e["peak_force_N"] = max(
            e["peak_force_N"], sum(t["normal_force_N"] for t in positive)
        )
    for t in unique.values():
        zone = t["zone"]
        predicates = {
            "positive_force": t["normal_force_N"] > 0,
            "direct_zone": zone in ("head", "torso"),
            "approach": t["approach_mps"] >= r["approach_min_mps"],
            "active_speed": t["active_speed_mps"] >= r["active_speed_min_mps"],
            "face_cos": t["face_cos"] >= r["face_cos_min"],
            "striking_face": bool(t["striking_face"]),
        }
        impulse = t["normal_force_N"] * r["dt_s"]
        first_onset = False
        if collecting and predicates["positive_force"]:
            e["window_impulse_Ns"] += impulse
            if zone == "guard":
                e["guard_window_impulse_Ns"] += impulse
            if predicates["direct_zone"]:
                e["direct_started"], e["last_direct_tick"] = True, tick
                if zone not in e["onset_eligible"]:
                    first_onset = True
                    e["onset_eligible"][zone] = all(
                        predicates[k]
                        for k in (
                            "approach",
                            "active_speed",
                            "face_cos",
                            "striking_face",
                        )
                    )
        latch = e["onset_eligible"].get(zone)
        rejects = []
        for key in ("positive_force", "direct_zone"):
            if not predicates[key]:
                rejects.append(key)
        if e["finalized"]:
            rejects.append("already_finalized")
        if not in_window:
            rejects.append("outside_window_or_bell")
        if predicates["direct_zone"]:
            if latch is not True:
                rejects.append(
                    "onset_ineligible" if latch is False else "onset_not_observed"
                )
            for key in ("face_cos", "striking_face"):
                if not predicates[key]:
                    rejects.append(key)
        eligible = not rejects
        if eligible:
            e["impulses"][zone] = e["impulses"].get(zone, 0.0) + impulse
        details["samples"].append(
            {
                "measured": copy.deepcopy(t),
                "instantaneous_predicates": predicates,
                "first_zone_onset": first_onset,
                "latched_onset_eligible": latch,
                "onset_cause_available": first_onset,
                "impulse_Ns": impulse,
                "eligible_direct_impulse_Ns": impulse if eligible else 0.0,
                "effective_rejections": rejects,
            }
        )
    released = e["last_direct_tick"] is not None and tick - e[
        "last_direct_tick"
    ] >= round(r["gap_s"] / r["dt_s"])
    deadline = tick - e["onset_tick"] + 1 >= window_ticks
    events = []
    if not e["finalized"] and (bell or deadline or released or recovered):
        weights = {"head": r["head_points"], "torso": r["torso_points"]}
        ranked = sorted(e["impulses"], key=lambda z: (-e["impulses"][z], weights[z], z))
        zone = ranked[0] if ranked else None
        enough = zone is not None and e["impulses"][zone] >= r["impulse_min_Ns"]
        record = {
            "event_id": details["event_id"],
            "rules_hash": digest(r),
            "onset_tick": e["onset_tick"],
            "finalized_tick": tick,
            "reason": "bell" if bell else "window" if deadline else "release",
            "zone": zone,
            "points": weights[zone] if enough else 0,
            "direct_impulses_Ns": copy.deepcopy(e["impulses"]),
            "guard_window_impulse_Ns": e["guard_window_impulse_Ns"],
            "window_impulse_Ns": e["window_impulse_Ns"],
            "peak_force_N": e["peak_force_N"],
            "onset_eligible": copy.deepcopy(e["onset_eligible"]),
        }
        events = [record]
        s["ledger"][record["event_id"]] = copy.deepcopy(record)
        e["finalized"] = True
        details["award_rejections"] = (
            []
            if enough
            else [
                "no_eligible_direct_zone"
                if zone is None
                else "direct_impulse_below_threshold"
            ]
        )
    details.update(
        direct_released=released,
        deadline=deadline,
        direct_impulses_Ns=copy.deepcopy(e["impulses"]),
    )
    if e["finalized"] and recovered:
        s["current"], s["clear_since"] = None, None
    return {"after": s, "events": events, "explanation": details}


def audit_saved_trace(initial, final, intervals, horizon_ticks):
    """Explain every saved interval and compare events AND complete final state."""
    state = copy.deepcopy(initial)
    if state["ended"] or not 0 < len(intervals) <= 1000:
        raise ValueError("Invalid saved diagnostic extent")
    rows, counts = [], Counter()
    for interval in intervals:
        tick = interval["tick"]
        if tick != state["next_tick"] or set(interval["gloves"]) != set(
            state["trackers"]
        ):
            raise ValueError("Missing global diagnostic interval/glove")
        events = []
        for glove in sorted(state["trackers"]):
            g = interval["gloves"][glove]
            result = explain_interval(g, state["trackers"][glove], tick, state["rules"])
            state["trackers"][glove] = result["after"]
            events += result["events"]
            info = result["explanation"]
            for sample in info["samples"]:
                counts.update(sample["effective_rejections"])
            if info["samples"] or result["events"] or info.get("recovered"):
                rows.append(
                    {"glove": glove, **info, "events": copy.deepcopy(result["events"])}
                )
        state["next_tick"] += 1
        if tick + 1 == horizon_ticks:
            for glove in sorted(state["trackers"]):
                g = {**interval["gloves"][glove], "touches": []}
                result = explain_interval(
                    g, state["trackers"][glove], tick + 1, state["rules"], bell=True
                )
                state["trackers"][glove] = result["after"]
                events += result["events"]
                rows.append(
                    {
                        "glove": glove,
                        **result["explanation"],
                        "events": copy.deepcopy(result["events"]),
                    }
                )
            state["ended"] = True
        if sorted(events, key=lambda e: e["event_id"]) != interval["events"]:
            raise ValueError("Independent events differ from saved ledger")
    if state != final:
        raise ValueError("Independent final referee snapshot differs")
    return {
        "status": "passed",
        "intervals": len(intervals),
        "rejections": dict(counts),
        "explanations": rows,
        "final_sha256": digest(state),
        "new_physics": 0,
    }
