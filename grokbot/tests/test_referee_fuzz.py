"""Property-based fuzz of the candidate one-punch-one-score referee (cycle 0005).

Seeded random per-tick contact streams drive one ``HitTracker``. Each stream mixes
bursts of multi-zone contact (including zero-force and off-face samples), clear
and near-miss gaps, a drifting glove extension, and sometimes a final bell. The
emitted ledger is then checked against the rules as GAME.md states them, computed
independently from the raw stream:

* every encounter is recorded once, with contiguous sequence numbers;
* a new encounter opens only at the first force contact after the previous one
  recovered: 30 clear ticks at >= 2 cm clearance plus >= 3 cm retraction from
  the encounter's peak extension;
* window, guard and per-zone direct impulses equal the in-window integral, with
  each zone's eligibility latched at its first force contact;
* awards go only to the dominant eligible head/torso zone at threshold;
* cutting the stream, JSON round-tripping a snapshot and resuming changes
  nothing, and re-delivering a tick is a no-op.
"""

import json
from dataclasses import astuple

import numpy as np
import pytest

from joxing.rules import HitTracker, Rules, Touch

RULES = Rules()
WINDOW = round(RULES.impact_window_s / RULES.dt_s)
CLEAR = round(RULES.clear_s / RULES.dt_s)
GAP = round(RULES.gap_s / RULES.dt_s)
POINTS = {"head": RULES.head_points, "torso": RULES.torso_points}
ZONES = ("head", "torso", "guard", "native_non_scoring")
STREAMS = 300


def make_stream(seed, ticks=500):
    rng = np.random.default_rng(seed)
    # Odd seeds mirror each head touch onto the torso so impulses tie exactly.
    twins = seed % 2 == 1
    rows, extension, t = [], 0.3, 0
    while t < ticks:
        phase = rng.choice(["contact", "clear", "near"], p=[0.35, 0.45, 0.2])
        for _ in range(int(rng.integers(1, 60))):
            if t >= ticks:
                break
            drift = 0.004 if phase == "contact" else -0.003
            extension = float(
                np.clip(extension + drift + rng.normal(0, 0.01), 0.05, 0.7)
            )
            touches, clearance = [], 0.0
            if phase == "contact":
                for s in range(int(rng.integers(1, 4))):
                    zone = str(rng.choice(ZONES))
                    touches.append(
                        Touch(
                            f"{t}:{s}",
                            "head" if twins and zone == "torso" else zone,
                            0.0 if rng.random() < 0.1 else float(rng.uniform(0, 400)),
                            float(rng.uniform(0, 3)),
                            float(rng.uniform(0, 3)),
                            float(rng.uniform(-1, 1)),
                            bool(rng.random() < 0.8),
                        )
                    )
                    if twins and touches[-1].zone == "head":
                        h = touches[-1]
                        touches.append(
                            Touch(h.sample_id + "t", "torso", *astuple(h)[2:])
                        )
            elif phase == "clear":
                clearance = float(rng.uniform(RULES.clearance_m, 0.2))
            else:
                clearance = float(rng.uniform(0.0, RULES.clearance_m * 0.999))
            rows.append((t, touches, clearance, extension))
            t += 1
    bell = bool(rng.random() < 0.5)
    return rows, bell


def drive(rows, bell, tracker=None, start=0):
    tracker = tracker or HitTracker("fuzz", "A_glove_r", RULES)
    emitted = []
    for t, touches, clearance, extension in rows[start:]:
        emitted += tracker.update(
            t,
            touches,
            clearance_m=clearance,
            extension_m=extension,
            bell=bell and t == rows[-1][0],
        )
    return tracker, emitted


def forced(touches):
    return [x for x in touches if x.normal_force_N > 0]


def recovery_tick(rows, onset):
    """First tick ending a 30-tick clear streak with 3 cm retraction from peak."""
    streak, peak = 0, -np.inf
    for t, touches, clearance, extension in rows[onset:]:
        peak = max(peak, extension)
        streak = 0 if forced(touches) or clearance < RULES.clearance_m else streak + 1
        if streak >= CLEAR and peak - extension >= RULES.retraction_m:
            return t
    return None


def finalization_tick(rows, onset, bell_tick, recovered):
    """Earliest of bell, 50 ms deadline, 8 ms direct-contact gap, or recovery."""
    last_direct = None
    for t, touches, _, _ in rows[onset:]:
        if t != bell_tick and t - onset < WINDOW:
            if any(x.zone in POINTS for x in forced(touches)):
                last_direct = t
        released = last_direct is not None and t - last_direct >= GAP
        if t == bell_tick or t - onset + 1 >= WINDOW or released or t == recovered:
            return (
                t,
                "bell"
                if t == bell_tick
                else "window"
                if t - onset + 1 >= WINDOW
                else "release",
            )
    return None, None


def window_accounting(rows, onset, final, bell_tick):
    total = guard = 0.0
    impulses, eligible = {}, {}
    for t, touches, _, _ in rows[onset : min(onset + WINDOW - 1, final) + 1]:
        if t == bell_tick:
            break
        for x in forced(touches):
            impulse = x.normal_force_N * RULES.dt_s
            total += impulse
            guard += impulse if x.zone == "guard" else 0.0
            if x.zone not in POINTS:
                continue
            eligible.setdefault(
                x.zone,
                x.approach_mps >= RULES.approach_min_mps
                and x.active_speed_mps >= RULES.active_speed_min_mps
                and x.face_cos >= RULES.face_cos_min
                and x.striking_face,
            )
            if (
                eligible[x.zone]
                and x.striking_face
                and x.face_cos >= RULES.face_cos_min
            ):
                impulses[x.zone] = impulses.get(x.zone, 0.0) + impulse
    return total, guard, impulses, eligible


@pytest.mark.parametrize("seed", range(STREAMS))
def test_referee_ledger_matches_stated_rules(seed):
    rows, bell = make_stream(seed)
    tracker, emitted = drive(rows, bell)
    assert sorted(e["event_id"] for e in emitted) == sorted(tracker.ledger)
    records = sorted(emitted, key=lambda e: int(e["event_id"].rsplit("/", 1)[1]))
    sequences = [int(e["event_id"].rsplit("/", 1)[1]) for e in records]
    assert sequences == list(range(1, len(records) + 1))
    bell_tick = rows[-1][0] if bell else None

    contact_ticks = [t for t, touches, _, _ in rows if forced(touches)]
    expected_onset = contact_ticks[0] if contact_ticks else None
    for k, record in enumerate(records):
        onset, final = record["onset_tick"], record["finalized_tick"]
        assert onset == expected_onset, (seed, k)
        recovered = recovery_tick(rows, onset)
        assert (final, record["reason"]) == finalization_tick(
            rows, onset, bell_tick, recovered
        )
        total, guard, impulses, eligible = window_accounting(
            rows, onset, final, bell_tick
        )
        assert record["window_impulse_Ns"] == pytest.approx(total, abs=1e-12)
        assert record["guard_window_impulse_Ns"] == pytest.approx(guard, abs=1e-12)
        assert record["onset_eligible"] == eligible
        assert record["direct_impulses_Ns"].keys() == impulses.keys()
        for zone, value in impulses.items():
            assert record["direct_impulses_Ns"][zone] == pytest.approx(value, abs=1e-12)
        if impulses:
            best = max(impulses.values())
            assert impulses[record["zone"]] == best
            tied = [z for z, v in impulses.items() if v == best]
            assert POINTS[record["zone"]] == min(POINTS[z] for z in tied)
            reached = (
                record["direct_impulses_Ns"][record["zone"]] >= RULES.impulse_min_Ns
            )
            assert record["points"] == (POINTS[record["zone"]] if reached else 0)
        else:
            assert record["zone"] is None and record["points"] == 0
        later = [t for t in contact_ticks if recovered is not None and t > recovered]
        expected_onset = later[0] if later else None
    # A bell finalizes everything; otherwise only the open encounter is unrecorded.
    if expected_onset is not None:
        assert not bell and not tracker.current.finalized
        assert tracker.current.onset_tick == expected_onset


@pytest.mark.parametrize("seed", range(0, STREAMS, 10))
def test_cut_restore_and_redelivery_change_nothing(seed):
    rows, bell = make_stream(seed)
    _, whole = drive(rows, bell)
    cut = int(np.random.default_rng(seed + 10_000).integers(1, len(rows) - 1))
    first, before = drive(rows[:cut], False)
    t, touches, clearance, extension = rows[cut - 1]
    assert first.update(t, touches, clearance_m=clearance, extension_m=extension) == []
    resumed = HitTracker.restore(json.loads(json.dumps(first.snapshot())))
    _, after = drive(rows, bell, resumed, start=cut)
    assert before + after == whole


@pytest.mark.xfail(
    strict=True,
    reason="cycle-0005 F1: HitTracker.restore aliases the snapshot's ledger and "
    "encounter dicts; callers must deep-copy (all current call sites do)",
)
def test_restore_does_not_alias_the_snapshot():
    rows, bell = make_stream(3)
    tracker, _ = drive(rows[:100], False)
    snapshot = tracker.snapshot()
    frozen = json.dumps(snapshot, sort_keys=True)
    drive(rows, bell, HitTracker.restore(snapshot), start=100)
    assert json.dumps(snapshot, sort_keys=True) == frozen
