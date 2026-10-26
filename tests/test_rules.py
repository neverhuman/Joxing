import json
from itertools import pairwise

import numpy as np
import pytest

from joxing.rules import HitTracker, Touch, match_reward


def touch(zone="head", force=100.0, speed=2.0, active=2.0, sample="c"):
    return Touch(sample, zone, force, speed, active, 1.0)


def tick(tracker, k, contacts=(), clear=0.0, extension=0.4, bell=False):
    return tracker.update(
        k, list(contacts), clearance_m=clear, extension_m=extension, bell=bell
    )


@pytest.mark.parametrize(
    "zone,expected", [("head", 2), ("torso", 1), ("guard", 0), ("illegal", 0)]
)
def test_contact_window_scores_once_despite_held_pressure(zone, expected):
    t = HitTracker("e", "A/right")
    records = []
    for k in range(300):
        records += tick(t, k, [touch(zone)])
    assert len(records) == 1
    assert records[0]["points"] == expected
    assert records[0]["window_impulse_Ns"] == pytest.approx(5.0)


def test_guard_brush_then_direct_target_uses_target_onset():
    t = HitTracker("e", "g")
    for k in range(10):
        tick(t, k, [touch("guard")] if k == 0 else [])
    for k in range(10, 14):
        tick(t, k, [touch("torso")])
    records = []
    for k in range(14, 50):
        records += tick(t, k)
    assert len(records) == 1 and records[0]["points"] == 1
    assert records[0]["guard_window_impulse_Ns"] == pytest.approx(0.1)


def test_target_cannot_inherit_guard_speed_or_impulse():
    t = HitTracker("e", "g")
    tick(t, 0, [touch("guard", force=10000.0)])
    for k in range(1, 50):
        tick(t, k, [touch("head", speed=0.0)])
    assert next(iter(t.ledger.values()))["points"] == 0


def test_onset_is_latched_before_deceleration():
    t = HitTracker("e", "g")
    tick(t, 0, [touch(force=1.0)])
    for k in range(1, 50):
        tick(t, k, [touch(speed=0.0)])
    assert next(iter(t.ledger.values()))["points"] == 2


def test_stationary_guard_rejects_opponent_running_into_it():
    t = HitTracker("e", "g")
    for k in range(50):
        tick(t, k, [touch(active=0.0)])
    assert next(iter(t.ledger.values()))["points"] == 0


def test_manifold_sum_duplicate_delivery_and_dominant_zone_tie():
    t = HitTracker("e", "g")
    for k in range(4):
        tick(
            t,
            k,
            [
                touch("head", force=50, sample="h1"),
                touch("head", force=50, sample="h2"),
                touch("torso", force=100, sample="t"),
                touch("torso", force=100, sample="t"),
            ],
        )
        assert tick(t, k, [touch()]) == []
    records = []
    for k in range(4, 50):
        records += tick(t, k)
    assert records[0]["zone"] == "torso" and records[0]["points"] == 1
    assert records[0]["peak_force_N"] == 200


def test_subthreshold_zones_cannot_pool_impulse():
    t = HitTracker("e", "g")
    tick(
        t,
        0,
        [touch("head", force=200, sample="h"), touch("torso", force=200, sample="t")],
    )
    for k in range(1, 50):
        tick(t, k)
    assert next(iter(t.ledger.values()))["points"] == 0


def test_clearance_and_retraction_required_after_contact():
    t = HitTracker("e", "g")
    for k in range(50):
        tick(t, k, [touch()])
    for k in range(50, 100):
        tick(t, k, clear=0.03, extension=0.4)
    assert t.current is not None
    for k in range(100, 131):
        tick(t, k, clear=0.03, extension=0.35)
    assert t.current is None
    for k in range(131, 181):
        tick(t, k, [touch()])
    assert len(t.ledger) == 2


def test_gap_and_recontact_cannot_restart_window():
    t = HitTracker("e", "g")
    for k in range(100):
        tick(t, k, [touch()] if k % 7 else [])
    assert len(t.ledger) == 1


def test_bell_cannot_collect_new_impulse():
    t = HitTracker("e", "g")
    tick(t, 0, [touch(force=200.0)])
    record = tick(t, 1, [touch(force=1000.0)], bell=True)[0]
    assert record["points"] == 0 and record["direct_impulses_Ns"][
        "head"
    ] == pytest.approx(0.2)


def test_restore_mid_hit_reproduces_ledger():
    t = HitTracker("e", "g")
    for k in range(3):
        tick(t, k, [touch()])
    copy = HitTracker.restore(json.loads(json.dumps(t.snapshot())))
    for k in range(3, 70):
        contacts = [touch()] if k < 10 else []
        assert tick(t, k, contacts) == tick(copy, k, contacts)
    assert t.snapshot() == copy.snapshot()


def test_shaping_telescopes_to_terminal_match_utility():
    diffs = [0, 2, -1, 3, 1]
    rewards = [
        match_reward(a, b, terminal=i == 3, outcome=1 if i == 3 else 0)
        for i, (a, b) in enumerate(pairwise(diffs))
    ]
    assert np.sum(rewards, axis=0) == pytest.approx([1.0, -1.0])


def test_invalid_and_missing_samples_fail_loudly():
    with pytest.raises(ValueError):
        touch(force=float("nan"))
    t = HitTracker("e", "g")
    tick(t, 0)
    with pytest.raises(ValueError):
        tick(t, 2)
