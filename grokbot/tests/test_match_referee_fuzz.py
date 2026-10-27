"""Four-glove fuzz of the candidate MatchReferee (cycle 0012).

Each glove gets its own seeded random contact stream (the generator from
``test_referee_fuzz``). The match referee must agree exactly with four
independent HitTrackers, mirror its result when the A and B streams swap,
finalize everything at the bell, ignore delivery order, and resume
identically from a JSON snapshot.
"""

import json

import pytest

from grokbot.tests.test_referee_fuzz import make_stream
from joxing.referee import GloveInterval, MatchReferee
from joxing.rules import HitTracker
from joxing.scene import build_scene

MATCHES = 30
TICKS = 400


@pytest.fixture(scope="module")
def scene():
    return build_scene()


@pytest.fixture(scope="module")
def gloves(scene):
    names = sorted(MatchReferee(scene, "names").gloves)
    assert [n[0] for n in names] == ["A", "A", "B", "B"]
    return names


def streams(seed, names):
    return {name: make_stream(seed * 4 + k, TICKS)[0] for k, name in enumerate(names)}


def intervals(rows, t, order=None):
    names = order or list(rows)
    return {
        g: GloveInterval(tuple(rows[g][t][1]), rows[g][t][2], rows[g][t][3])
        for g in names
    }


def drive(scene, rows, *, bell, game_id="fuzz", start=0, referee=None, order=None):
    referee = referee or MatchReferee(scene, game_id)
    emitted = []
    for t in range(start, TICKS):
        emitted += referee.commit(
            intervals(rows, t, order), t, bell=bell and t == TICKS - 1
        )
    return referee, emitted


@pytest.mark.parametrize("seed", range(MATCHES))
def test_match_referee_equals_four_independent_trackers(scene, gloves, seed):
    rows, bell = streams(seed, gloves), seed % 2 == 0
    referee, emitted = drive(scene, rows, bell=bell)
    totals = {"A": 0, "B": 0}
    for glove in gloves:
        tracker = HitTracker("fuzz", glove, referee.rules)
        for t, touches, clearance, extension in rows[glove]:
            tracker.update(t, touches, clearance_m=clearance, extension_m=extension)
        if bell:
            _, _, clearance, extension = rows[glove][-1]
            tracker.update(
                TICKS, [], clearance_m=clearance, extension_m=extension, bell=True
            )
        assert referee.trackers[glove].ledger == tracker.ledger
        totals[glove[0]] += sum(e["points"] for e in tracker.ledger.values())
    assert referee.points == (totals["A"], totals["B"])
    assert sorted(e["event_id"] for e in emitted) == [
        e["event_id"] for e in referee.events
    ]


@pytest.mark.parametrize("seed", range(0, MATCHES, 3))
def test_swapping_corners_mirrors_points_and_events(scene, gloves, seed):
    rows = streams(seed, gloves)
    swapped = {("B" if g[0] == "A" else "A") + g[1:]: rows[g] for g in gloves}
    direct, _ = drive(scene, rows, bell=True)
    mirror, _ = drive(scene, swapped, bell=True)
    assert mirror.points == direct.points[::-1]
    for glove in gloves:
        twin = ("B" if glove[0] == "A" else "A") + glove[1:]
        assert without_ids(mirror.trackers[twin]) == without_ids(direct.trackers[glove])


def without_ids(tracker):
    return [
        {k: v for k, v in e.items() if k != "event_id"} for e in tracker.ledger.values()
    ]


@pytest.mark.parametrize("seed", range(0, MATCHES, 3))
def test_bell_finalizes_every_glove_and_ends_the_match(scene, gloves, seed):
    referee, _ = drive(scene, streams(seed, gloves), bell=True)
    assert referee.ended and referee.next_tick == TICKS
    for tracker in referee.trackers.values():
        assert tracker.current is None or tracker.current.finalized
    with pytest.raises(RuntimeError):
        referee.commit(intervals(streams(seed, gloves), 0), TICKS)


@pytest.mark.parametrize("seed", range(1, MATCHES, 3))
def test_delivery_order_and_mid_match_restore_change_nothing(scene, gloves, seed):
    rows = streams(seed, gloves)
    whole, emitted = drive(scene, rows, bell=True)
    _, reversed_order = drive(scene, rows, bell=True, order=gloves[::-1])
    assert reversed_order == emitted
    cut = 137 + seed
    first, before = drive_until(scene, rows, cut)
    resumed = MatchReferee.restore(scene, json.loads(json.dumps(first.snapshot())))
    _, after = drive(scene, rows, bell=True, start=cut, referee=resumed)
    assert before + after == emitted
    assert resumed.snapshot() == whole.snapshot()


def drive_until(scene, rows, cut):
    referee = MatchReferee(scene, "fuzz")
    emitted = []
    for t in range(cut):
        emitted += referee.commit(intervals(rows, t), t)
    return referee, emitted
