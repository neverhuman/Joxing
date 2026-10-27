"""Reference ending table: every termination type has a defined utility (or an
explicit no-result), shaping still telescopes to it, and falls are measured from
persistent non-foot floor support rather than a height shortcut."""

import pytest

from grokbot.endings import (
    ATTRIBUTION_LOOKBACK_TICKS,
    SUPPORT_FORCE_N,
    SUPPORT_PERSIST_TICKS,
    LossEvent,
    attribute,
    decide,
    support_losses,
)
from joxing.rules import match_reward


def fall(player, tick, attribution="self", kind="fall"):
    return LossEvent(player, tick, kind, attribution)


@pytest.mark.parametrize(
    "points, events, bell, utility, reason",
    [
        ((3, 1), [], True, (1, -1), "bell: points"),
        ((2, 2), [], True, (0, 0), "bell: points"),
        ((5, 0), [fall("A", 100)], True, (-1, 1), "A self-caused fall"),
        ((0, 5), [fall("B", 9, kind="exit")], True, (1, -1), "B self-caused exit"),
        ((1, 0), [fall("A", 50), fall("B", 50)], True, (0, 0), "draw: simultaneous"),
        ((0, 0), [fall("B", 200), fall("A", 100)], True, (-1, 1), "A self-caused"),
        ((4, 0), [fall("B", 10, "opponent")], True, (1, -1), "bell: points"),
        ((0, 0), [fall("B", 10, "opponent"), fall("A", 20)], True, (-1, 1), "A self"),
        ((9, 0), [fall("A", 30, "unresolved")], True, None, "blocked"),
        ((0, 0), [fall("A", 30, "unresolved"), fall("B", 30)], True, None, "blocked"),
        ((3, 0), [], False, None, "no sporting end"),
        ((0, 0), [fall("A", 30)], False, (-1, 1), "A self-caused fall"),
    ],
)
def test_ending_table(points, events, bell, utility, reason):
    ending = decide(points, events, bell=bell, bell_tick=999)
    assert ending.utility == utility
    assert ending.reason.startswith(reason)
    if utility is not None:
        assert sum(utility) == 0


def test_invalid_is_never_a_result():
    ending = decide((5, 0), [fall("B", 1)], bell=True, invalid_reason="NaN act")
    assert ending.utility is None and ending.reason == "invalid: NaN act"


def test_malformed_inputs_are_rejected():
    with pytest.raises(ValueError):
        LossEvent("C", 1, "fall", "self")
    with pytest.raises(ValueError):
        LossEvent("A", -1, "fall", "self")
    with pytest.raises(ValueError):
        LossEvent("A", 1, "slip", "self")
    with pytest.raises(ValueError):
        LossEvent("A", 1, "fall", "maybe")
    with pytest.raises(ValueError):
        decide((1, -1), [], bell=True)


@pytest.mark.parametrize(
    "events, bell",
    [([], True), ([fall("A", 3)], True), ([fall("A", 3), fall("B", 3)], False)],
)
def test_shaped_return_telescopes_to_the_ending(events, bell):
    """Bout ends at the ending tick; potential shaping must not change its value."""
    diffs = [0, 1, 2, 1, 3]  # A minus B after each decision
    ending = decide((3 + diffs[-1], 3), events, bell=bell, bell_tick=4)
    total_a = total_b = 0.0
    for step, (old, new) in enumerate(zip(diffs, diffs[1:]), 1):
        terminal = step == len(diffs) - 1
        a, b = match_reward(
            old,
            new,
            terminal=terminal,
            outcome=ending.utility[0] if terminal else 0,
        )
        total_a, total_b = total_a + a, total_b + b
    assert total_a == pytest.approx(ending.utility[0])
    assert total_b == pytest.approx(ending.utility[1])


REGISTRY = {
    0: {"actor": None, "role": "floor", "body": "world"},
    1: {"actor": "A", "role": "native_non_scoring", "body": "A_pelvis"},
    2: {"actor": "A", "role": "native_non_scoring", "body": "A_calcn_r"},
    3: {"actor": "B", "role": "torso", "body": "B_thorax"},
    4: {"actor": "B", "role": "native_non_scoring", "body": "B_toes_l"},
}


def rows(ticks, *contacts_per_tick):
    """Evidence rows: contacts_per_tick(tick) -> [(geom1, geom2, force), ...]."""
    out = []
    for t in range(ticks):
        contacts = []
        for make in contacts_per_tick:
            contacts += [
                {"geom1": g1, "geom2": g2, "normal_force_N": f} for g1, g2, f in make(t)
            ]
        out.append({"tick": t, "contacts": contacts})
    return out


def test_feet_carry_any_load_without_a_fall():
    feet = rows(400, lambda t: [(0, 2, 600.0), (4, 0, 600.0)])
    assert support_losses(feet, REGISTRY) == []


def test_persistent_non_foot_support_is_a_self_caused_fall():
    start = 50
    load = rows(
        start + SUPPORT_PERSIST_TICKS,
        lambda t: [(0, 1, SUPPORT_FORCE_N)] if t >= start else [],
    )
    assert support_losses(load, REGISTRY) == [fall("A", start)]


def test_brushes_light_touches_and_short_support_are_not_falls():
    light = rows(400, lambda t: [(1, 0, SUPPORT_FORCE_N * 0.99)])
    short = rows(400, lambda t: [(0, 3, 500.0)] if t % 100 < 99 else [])
    assert support_losses(light, REGISTRY) == []
    assert support_losses(short, REGISTRY) == []


def test_opponent_contact_before_a_fall_leaves_it_unresolved():
    start = ATTRIBUTION_LOOKBACK_TICKS + 100

    def support(t):
        return [(3, 0, 500.0)] if t >= start else []

    for touch_tick, verdict in [
        (start - ATTRIBUTION_LOOKBACK_TICKS - 1, "self"),
        (start - ATTRIBUTION_LOOKBACK_TICKS, "unresolved"),
        (start + 10, "unresolved"),  # touched during the fall itself
    ]:
        evidence = rows(
            start + SUPPORT_PERSIST_TICKS,
            support,
            lambda t, k=touch_tick: [(1, 3, 5.0)] if t == k else [],
        )
        assert support_losses(evidence, REGISTRY) == [fall("B", start, verdict)]


def test_attribute_window_bounds():
    assert attribute([], 1000, 1000) == "self"
    assert attribute([499], 1000, 1100, lookback_ticks=500) == "self"
    assert attribute([500], 1000, 1100, lookback_ticks=500) == "unresolved"
    assert attribute([1100], 1000, 1100, lookback_ticks=500) == "unresolved"
    assert attribute([1101], 1000, 1100, lookback_ticks=500) == "self"
