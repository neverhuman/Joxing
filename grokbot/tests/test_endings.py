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
    classify_contacts,
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
        ((3, 0), [fall("A", 40, "knockdown")], True, (-1, 1), "A knocked down fall"),
        ((0, 0), [fall("A", 9, "knockdown"), fall("B", 9)], True, (0, 0), "draw"),
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
    gid: {"name": body, "body": body, **fields}
    for gid, body, fields in [
        (0, "world", {"actor": None, "role": "floor"}),
        (1, "A_pelvis", {"actor": "A", "role": "native_non_scoring"}),
        (2, "A_calcn_r", {"actor": "A", "role": "native_non_scoring"}),
        (3, "B_thorax", {"actor": "B", "role": "torso"}),
        (4, "B_toes_l", {"actor": "B", "role": "native_non_scoring"}),
        (5, "A_glove_r", {"actor": "A", "role": "glove"}),
        (6, "B_head_neck", {"actor": "B", "role": "head"}),
        (7, "B_back", {"actor": "B", "role": "native_non_scoring"}),
        (8, "B_glove_l", {"actor": "B", "role": "glove"}),
        (9, "A_thorax", {"actor": "A", "role": "torso"}),
    ]
}
FAST, SLOW = 2.0, 0.1


def rows(ticks, *contacts_per_tick):
    """Evidence rows shaped like ``BoxingEnv.evidence``.

    contacts_per_tick(tick) -> [(g1, g2, force[, approach[, active1[, active2]]])].
    Each glove in a contact also gets the referee touch the env would record,
    with its own speed against its own trunk (default: ``approach``).
    """
    out = []
    for t in range(ticks):
        contacts, gloves = [], {}
        for make in contacts_per_tick:
            for g1, g2, f, *speeds in make(t):
                approach = speeds[0] if speeds else 0.0
                active1 = speeds[1] if len(speeds) > 1 else approach
                active = {g1: active1, g2: speeds[2] if len(speeds) > 2 else active1}
                sample = f"{len(contacts)}:{g1}:{g2}"
                contacts.append(
                    {
                        "geom1": g1,
                        "geom2": g2,
                        "normal_force_N": f,
                        "approach_mps": approach,
                    }
                )
                for g in (g1, g2):
                    if REGISTRY[g]["role"] == "glove":
                        touches = gloves.setdefault(
                            REGISTRY[g]["name"], {"touches": []}
                        )
                        touches["touches"].append(
                            {"sample_id": sample, "active_speed_mps": active[g]}
                        )
        out.append({"tick": t, "contacts": contacts, "gloves": gloves})
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
    none = ([], [])
    assert attribute(none, 1000, 1000) == "self"
    for ticks, verdict in [
        ([499], "self"),
        ([500], "x"),
        ([1100], "x"),
        ([1101], "self"),
    ]:
        for contacts, hit in [((ticks, []), "knockdown"), (([], ticks), "unresolved")]:
            expected = hit if verdict == "x" else "self"
            assert attribute(contacts, 1000, 1100, lookback_ticks=500) == expected
    assert attribute(([1000], [1000]), 1000, 1100) == "unresolved"  # blocking wins


B_FALLS_AT = 300


def b_falls(t):
    return [(3, 0, 500.0)] if t >= B_FALLS_AT else []


def verdict_after(*strike):
    evidence = rows(B_FALLS_AT + SUPPORT_PERSIST_TICKS, b_falls, *strike)
    (event,) = support_losses(evidence, REGISTRY)
    return event.attribution


@pytest.mark.parametrize(
    "contact, verdict",
    [
        # A fast glove to the head, then B goes down: a knockdown, B loses.
        (lambda t: [(5, 6, 300.0, FAST)] if 200 <= t < 205 else [], "knockdown"),
        # The same strike listed with the glove second.
        (lambda t: [(6, 5, 300.0, FAST)] if 200 <= t < 205 else [], "knockdown"),
        # Gloves meeting at speed are boxing too.
        (lambda t: [(8, 5, 300.0, FAST)] if 200 <= t < 205 else [], "knockdown"),
        # Gloves meeting: whichever glove actually punched makes it boxing...
        (
            lambda t: [(8, 5, 300.0, FAST, FAST, 0.0)] if 200 <= t < 205 else [],
            "knockdown",
        ),
        (
            lambda t: [(8, 5, 300.0, FAST, 0.0, FAST)] if 200 <= t < 205 else [],
            "knockdown",
        ),
        # ...but two still gloves pressed together are not.
        (
            lambda t: [(8, 5, 300.0, FAST, 0.0, 0.0)] if 200 <= t < 205 else [],
            "unresolved",
        ),
        # B's own fast strike on A's torso while falling buys no block.
        (lambda t: [(8, 9, 300.0, FAST)] if 200 <= t < 205 else [], "knockdown"),
        # The defender walking into a still glove: fast approach, no punch.
        (lambda t: [(5, 6, 300.0, FAST, 0.0)] if 200 <= t < 205 else [], "unresolved"),
        # A slow glove push is not a strike.
        (lambda t: [(5, 6, 300.0, SLOW)] if 200 <= t < 205 else [], "unresolved"),
        # A strike held past the 50 ms impact window becomes holding.
        (lambda t: [(5, 6, 300.0, FAST)] if 200 <= t < 260 else [], "unresolved"),
        # A fast glove to the back is a possible foul, never boxing.
        (lambda t: [(5, 7, 300.0, FAST)] if 200 <= t < 205 else [], "unresolved"),
        # Body on body (clinch, lean, shove).
        (lambda t: [(1, 3, 300.0)] if 200 <= t < 205 else [], "unresolved"),
    ],
)
def test_strike_classification(contact, verdict):
    assert verdict_after(contact) == verdict


def test_glove_contact_without_a_referee_touch_blocks():
    evidence = rows(
        B_FALLS_AT + SUPPORT_PERSIST_TICKS,
        b_falls,
        lambda t: [(5, 6, 300.0, FAST)] if 200 <= t < 205 else [],
    )
    assert support_losses(evidence, REGISTRY)[0].attribution == "knockdown"
    for row in evidence:
        row["gloves"] = {}
    assert support_losses(evidence, REGISTRY)[0].attribution == "unresolved"


def test_a_strike_rearms_only_after_the_gap():
    def two(gap):
        def make(t):
            first = 100 <= t < 140
            second = 140 + gap <= t < 150 + gap
            return (
                [(5, 6, 300.0, FAST if t in (100, 140 + gap) else SLOW)]
                if (first or second)
                else []
            )

        return make

    # Up to 7 empty ticks (the referee's 8 ms gap) continue the first run, so
    # contact beyond its 50 ms window is holding.
    strikes, blocking = classify_contacts(rows(250, two(7)), REGISTRY)
    assert strikes == list(range(100, 140)) + list(range(147, 150))
    assert blocking == list(range(150, 157))
    # 8 empty ticks release it; the next fast contact is a fresh strike.
    strikes, blocking = classify_contacts(rows(250, two(8)), REGISTRY)
    assert strikes == list(range(100, 140)) + list(range(148, 158)) and blocking == []
