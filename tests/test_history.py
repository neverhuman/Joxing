import copy
import json

import numpy as np
import pytest

from joxing.history import TimedHistory, TimedSample


def sample(t, received=None, value=None, **kwargs):
    return TimedSample(
        kwargs.get("game_id", "game-1"),
        kwargs.get("player", "A"),
        "features-v1",
        t,
        t if received is None else received,
        np.array([t if value is None else value, -t], np.float32),
    )


def test_history_preserves_delay_order_and_excludes_future_arrivals():
    h = TimedHistory("game-1", "A", "features-v1", 2, capacity=3)
    h.receive(sample(2, received=3), now_tick=3)
    h.receive(sample(1, received=4), now_tick=4)  # Reordered arrival.
    h.receive(sample(4, received=6), now_tick=6)
    view = h.at(4)
    assert view.sampled_ticks.tolist() == [-1, 1, 2]
    assert view.sample_age_ticks.tolist() == [-1, 3, 2]
    assert view.receipt_age_ticks.tolist() == [-1, 0, 1]
    assert view.valid.tolist() == [False, True, True]
    assert not view.values.flags.writeable
    np.testing.assert_array_equal(h.at(6).values, [[1, -1], [2, -2], [4, -4]])
    assert not h.at(0).valid.any()


def test_cross_corner_episode_future_and_conflicting_packets_rejected():
    h = TimedHistory("game-1", "A", "features-v1", 2)
    for packet, now in [
        (sample(0, player="B"), 0),
        (sample(0, game_id="old"), 0),
        (sample(0), 1),
    ]:
        with pytest.raises(ValueError):
            h.receive(packet, now_tick=now)
    with pytest.raises(ValueError):
        sample(3, received=2)
    packet = sample(0)
    h.receive(packet, now_tick=0)
    h.receive(sample(0, received=2), now_tick=2)
    assert h.at(2).receipt_age_ticks[-1] == 2  # Duplicate did not alter history.
    with pytest.raises(ValueError):
        h.receive(sample(0, received=3, value=9), now_tick=3)
    with pytest.raises(ValueError):
        h.at(-1)


def test_bounded_owned_history_restores_exactly_and_rejects_corruption():
    h = TimedHistory("game-1", "A", "features-v1", 2, capacity=2)
    packet = sample(1)
    h.receive(packet, now_tick=1)
    with pytest.raises(ValueError):
        packet.values[0] = 99
    h.receive(sample(3), now_tick=3)
    h.receive(sample(2, received=4), now_tick=4)
    assert h.at(5).sampled_ticks.tolist() == [2, 3]
    state = json.loads(json.dumps(h.snapshot()))
    restored = TimedHistory.restore(
        state, game_id="game-1", player="A", schema_hash="features-v1"
    )
    assert restored.snapshot() == state
    np.testing.assert_array_equal(h.at(5).values, restored.at(5).values)
    for change in ("owner", "future", "nan", "duplicate"):
        bad = copy.deepcopy(state)
        if change == "owner":
            bad["player"] = "B"
        if change == "future":
            bad["samples"][0]["received_tick"] = 0
        if change == "nan":
            bad["samples"][0]["values"][0] = float("nan")
        if change == "duplicate":
            bad["samples"][1] = bad["samples"][0]
        with pytest.raises(ValueError):
            TimedHistory.restore(
                bad, game_id="game-1", player="A", schema_hash="features-v1"
            )
