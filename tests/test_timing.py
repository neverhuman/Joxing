import copy
import json

import numpy as np
import pytest

from joxing.timing import ActionQueue, CommandPacket


def packet(decision=0, apply=0, value=0.2, **kwargs):
    return CommandPacket(
        kwargs.get("game_id", "g"),
        kwargs.get("player", "A"),
        decision,
        apply,
        np.full(700, value, np.float32),
        kwargs.get("bundle_hash", "bundle"),
    )


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_zero_one_two_tick_delay_and_ownership(delay):
    q = ActionQueue("g", "A", "bundle")
    p = packet(apply=delay)
    assert q.enqueue(p, now_tick=0)
    assert not p.command.flags.writeable
    for t in range(delay):
        assert q.pop_due(t) is None
    assert q.pop_due(delay).hash == p.hash
    assert q.pop_due(delay) is None
    assert not q.enqueue(p, now_tick=delay)
    assert q.pop_due(delay + 1) is None  # Caller holds the previous applied command.
    assert q.last_applied.hash == p.hash


def test_delayed_old_decision_never_overwrites_newer_and_conflicts_reject():
    q = ActionQueue("g", "A", "bundle")
    q.enqueue(packet(0, 2), now_tick=0)
    assert q.pop_due(0) is None
    q.enqueue(packet(1, 1, 0.7), now_tick=1)
    assert q.pop_due(1).decision_tick == 1
    assert q.pop_due(2) is None
    assert q.last_delivery["discarded_decisions"] == [0]
    with pytest.raises(ValueError):
        q.enqueue(packet(1, 1, 0.8), now_tick=2)
    for foreign in [
        packet(3, 3, player="B"),
        packet(3, 3, game_id="old"),
        packet(3, 3, bundle_hash="other"),
    ]:
        with pytest.raises(ValueError):
            q.enqueue(foreign, now_tick=3)


def test_snapshot_pending_commands_and_reordered_arrivals_restore_exactly():
    q = ActionQueue("g", "A", "bundle")
    q.pop_due(0)
    q.enqueue(packet(1, 2), now_tick=1)
    q.enqueue(packet(0, 2, 0.4), now_tick=1)
    q.pop_due(1)
    saved = json.loads(json.dumps(q.snapshot()))
    other = ActionQueue.restore(saved, game_id="g", player="A", bundle_hash="bundle")
    assert q.pop_due(2).hash == other.pop_due(2).hash
    assert q.snapshot() == other.snapshot()
    for kind in ["player", "nan", "clock", "duplicate"]:
        bad = copy.deepcopy(saved)
        if kind == "player":
            bad["player"] = "B"
        if kind == "nan":
            bad["pending"][0]["command"][0] = float("nan")
        if kind == "clock":
            bad["last_polled_tick"] = 3
        if kind == "duplicate":
            bad["pending"].append(bad["pending"][0])
        with pytest.raises(ValueError):
            ActionQueue.restore(bad, game_id="g", player="A", bundle_hash="bundle")


def test_missing_ticks_capacity_future_and_invalid_commands_stop_before_mutation():
    q = ActionQueue("g", "A", "bundle", capacity=1, max_delay_ticks=2)
    for p, now in [(packet(1, 1), 0), (packet(0, 3), 0)]:
        with pytest.raises(ValueError):
            q.enqueue(p, now_tick=now)
    assert q.snapshot()["pending"] == []
    q.enqueue(packet(0, 2), now_tick=0)
    q.pop_due(0)
    with pytest.raises(TimeoutError):
        q.enqueue(packet(1, 2), now_tick=1)
    with pytest.raises(ValueError):
        q.pop_due(2)
    with pytest.raises(ValueError):
        packet(value=float("nan"))
    with pytest.raises(ValueError):
        packet(value=1.1)
    # Bounded retired fingerprint storage; no per-round unbounded command tape.
    q = ActionQueue("g", "A", "bundle", capacity=1)
    for t in range(20):
        q.enqueue(packet(t, t), now_tick=t)
        q.pop_due(t)
    assert len(q.seen) == 2
