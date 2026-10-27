import copy
import json

import numpy as np
import pytest

from joxing.observations import DIMENSION, SCHEMA_HASH
from joxing.sensing import SensorPacket, SensorQueue


def packet(t, delay=0, value=None, game="private-seed-123"):
    return SensorPacket(
        game,
        "A",
        SCHEMA_HASH,
        t,
        t + delay,
        np.full(DIMENSION, t if value is None else value, np.float32),
    )


@pytest.mark.parametrize("delay", [0, 1, 2, 20, 40])
def test_delivery_ticks_ages_empty_cold_start_and_owned_features(delay):
    p = packet(0, delay)
    q = SensorQueue(p.game_id, "A")
    q.enqueue(p, now_tick=0)
    for t in range(delay + 1):
        q.pop_due(t)
        frame = q.actor_input(t)
        assert frame.available == (t == delay)
        if t < delay:
            assert (
                frame.latest_observation is None
                and not frame.history.valid.any()
                and frame.sample_age_ticks == -1
            )
    assert frame.sample_age_ticks == delay and frame.receipt_age_ticks == 0
    assert frame.latest_observation.game_id == "actor-visible"
    with pytest.raises(ValueError):
        frame.latest_observation.values[0] = 2
    assert not np.shares_memory(p.values, frame.latest_observation.values)


def test_late_older_sample_enters_history_without_replacing_latest():
    q = SensorQueue("private-seed-123", "A")
    q.enqueue(packet(0, 40), now_tick=0)
    for t in range(41):
        if t == 20:
            q.enqueue(packet(20), now_tick=t)
        if t == 40:
            q.enqueue(packet(40), now_tick=t)
        q.pop_due(t)
    frame = q.actor_input(40)
    assert frame.latest_observation.sampled_tick == 40
    assert frame.history.sampled_ticks[-3:].tolist() == [0, 20, 40]
    assert q.records[0]["disposition"] == "late_history"
    assert q.records[0]["received_tick"] == 40
    assert q.pop_due(40) == []


def test_drop_duplicates_conflicts_capacity_and_missing_clock():
    q = SensorQueue("private-seed-123", "A", capacity=1)
    q.enqueue(packet(0, 40), now_tick=0)
    assert not q.enqueue(packet(0, 40), now_tick=0)
    with pytest.raises(ValueError):
        q.enqueue(packet(0, 40, value=1), now_tick=0)
    with pytest.raises(ValueError):
        q.enqueue(packet(0, 40), now_tick=0, drop=True)
    q.pop_due(0)
    with pytest.raises(TimeoutError):
        q.enqueue(packet(1, 40), now_tick=1)
    q.enqueue(packet(1), now_tick=1, drop=True)
    q.pop_due(1)
    assert q.records[1]["disposition"] == "dropped" and not q.actor_input(1).available
    with pytest.raises(ValueError):
        q.pop_due(3)
    with pytest.raises(ValueError):
        q.enqueue(packet(1, game="other"), now_tick=1)
    with pytest.raises(ValueError):
        q.enqueue(packet(2, 41), now_tick=2)


def test_pending_history_and_first_receipts_roundtrip_with_corruption_rejection():
    q = SensorQueue("private-seed-123", "A")
    q.enqueue(packet(0, 2), now_tick=0)
    q.pop_due(0)
    q.pop_due(1)
    saved = json.loads(json.dumps(q.snapshot()))
    r = SensorQueue.restore(saved, game_id=q.game_id, player="A")
    assert r.snapshot() == q.snapshot()
    assert q.pop_due(2) == r.pop_due(2)
    assert r.snapshot() == q.snapshot()
    for kind in [
        "future",
        "fingerprint",
        "owner",
        "missing",
        "noncanonical",
        "receipt",
    ]:
        bad = copy.deepcopy(saved)
        if kind == "future":
            bad["last_polled_tick"] = 3
        if kind == "fingerprint":
            bad["records"][0]["sha256"] = "0" * 64
        if kind == "owner":
            bad["pending"][0]["player"] = "B"
        if kind == "missing":
            bad["pending"] = []
        if kind == "noncanonical":
            bad["pending"][0]["values"][0] = 1e-30
        if kind == "receipt":
            bad["records"][0]["disposition"] = "delivered"
            bad["records"][0]["received_tick"] = 0
        with pytest.raises(ValueError):
            SensorQueue.restore(bad, game_id=q.game_id, player="A")


def test_long_drop_burst_keeps_receipts_for_old_but_still_visible_history():
    q = SensorQueue("private-seed-123", "A")
    for t in range(200):
        q.enqueue(packet(t), now_tick=t, drop=t > 0)
        q.pop_due(t)
    assert len(q.records) == 128 and 0 in q.records
    assert q.actor_input(199).latest_observation.sampled_tick == 0
    assert (
        SensorQueue.restore(q.snapshot(), game_id=q.game_id, player="A").snapshot()
        == q.snapshot()
    )


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan", "future", "schema"])
def test_packet_rejects_untyped_or_noncausal_source(kind):
    args = {
        "game_id": "g",
        "player": "A",
        "schema_hash": SCHEMA_HASH,
        "sampled_tick": 1,
        "delivery_tick": 1,
        "values": np.zeros(DIMENSION, np.float32),
    }
    if kind == "dtype":
        args["values"] = args["values"].astype(np.float64)
    if kind == "shape":
        args["values"] = args["values"][:-1]
    if kind == "nan":
        args["values"][0] = np.nan
    if kind == "future":
        args["delivery_tick"] = 0
    if kind == "schema":
        args["schema_hash"] = "foreign"
    with pytest.raises(ValueError):
        SensorPacket(**args)
