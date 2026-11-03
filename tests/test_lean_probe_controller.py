import math

import pytest
from control_lean_worker_probe import allowance, source_command


@pytest.mark.parametrize(
    "epoch,started,now,expected",
    [(10, 20, 25, 25), (10, 290, 300, 10), (10, 20, 20, 30)],
)
def test_audit_clock_keeps_original_epoch(epoch, started, now, expected):
    assert allowance(epoch, started, 30, clock=lambda: now) == expected


@pytest.mark.parametrize(
    "epoch,started,now",
    [(10, 20, 311), (10, 20, 51), (20, 10, 21), (10, 20, 19), (math.nan, 20, 21)],
)
def test_expired_future_or_nonfinite_epoch_rejects(epoch, started, now):
    with pytest.raises((ValueError, TimeoutError)):
        allowance(epoch, started, 30, clock=lambda: now)


def test_child_source_command_preserves_hashes_and_original_clock():
    cmd = source_command("packet.json", "sourceSHA", "protocolSHA", 1.25)
    assert cmd[1:] == [
        "scripts/qualify_lean_worker_probe.py",
        "--packet",
        "packet.json",
        "--source-sha",
        "sourceSHA",
        "--protocol-sha256",
        "protocolSHA",
        "--epoch-start-monotonic",
        "1.25",
    ]
