from types import SimpleNamespace

import pytest

from joxing.native_pair_motor_evidence import EvidenceBudget
from joxing.native_pair_motor_pilot import LimitedBuffer, MotorMetrics, write_binary


def test_checkpoint_serialization_and_output_bounds_precede_disk_write(tmp_path):
    buf = LimitedBuffer(16)
    buf.write(b"x" * 16)
    with pytest.raises(ValueError, match="serialization"):
        buf.write(b"x")
    budget = EvidenceBudget(tmp_path, raw_limit=200000, text_limit=100000)
    with pytest.raises(ValueError, match="serialization"):
        write_binary(budget, tmp_path / "model.zip", b"x" * 17, 16)
    assert not list(tmp_path.iterdir())
    write_binary(budget, tmp_path / "model.zip", b"x" * 16, 16)
    with pytest.raises(FileExistsError):
        write_binary(budget, tmp_path / "model.zip", b"x", 16)


def test_motor_cycles_require_true_persistent_order_and_new_guard():
    m = MotorMetrics()
    world = SimpleNamespace(last_metrics={})

    def observe(phase, n, good=True):
        for _ in range(n):
            world.last_metrics = {
                p: {
                    "phase": phase,
                    "hand_target_distance_m": [0.1, 0.1] if good else [0.2, 0.1],
                    "face_cos_positive": 0.7,
                    "tilt_rad": 0.0,
                    "drop_m": 0.0,
                    "up_world_z": 1.0,
                }
                for p in "AB"
            }
            m.observe(world)

    observe(0, 39)
    observe(1, 50)
    observe(2, 50)
    assert m.cycles == {"A": 0, "B": 0}
    observe(0, 40)
    observe(1, 39)
    observe(1, 1, False)
    observe(1, 39)
    observe(2, 50)
    assert m.cycles == {"A": 0, "B": 0}
    observe(0, 40)
    observe(1, 40)
    observe(2, 40)
    assert m.cycles == {"A": 1, "B": 1}
    observe(2, 100)
    assert m.cycles == {"A": 1, "B": 1}
