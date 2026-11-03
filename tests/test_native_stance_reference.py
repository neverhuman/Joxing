import copy

import numpy as np
import pytest

# Reuse the qualified native constructor and synthetic teacher; no real weights.
import test_native_pair_motor_rl as native_fixtures

from joxing.native_stance_reference import (
    PROFILE,
    WalkToHoldReferenceDriver,
    reference_contract,
    stopped_phase,
)
from joxing.provenance import digest
from joxing.shared_motor_task import NativeReferenceDriver


@pytest.fixture(scope="module")
def native_factory():
    yield from native_fixtures.factory.__wrapped__()


def test_phase_seconds_endpoints_monotone_future_order_and_hold():
    times = np.r_[0, 0.02, np.linspace(0.02, 0.4, 401), 0.42, 3, 1e300]
    phase = stopped_phase(times)
    assert phase[0] == 0 and phase[-1] == phase[-2] == phase[-3] == 0.2
    assert float(stopped_phase(0.02)) == 0.0195
    assert np.all(np.diff(phase) >= 0) and np.max(phase) == 0.2
    assert np.all(np.diff(stopped_phase(0.18 + np.arange(6) * 0.02)) > 0)
    assert np.all(stopped_phase(0.4 + np.arange(6) * 0.02) == 0.2)


@pytest.mark.parametrize("bad", [-0.001, np.nan, np.inf, -np.inf])
def test_phase_rejects_invalid_seconds(bad):
    with pytest.raises(ValueError):
        stopped_phase(bad)


def test_stopped_reference_preserves_actual_reset_and_native_layout(native_factory):
    donor = native_factory[1]
    walking = NativeReferenceDriver(donor, 71000)
    stopped = WalkToHoldReferenceDriver(donor, 71000)
    assert donor.physics_total == 0
    for field in walking.initial:
        assert walking.initial[field].tobytes() == stopped.initial[field].tobytes()
    assert np.any(stopped.initial["qvel"] != 0)  # Actual walking speed NOT zeroed.
    assert walking.reference.qpos.tobytes() == stopped.reference.qpos.tobytes()
    assert (
        walking.reference.future_qpos.tobytes()
        != stopped.reference.future_qpos.tobytes()
    )
    assert stopped.reference.future_qpos.shape == (5, 85)
    assert stopped.reference.keypoints_relative_pelvis.shape == (6, 3)
    stopped.sample(400, 0.4)
    held = copy.deepcopy(stopped.snapshot())
    stopped.sample(3000, 3.0)
    for field in ("qpos", "future_qpos", "keypoints_relative_pelvis"):
        assert digest(held[field]) == digest(stopped.snapshot()[field])
    assert stopped.snapshot()["reference_profile"]["profile"] == PROFILE
    assert "reference_profile" not in walking.snapshot()


def test_profile_queries_native_reset_and_next_control_lead(
    native_factory, monkeypatch
):
    donor = native_factory[1]
    calls = []
    original = donor.env.trajectory.query_batch

    def query(times, index):
        calls.append(np.array(times, copy=True))
        return original(times, index)

    monkeypatch.setattr(donor.env.trajectory, "query_batch", query)
    driver = WalkToHoldReferenceDriver(donor, 71000)
    assert (
        calls[-1].tobytes()
        == (
            driver.init_time + stopped_phase(np.arange(6, dtype=np.float64) * 0.02)
        ).tobytes()
    )
    driver.sample(20, 0.02)
    assert (
        calls[-1].tobytes()
        == (
            driver.init_time + stopped_phase(0.02 + 0.02 + np.arange(6) * 0.02)
        ).tobytes()
    )
    assert len(calls[-1]) == 6


@pytest.mark.parametrize("fault", ["mode", "T", "sha", "axis", "formula"])
def test_reference_identity_drift_rejected(native_factory, fault):
    driver = WalkToHoldReferenceDriver(native_factory[1], 71000)
    if fault == "mode":
        driver.profile = "walking"
    elif fault == "sha":
        driver.identity_sha256 = "0" * 64
    else:
        field = {
            "T": "stop_time_s",
            "axis": "native_axes_sha256",
            "formula": "phase_clock",
        }[fault]
        driver.identity[field] = 0.5 if fault == "T" else "corrupt"
    with pytest.raises(ValueError):
        driver.sample(20, 0.02)


@pytest.mark.parametrize(
    "tick,time", [(1, 0.001), (20, 0.04), (20, np.nan), (-20, -0.02), (True, 0.001)]
)
def test_real_clock_and_control_cadence_not_warped(native_factory, tick, time):
    driver = WalkToHoldReferenceDriver(native_factory[1], 71000)
    with pytest.raises(ValueError):
        driver.sample(tick, time)


def test_profile_contract_fresh_copy():
    changed = reference_contract()
    changed["stop_time_s"] = 9
    assert reference_contract()["stop_time_s"] == 0.4
