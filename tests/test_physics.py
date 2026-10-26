import numpy as np
import pytest

from joxing.physics import momentum_fixture, scoring_fixture


@pytest.mark.parametrize("angle", [0.0, 0.7, 2.1])
@pytest.mark.parametrize("reverse", [False, True])
def test_signed_impulse_equals_independent_momentum_change(angle, reverse):
    r = momentum_fixture(angle=angle, reverse=reverse)
    assert r["error"] < 1e-9
    assert r["force_samples"] > 0
    assert r["first_approach_mps"] == pytest.approx(2.0)
    assert np.dot(r["impulse_Ns"], [np.cos(angle), np.sin(angle), 0]) < 0


def test_fixture_impulse_converges_on_halved_timestep():
    a = momentum_fixture(0.001)
    b = momentum_fixture(0.0005)
    assert (
        np.linalg.norm(np.array(a["impulse_Ns"]) - b["impulse_Ns"])
        / np.linalg.norm(b["impulse_Ns"])
        < 0.05
    )


@pytest.mark.parametrize("zone,points", [("head", 2), ("torso", 1), ("guard", 0)])
def test_real_solver_force_drives_one_expected_referee_event(zone, points):
    fixture = scoring_fixture(zone)
    assert len(fixture["events"]) == 1
    event = fixture["events"][0]
    assert event["points"] == points
    assert event["window_impulse_Ns"] > 0.25
