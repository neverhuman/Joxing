import numpy as np
import pytest

from joxing.physics import momentum_fixture


@pytest.mark.parametrize("angle",[0.,.7,2.1])
@pytest.mark.parametrize("reverse",[False,True])
def test_signed_impulse_equals_independent_momentum_change(angle,reverse):
    r=momentum_fixture(angle=angle,reverse=reverse)
    assert r["error"] < 1e-9
    assert r["force_samples"]>0
    assert r["first_approach_mps"] == pytest.approx(2.)
    assert np.dot(r["impulse_Ns"],[np.cos(angle),np.sin(angle),0])<0


def test_fixture_impulse_converges_on_halved_timestep():
    a=momentum_fixture(.001); b=momentum_fixture(.0005)
    assert np.linalg.norm(np.array(a["impulse_Ns"])-b["impulse_Ns"])/np.linalg.norm(b["impulse_Ns"]) < .05
