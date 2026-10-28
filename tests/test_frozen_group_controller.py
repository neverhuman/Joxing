import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from joxing.frozen_group_controller import FrozenGroupController
from joxing.provenance import ROOT


@pytest.fixture(scope="module")
def params():
    return json.loads(
        (
            ROOT / "grokbot/redteam/data/cycle-0025-hit-stay-gate-summary.json"
        ).read_text()
    )["final_params"]


def observation(tick):
    return SimpleNamespace(
        sampled_tick=tick,
        values=np.linspace(-0.1, 0.2, 1970, dtype=np.float32) + np.float32(tick * 1e-5),
    )


def test_bridge_matches_published_policy_and_resumes_private_origin_and_rng(params):
    from grokbot.motor_primitives import GroupPDPolicy

    bridge, historical = (
        FrozenGroupController(params),
        GroupPDPolicy(params, noise=0.05),
    )
    rng, original_rng = np.random.default_rng(19), np.random.default_rng(19)
    for tick in [0, 20, 40]:
        assert np.array_equal(
            bridge(observation(tick), None, rng),
            historical(observation(tick), None, original_rng),
        )
    state = bridge.snapshot()
    unchanged = copy.deepcopy(state)
    resumed = FrozenGroupController(params)
    resumed.restore(state)
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    for tick in [60, 80, 100]:
        assert np.array_equal(
            bridge(observation(tick), None, rng),
            resumed(observation(tick), object(), restored_rng),
        )
    assert state == unchanged
    resumed._actor.punch.posture.origin[0] += 1.0
    assert state == unchanged
    assert not bridge.bundle["uses_history"] and not bridge.bundle["uses_opponent"]


@pytest.mark.parametrize("fault", ["bundle", "shape", "nan"])
def test_restore_is_atomic_and_rejects_changed_identity_or_origin(params, fault):
    controller = FrozenGroupController(params)
    controller(observation(0), None, np.random.default_rng(0))
    before = controller.snapshot()
    # Establish a valid baseline before corrupting the saved state.
    controller.restore(before)
    changed = copy.deepcopy(before)
    if fault == "bundle":
        changed["bundle_sha256"] = "wrong"
    elif fault == "shape":
        changed["origin"] = [0.0]
    else:
        changed["origin"][0] = float("nan")
    with pytest.raises(ValueError):
        controller.restore(changed)
    assert controller.snapshot() == before


def test_policy_summary_requires_exact_bytes_and_holds_private_parameters(
    params, tmp_path
):
    f = tmp_path / "summary.json"
    f.write_text(json.dumps({"final_params": params}))
    private_params = list(params)
    controller = FrozenGroupController(private_params)
    before = controller._params.copy()
    private_params[0] += 1  # Constructor retained a private, read-only copy.
    assert np.array_equal(controller._params, before)
    with pytest.raises(ValueError, match="summary hash"):
        FrozenGroupController.from_summary(f, expected_sha256="wrong")
