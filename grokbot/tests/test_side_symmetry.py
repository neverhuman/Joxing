"""Red-team invariant: neither corner has a built-in advantage.

Paired evaluation with swapped sides, and one shared self-play policy acting for
both corners, assume the world is side-equivariant: a game where A plays c1 and B
plays c2 must be the mirror of the game where A plays c2 and B plays c1. Scene,
contact ordering, observation or referee asymmetries would silently bias every
Elo/payoff estimate built on top of it.
"""

import numpy as np
import pytest

from joxing.env import PLAYERS, BoxingEnv
from joxing.observations import FEATURE_SPEC
from joxing.scene import build_scene

SCENARIOS = ("native-keyframe-v1", "close-overlap-collision-probe-v2")
STEPS = 25  # 500 physics ticks at 20 ticks per control step
TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def scene():
    return build_scene()


def _blocks_over(a, b):
    return [
        block["name"]
        for block in FEATURE_SPEC
        if np.max(
            np.abs(
                a[block["start"] : block["stop"]] - b[block["start"] : block["stop"]]
            )
        )
        > TOLERANCE
    ]


def _rollout(scene, scenario, swap):
    env = BoxingEnv(scene, horizon_ticks=2000, scenario=scenario)
    env.reset(seed=0)
    rng = np.random.default_rng(7)
    frames = []
    for _ in range(STEPS):
        c1 = rng.uniform(-1, 1, 700).astype(np.float32)
        c2 = rng.uniform(-1, 1, 700).astype(np.float32)
        result = env.step({"A": c2, "B": c1} if swap else {"A": c1, "B": c2})
        assert result["rewards"]["A"] == -result["rewards"]["B"]
        frames.append(
            (
                {a: result["observations"][a].values.copy() for a in PLAYERS},
                result["info"]["points"],
            )
        )
    registry = {g["id"]: g for g in scene.report["physical_geoms"]}
    inter_actor = sum(
        {registry[c["geom1"]]["actor"], registry[c["geom2"]]["actor"]} == {"A", "B"}
        for row in env.evidence
        for c in row["contacts"]
    )
    return frames, inter_actor


@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_reset_observations_are_corner_identical(scene, seed):
    observations, _ = BoxingEnv(scene, horizon_ticks=2000).reset(seed=seed)
    a, b = observations["A"].values, observations["B"].values
    assert _blocks_over(a, b) == []


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_swapping_corners_mirrors_the_game(scene, scenario):
    direct, contacts = _rollout(scene, scenario, swap=False)
    swapped, swapped_contacts = _rollout(scene, scenario, swap=True)
    assert contacts == swapped_contacts
    if scenario != "native-keyframe-v1":
        # Exercise the contact solver's A/B ordering, not just free motion.
        assert contacts > 1000
    for step, ((obs, points), (obs_swap, points_swap)) in enumerate(
        zip(direct, swapped, strict=True)
    ):
        assert _blocks_over(obs["A"], obs_swap["B"]) == [], step
        assert _blocks_over(obs["B"], obs_swap["A"]) == [], step
        assert points == points_swap[::-1], step
