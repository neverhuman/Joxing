import json

import pytest

from joxing.referee import GloveInterval, MatchReferee
from joxing.rules import Touch
from joxing.scene import build_scene


@pytest.fixture(scope="module")
def scene():
    return build_scene()


def intervals(ref):
    # Pure logical trade fixture; this is not a physical human punch result.
    return {
        name: GloveInterval(
            (Touch(name, "head" if name[0] == "A" else "torso", 100.0, 2.0, 2.0, 1.0),),
            0.0,
            0.4,
        )
        for name in ref.gloves
    }


def test_all_four_gloves_finalize_prior_impulses_before_bell_and_order_is_irrelevant(
    scene,
):
    a, b = MatchReferee(scene, "trade"), MatchReferee(scene, "trade")
    for t in range(3):
        rows = intervals(a)
        assert a.commit(rows, t, bell=t == 2) == b.commit(
            dict(reversed(list(rows.items()))), t, bell=t == 2
        )
    assert a.points == (4, 2) and len(a.events) == 4
    assert a.snapshot() == b.snapshot() and a.next_tick == 3
    restored = MatchReferee.restore(scene, json.loads(json.dumps(a.snapshot())))
    assert restored.snapshot() == a.snapshot()
    with pytest.raises(RuntimeError):
        a.commit(intervals(a), 3)
