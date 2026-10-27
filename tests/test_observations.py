from dataclasses import replace

import mujoco
import numpy as np
import pytest

from joxing.bodies import NativeBodyAdapter
from joxing.observations import DIMENSION, FEATURE_SPEC, PublicState, observe
from joxing.scene import build_scene


@pytest.fixture(scope="module")
def measurements():
    scene = build_scene()
    data = mujoco.MjData(scene.model)
    data.qpos[:] = scene.report["initial_qpos"]
    mujoco.mj_forward(scene.model, data)
    return tuple(NativeBodyAdapter(scene, a).measure(data) for a in ("A", "B"))


def public():
    return PublicState(
        "game",
        0,
        60,
        (0, 0),
        np.array([[-2.5, -2.5, 0], [2.5, -2.5, 0], [2.5, 2.5, 0], [-2.5, 2.5, 0]]),
    )


def test_corner_frame_and_owned_real_kinematics(measurements):
    a, b = measurements
    oa, ob = (observe(x, y, public(), np.zeros(700)) for x, y in [(a, b), (b, a)])
    assert len(oa.values) == DIMENSION == FEATURE_SPEC[-1]["stop"]
    assert oa.values.dtype == np.float32 and not oa.values.flags.writeable
    np.testing.assert_allclose(oa.values, ob.values, atol=1e-6, rtol=0)
    moved = replace(
        b, world_linear_velocity_mps=b.world_linear_velocity_mps + np.array([1, 0, 0])
    )
    assert not np.array_equal(
        observe(a, moved, public(), np.zeros(700)).values, oa.values
    )


def test_private_opponent_fields_never_enter_actor_and_ticks_must_match(measurements):
    a, b = measurements
    baseline = observe(a, b, public(), np.zeros(700)).values
    private = replace(
        b, activation=b.activation + 0.6, qpos=b.qpos + 0.2, qvel=b.qvel - 0.3
    )
    np.testing.assert_array_equal(
        observe(a, private, public(), np.zeros(700)).values, baseline
    )
    with pytest.raises(ValueError, match="same completed"):
        observe(a, replace(b, time_s=0.001), public(), np.zeros(700))
    with pytest.raises(ValueError, match="ownership"):
        observe(a, a, public(), np.zeros(700))
    points = [0, 1]
    immutable_public = replace(public(), points=points)
    points[0] = 99
    assert immutable_public.points == (0, 1)


def test_pure_global_yaw_translation_frame_invariance(measurements):
    a, b = measurements
    p = public()
    theta = 0.7
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    shift = np.array([3, -2, 0])

    def transform(body):
        return replace(
            body,
            world_position_m=body.world_position_m @ rotation.T + shift,
            world_linear_velocity_mps=body.world_linear_velocity_mps @ rotation.T,
            world_rotation_matrix=np.einsum(
                "ij,njk->nik", rotation, body.world_rotation_matrix
            ),
            world_angular_velocity_radps=body.world_angular_velocity_radps @ rotation.T,
        )

    changed = replace(
        p, ring_vertices_world_m=p.ring_vertices_world_m @ rotation.T + shift
    )
    np.testing.assert_allclose(
        observe(transform(a), transform(b), changed, np.zeros(700)).values,
        observe(a, b, p, np.zeros(700)).values,
        atol=1e-6,
        rtol=0,
    )
    # This is frame arithmetic on measured arrays, not dynamic motor equivariance.
