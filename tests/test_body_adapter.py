import mujoco
import numpy as np
import pytest

from joxing.bodies import NativeBodyAdapter
from joxing.scene import build_scene


def test_both_corners_read_real_shared_kinematics_without_aliases():
    scene = build_scene()
    a, b = (NativeBodyAdapter(scene, corner) for corner in ("A", "B"))
    assert not set(a.actuator_ids) & set(b.actuator_ids)
    assert not set(a.activation_ids) & set(b.activation_ids)
    data = mujoco.MjData(scene.model)
    data.qpos[:] = scene.report["initial_qpos"]
    data.qvel[a.qvel_ids] = np.linspace(-0.05, 0.05, 85)
    data.act[a.activation_ids] = 0.25
    data.act[b.activation_ids] = 0.75
    mujoco.mj_forward(scene.model, data)
    before = data.qpos.copy(), data.qvel.copy(), data.act.copy(), data.time
    ma, mb = a.measure(data), b.measure(data)
    assert ma.time_s == mb.time_s == data.time
    assert ma.qpos.shape == ma.qvel.shape == (85,)
    assert ma.activation.shape == mb.activation.shape == (700,)
    np.testing.assert_array_equal(ma.activation, 0.25)
    np.testing.assert_array_equal(mb.activation, 0.75)
    np.testing.assert_array_equal(mb.world_linear_velocity_mps, 0)
    assert np.max(np.abs(ma.world_linear_velocity_mps)) > 0
    for k, body in enumerate(a.body_ids):
        jacp, jacr = np.zeros((3, scene.model.nv)), np.zeros((3, scene.model.nv))
        mujoco.mj_jac(scene.model, data, jacp, jacr, data.xpos[body], body)
        np.testing.assert_allclose(
            ma.world_linear_velocity_mps[k], jacp @ data.qvel, atol=1e-15
        )
    for array in (
        ma.qpos,
        ma.qvel,
        ma.activation,
        ma.world_position_m,
        ma.world_linear_velocity_mps,
    ):
        assert not array.flags.writeable and not np.shares_memory(array, data.qpos)
    for actual, expected in zip(
        (data.qpos, data.qvel, data.act, data.time), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    data.act[a.activation_ids] = 0
    np.testing.assert_array_equal(ma.activation, 0.25)
    data.qvel[a.qvel_ids[0]] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        a.measure(data)
