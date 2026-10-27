"""Zero-step semantic teacher feature checks; no motor competence claim."""

import mujoco
import numpy as np
import pytest

from joxing.scene import SOURCE, build_scene
from joxing.shared_motor import (
    MotorReferenceState,
    NativeMotorMap,
    native_teacher_observation,
)


@pytest.fixture(scope="module")
def fixture():
    scene = build_scene()
    canonical = mujoco.MjModel.from_xml_path(str(SOURCE))
    data = mujoco.MjData(scene.model)
    data.qpos[:] = scene.report["initial_qpos"]
    mujoco.mj_forward(scene.model, data)
    # Synthetic unique sampled dynamics identify any cross-corner mix-up. This
    # is not an evaluated trajectory or a model stepping/normalization oracle.
    data.qvel[:] = np.arange(scene.model.nv) / 10
    data.qacc[:] = np.arange(scene.model.nv) / 7
    data.act[:] = np.arange(scene.model.na) / 2000
    data.actuator_force[:] = np.arange(scene.model.nu) * 13
    data.actuator_velocity[:] = np.arange(scene.model.nu) / 5
    return scene, canonical, data


@pytest.mark.parametrize("actor,yaw", [("A", 0.0), ("B", np.pi)])
def test_teacher_features_match_native_semantic_order_and_do_not_mutate(
    fixture, actor, yaw
):
    scene, canonical, data = fixture
    mapping = NativeMotorMap(scene.model, canonical, actor, spawn_yaw_rad=yaw)
    reference = MotorReferenceState(
        np.arange(85), np.arange(425).reshape(5, 85), np.arange(18).reshape(6, 3)
    )
    before = {
        key: getattr(data, key).copy()
        for key in [
            "qpos",
            "qvel",
            "qacc",
            "act",
            "ctrl",
            "actuator_force",
            "actuator_velocity",
        ]
    }
    values, blocks = native_teacher_observation(data, mapping, reference)
    assert values.shape == (3601,) and values.dtype == np.float32
    expected_velocity = np.array(
        [
            data.qvel[
                scene.model.joint(actor + "_" + canonical.joint(i).name).dofadr[0]
            ]
            for i in range(85)
        ]
    )
    assert np.array_equal(blocks["qvel"], expected_velocity)
    assert np.array_equal(blocks["qacc"], data.qacc[mapping.qvel_ids])
    assert np.array_equal(
        blocks["actuator_forces"],
        (data.actuator_force[mapping.actuator_ids] / 1000).clip(-100, 100),
    )
    assert np.array_equal(
        blocks["actuator_velocity"],
        data.actuator_velocity[mapping.actuator_ids].clip(-100, 100),
    )
    if actor == "B":
        world = data.xpos[mapping.keypoint_ids] - data.xpos[mapping.pelvis_id]
        assert np.allclose(
            blocks["key_xpos"].reshape(6, 3), world * [-1, -1, 1], rtol=0, atol=1e-14
        )
    for key, old in before.items():
        assert np.array_equal(getattr(data, key), old)
    assert not values.flags.writeable and all(
        not x.flags.writeable for x in blocks.values()
    )
    with pytest.raises(ValueError):
        values[0] = 1


def test_native_spawn_frame_must_be_explicit_and_correct(fixture):
    scene, canonical, _ = fixture
    with pytest.raises(ValueError, match="yaw"):
        NativeMotorMap(scene.model, canonical, "B", spawn_yaw_rad=0)


@pytest.mark.parametrize("kind", ["shape", "nonfinite"])
def test_reference_rejects_wrong_future_targets(kind):
    future = np.zeros((5, 85)) if kind == "nonfinite" else np.zeros((4, 85))
    if kind == "nonfinite":
        future[0, 0] = np.nan
    with pytest.raises(ValueError):
        MotorReferenceState(np.zeros(85), future, np.zeros((6, 3)))
