import mujoco
import numpy as np
import pytest

from joxing.physics import point_velocity, step_contacts


def test_point_velocity_includes_rotation_about_body_center():
    m = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0"/>
        <worldbody><body pos="0 0 0"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qvel[5] = 3.0
    mujoco.mj_forward(m, d)
    point = np.array([0.2, 0.1, 0.0])
    assert point_velocity(m, d, 1, point) == pytest.approx(np.cross([0, 0, 3.0], point))
    assert np.linalg.norm(point_velocity(m, d, 1, point)) > 0


def test_contact_filter_preserves_force_and_staged_point_velocities():
    m = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0" integrator="implicitfast"/>
        <worldbody><geom name="fixed" type="sphere" size=".1"/>
        <body pos="-.19 0 0"><freejoint/><geom name="moving" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>""")
    a, b = mujoco.MjData(m), mujoco.MjData(m)
    a.qvel[0] = b.qvel[0] = 2.0
    full = step_contacts(m, a)
    filtered = step_contacts(m, b, {m.geom("moving").id})
    assert len(full) == len(filtered) == 1
    assert full[0].force_on_geom2 == pytest.approx(filtered[0].force_on_geom2)
    assert full[0].velocity1_at_contact[0] - full[0].velocity2_at_contact[
        0
    ] == pytest.approx(2.0 if full[0].geom1 == m.geom("moving").id else -2.0)
    assert a.qpos == pytest.approx(b.qpos)
