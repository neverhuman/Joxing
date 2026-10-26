import mujoco
import numpy as np
import pytest

from joxing.scene import build_scene


@pytest.fixture(scope="module")
def scene():
    return build_scene()


def test_two_complete_anatomies_and_equipment_mass(scene):
    assert (scene.model.nq, scene.model.nv, scene.model.nu, scene.model.neq) == (
        170,
        170,
        1400,
        84,
    )
    assert scene.report["mass_kg"] == pytest.approx(
        scene.report["expected_mass_kg"], abs=1e-10
    )
    assert all(
        error == 0
        for field in scene.report["preserved_actuator_max_error"].values()
        for error in field.values()
    )


def test_independent_actuator_ownership(scene):
    a, b = scene.actuator_ids.values()
    assert len(a) == len(b) == 700 and not set(a) & set(b)
    d = mujoco.MjData(scene.model)
    d.ctrl[a] = 0.7
    assert np.all(d.ctrl[b] == 0.0)


def test_muscle_tendon_references_stay_inside_actor(scene):
    m = scene.model
    for a, ids in scene.actuator_ids.items():
        for i in ids:
            tendon = m.tendon(m.actuator_trnid[i, 0])
            assert tendon.name.startswith(a + "_")


def test_semantic_registry_covers_every_physical_geom(scene):
    m = scene.model
    assert len(scene.report["physical_geoms"]) == np.count_nonzero(
        m.geom_contype | m.geom_conaffinity
    )
    names = [g["name"] for g in scene.report["physical_geoms"]]
    assert len(names) == len(set(names))
