import json
import shutil

import pytest

from joxing.motor_storage import save_npz, write_json
from joxing.scene import build_scene
from joxing.sensed_env import SensedBoxingEnv
from joxing.sensor_audit import read_arrays
from joxing.sensor_evidence import SensorTape
from joxing.sensor_physics_audit import audit_sensor_physics

DELAYS = {"A": 1, "B": 2}


@pytest.fixture(scope="module")
def contact(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sensed-contact")
    scene = build_scene()
    env = SensedBoxingEnv(
        scene,
        horizon_ticks=20,
        scenario="close-overlap-collision-probe-v2",
        sensor_schedule={"A": [1], "B": [2]},
    )
    env.reset(seed=90204)
    tape = SensorTape(env, maximum_ticks=20)
    env.begin_control(tape.choose(env), delays=DELAYS)
    while not env.done:
        env.advance_substep()
        tape.capture(env)
    tape.save(env, folder)
    write_json(folder / "contract.json", env.contract)
    return folder, scene


def test_sensed_physical_contacts_and_global_bell_are_audited_without_physics(contact):
    folder, scene = contact
    r = audit_sensor_physics(folder, scene, expected_physics=20, delays=DELAYS)
    assert r["physical_referee_audited"] and r["audit_new_physics"] == 0
    assert r["positive_glove_touches"] > 0 and not r["formal_sensor_proof"]


@pytest.mark.parametrize(
    "kind",
    [
        "force_sign",
        "approach",
        "touch",
        "glove",
        "event",
        "decision_points",
        "transition",
        "offset",
        "geom",
    ],
)
def test_sensor_physical_audit_rejects_semantic_corruption(contact, tmp_path, kind):
    source, scene = contact
    folder = tmp_path / "bad"
    shutil.copytree(source, folder)
    trace = json.loads((folder / "trace.json").read_text())
    tape = read_arrays(folder / "tape.npz")
    if kind == "force_sign":
        tape["contacts"][0, 6:9] *= -1
    elif kind == "approach":
        tape["contacts"][0, 16] += 1
    elif kind == "touch":
        next(g for g in trace["intervals"][0]["gloves"].values() if g["touches"])[
            "touches"
        ][0]["normal_force_N"] += 1
    elif kind == "glove":
        trace["intervals"][0]["gloves"].pop(next(iter(trace["intervals"][0]["gloves"])))
    elif kind == "event":
        trace["intervals"][0]["events"].append({"event_id": "invented", "points": 2})
    elif kind == "decision_points":
        trace["decisions"][0]["points"][0] += 1
    elif kind == "transition":
        trace["transition"]["bootstrap_mask"]["B"] = 1.0
    elif kind == "offset":
        tape["contact_offsets"][-1] -= 1
    else:
        tape["contact_geoms"][0, 0] = 100000
    write_json(folder / "trace.json", trace)
    save_npz(folder / "tape.npz", **tape)
    with pytest.raises(ValueError):
        audit_sensor_physics(folder, scene, expected_physics=20, delays=DELAYS)
