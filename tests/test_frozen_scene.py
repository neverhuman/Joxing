import copy

import mujoco
import numpy as np
import pytest

from joxing.env import BoxingEnv
from joxing.frozen_scene import file_sha, load_frozen_scene
from joxing.motor_storage import write_json
from joxing.scene import build_scene, save_scene


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("compiled-scene")
    scene = build_scene()
    binary, report = root / "scene.mjb", root / "scene.json"
    mujoco.mj_saveModel(scene.model, str(binary))
    write_json(report, scene.report)
    return binary, report, scene


def load(binary, report, scene, **kwargs):
    return load_frozen_scene(
        binary,
        report,
        binary_sha256=kwargs.get("binary_sha256", file_sha(binary)),
        report_sha256=kwargs.get("report_sha256", file_sha(report)),
        scene_hash=scene.report["scene_hash"],
    )


def test_exact_compiled_model_loads_without_rebuild_or_export(saved, tmp_path):
    binary, report, original = saved
    frozen = load(binary, report, original)
    assert frozen.report == original.report
    for a in ("A", "B"):
        np.testing.assert_array_equal(frozen.actuator_ids[a], original.actuator_ids[a])
        assert not frozen.actuator_ids[a].flags.writeable
    env = BoxingEnv(frozen)
    env.reset(seed=90201)
    assert env.physics_total == 0 and env.model.nu == 1400
    with pytest.raises(RuntimeError, match="cannot export/rebuild"):
        save_scene(frozen, tmp_path / "export")


@pytest.mark.parametrize(
    "kind",
    [
        "binary_hash",
        "report_hash",
        "dimensions",
        "muscles",
        "duplicate_geom",
        "missing_geom",
        "mask",
        "initial",
    ],
)
def test_compiled_loader_rejects_self_consistent_input_corruption(
    saved, tmp_path, kind
):
    binary, report, original = saved
    if kind in {"binary_hash", "report_hash"}:
        with pytest.raises(ValueError, match="bytes differ"):
            load(binary, report, original, **{kind.replace("hash", "sha256"): "0" * 64})
        return
    value = copy.deepcopy(original.report)
    if kind == "dimensions":
        value["dimensions"]["nu"] = 1399
    elif kind == "muscles":
        value["actuator_ids"]["B"][0] = value["actuator_ids"]["A"][0]
    elif kind == "duplicate_geom":
        value["physical_geoms"].append(value["physical_geoms"][0])
    elif kind == "missing_geom":
        value["physical_geoms"].pop()
    elif kind == "mask":
        value["physical_geoms"][0]["contype"] += 1
    else:
        value["initial_qpos"] = [0.0]
    altered = tmp_path / "scene.json"
    write_json(altered, value)
    assert file_sha(altered) != file_sha(report)
    with pytest.raises(ValueError):
        load(binary, altered, original)


def test_hash_bound_modified_muscles_still_fail_native_parameter_gate(saved, tmp_path):
    _, report, original = saved
    model = mujoco.MjModel.from_binary_path(str(saved[0]))
    model.actuator_gainprm[0, 0] += 0.1
    binary = tmp_path / "changed.mjb"
    mujoco.mj_saveModel(model, str(binary))
    with pytest.raises(ValueError, match="muscle parameters"):
        load(binary, report, original)
