import copy
import json

import mujoco
import numpy as np
import pytest

from joxing.bout_recording import STATE, BoutRecorder, frame_schedule, load_recording
from joxing.scene import build_scene


def test_shared_world_capture_is_read_only_and_corruption_rejected(tmp_path):
    scene = build_scene()
    data = mujoco.MjData(scene.model)
    data.qpos[:] = scene.report["initial_qpos"]
    mujoco.mj_forward(scene.model, data)
    recorder = BoutRecorder(scene, {"A": "fixture-A", "B": "fixture-B"}, max_frames=3)
    for tick in range(3):
        state = np.empty(mujoco.mj_stateSize(scene.model, STATE))
        mujoco.mj_getState(scene.model, data, state, STATE)
        recorder.capture(data, physics_steps=tick)
        after = np.empty_like(state)
        mujoco.mj_getState(scene.model, data, after, STATE)
        np.testing.assert_array_equal(state, after)
        if tick < 2:
            mujoco.mj_step(scene.model, data)
    with pytest.raises(TimeoutError):
        recorder.capture(data, physics_steps=3)
    path = tmp_path / "recording"
    report = recorder.save(path)
    loaded, states = load_recording(scene, path)
    assert report == loaded and states.shape[0] == 3
    assert loaded["classification"].startswith("DIAGNOSTIC")
    assert loaded["new_render_physics_steps"] == 0
    with pytest.raises(FileExistsError):
        recorder.save(path)
    foreign = copy.copy(scene)
    foreign.report = dict(scene.report, scene_hash="foreign")
    with pytest.raises(ValueError, match="Foreign"):
        load_recording(foreign, path)
    changed = copy.deepcopy(report)
    changed["frames"][1]["physics_steps"] = 9
    (path / "recording.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="clock/count"):
        load_recording(scene, path)
    report["frames"][1]["state_sha256"] = "0" * 64
    (path / "recording.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="checksum"):
        load_recording(scene, path)


def test_video_sampling_is_causal():
    report = {"frames": [{"time_s": t} for t in (0, 0.03, 0.09, 0.13)]}
    assert frame_schedule(report, fps=25).tolist() == [0, 1, 1, 2]
    with pytest.raises(ValueError):
        frame_schedule(report, fps=0)


def test_integer_clock_preserves_every_registered_frame_despite_engine_roundoff():
    report = {
        "physics_dt_s": 0.001,
        "frames": [
            {"time_s": t * 0.001 + 1e-14 if t else 0.0, "physics_steps": t}
            for t in range(0, 1001, 40)
        ],
    }
    assert frame_schedule(report, fps=25).tolist() == list(range(26))
