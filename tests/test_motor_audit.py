"""Adversarial independent audit over actual synthetic-teacher physical tapes."""

import copy
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from joxing.motor_audit import evaluation, replay_tail
from joxing.motor_evaluation import collect, replay
from joxing.motor_storage import write_json


def test_independent_physical_evaluation_and_corruption(tmp_path):
    from test_motor_evaluation import ZeroResidual
    from test_motor_residual import environment

    env, model = environment(), ZeroResidual()
    try:
        context = {
            "physical_specification": env.task.specification(),
            "floor_geom_id": env.task.floor,
        }
        from joxing.provenance import digest

        assert digest(context["physical_specification"]) == env.task.contract_hash
        result = collect(
            env,
            model,
            tmp_path / "eval",
            {"training_seed": 81011, "stage": "initial", "evaluation_seed": 83000},
            lambda n: None,
        )
        assert result["status"] == "completed"
        found = evaluation(tmp_path / "eval", env.groups, context)
        assert found["physics_steps"] == result["physics_steps"]
        rr = replay(env, model, tmp_path / "eval", tmp_path / "replay", lambda n: None)
        assert rr["status"] == "completed"
        assert replay_tail(tmp_path / "replay", tmp_path / "eval")["passed"]
        tracepath = tmp_path / "eval/trace.json"
        trace = json.loads(tracepath.read_text())
        altered = copy.deepcopy(trace)
        altered[-1]["reward"] += 0.001
        write_json(tracepath, altered)
        with pytest.raises(AssertionError):
            evaluation(tmp_path / "eval", env.groups, context)
        write_json(tracepath, trace)
        path = tmp_path / "eval/substeps.npz"
        with np.load(path, allow_pickle=False) as a:
            values = {k: a[k].copy() for k in a.files}
        original = {k: v.copy() for k, v in values.items()}
        values["contact_force_N"][0, 2] *= -1
        assert values["contact_force_N"][0, 2] != 0
        np.savez_compressed(path, **values)
        with pytest.raises(AssertionError):
            evaluation(tmp_path / "eval", env.groups, context)
        np.savez_compressed(path, **original)
        path = tmp_path / "eval/actions.npz"
        with np.load(path, allow_pickle=False) as a:
            values = {k: a[k].copy() for k in a.files}
        values["commands"][0, 0] += np.float32(0.001)
        np.savez_compressed(path, **values)
        with pytest.raises(AssertionError):
            evaluation(tmp_path / "eval", env.groups, context)
    finally:
        env.close()
