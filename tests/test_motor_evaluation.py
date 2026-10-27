"""Saved held-out evidence, held-command continuation and retained failures."""

import copy
import json

import numpy as np

from joxing.motor_evaluation import collect, replay


class ZeroResidual:
    def __init__(self):
        import torch

        self.actor = torch.nn.Linear(1, 1)

    def predict(self, observation, deterministic):
        assert deterministic
        return np.zeros(138, np.float32), None


def test_actual_evaluation_and_mid_command_replay(tmp_path):
    from test_motor_residual import environment

    env, model = environment(), ZeroResidual()
    row = {"training_seed": 81011, "stage": "initial", "evaluation_seed": 83000}
    try:
        result = collect(env, model, tmp_path / "evaluation", row, lambda n: None)
        assert result["status"] == "completed", result
        assert result["controls"] > 10 and result["mid_snapshot_available"]
        assert result["physics_steps"] == result["controls"] * 10
        assert 0 < result["survival_s"] <= 20
        before = env.task.physics_total
        result2 = replay(
            env,
            copy.deepcopy(model),
            tmp_path / "evaluation",
            tmp_path / "replay",
            lambda n: None,
        )
        assert result2["status"] == "completed" and result2["exact"], result2
        assert result2["physics_steps"] == result["physics_steps"] - 103
        assert env.task.physics_total - before == result2["physics_steps"]
        trace = json.loads((tmp_path / "evaluation/trace.json").read_text())
        assert trace[-1]["trace_mask"] == 0
        assert trace[-1]["bootstrap_mask"] == float(not trace[-1]["terminated"])
        # Corrupted actor metadata must stop before any replay physics.
        path = tmp_path / "evaluation/result.json"
        altered = json.loads(path.read_text())
        altered["residual_actor_hash"] = "wrong"
        path.write_text(json.dumps(altered))
        result3 = replay(
            env, model, tmp_path / "evaluation", tmp_path / "rejected", lambda n: None
        )
        assert result3["status"] == "invalid" and result3["physics_steps"] == 0
        assert (tmp_path / "rejected/fault-arrays.npz").exists()
    finally:
        env.close()


def test_evaluation_budget_stop_retains_attempt(tmp_path):
    from test_motor_residual import environment

    env, model = environment(), ZeroResidual()

    def guard(n):
        if n:
            raise TimeoutError("registered budget")

    try:
        result = collect(
            env,
            model,
            tmp_path / "stopped",
            {"training_seed": 81029, "stage": "initial", "evaluation_seed": 83007},
            guard,
        )
        assert result["status"] == "incomplete" and result["physics_steps"] == 0
        assert result["controls"] == 0 and "survival_s" not in result
        assert (tmp_path / "stopped/fault-arrays.npz").exists()
        assert (
            json.loads((tmp_path / "stopped/result.json").read_text())[
                "evaluation_seed"
            ]
            == 83007
        )
    finally:
        env.close()
