"""Batching preserves every observation/action row; no large checkpoint needed."""

import runpy

import numpy as np
import pytest

from joxing.provenance import ROOT

helpers = runpy.run_path(str(ROOT / "scripts/benchmark_inference.py"))
evaluate = helpers["evaluate_batch"]


class Normalizer:
    def apply(self, obs):
        return obs * 0.1


class Model:
    def predict(self, obs, deterministic=True):
        return np.tanh(obs[..., :700]), None


@pytest.mark.parametrize("batch", [1, 2, 8, 16])
def test_batching_keeps_row_identity_and_seven_hundred_actions(batch):
    raw = np.arange(batch * 3601, dtype=np.float32).reshape(batch, 3601) * 1e-5
    before = raw.copy()
    scalar = evaluate(Model(), Normalizer(), raw, "scalar")
    batched = evaluate(Model(), Normalizer(), raw, "batched")
    assert scalar.shape == (batch, 700) and np.array_equal(scalar, batched)
    assert np.array_equal(raw, before)
    assert np.array_equal(
        evaluate(Model(), Normalizer(), raw[::-1], "batched"), scalar[::-1]
    )


def test_bad_observations_and_modes_fail_before_wrong_action_application():
    for raw in (np.zeros((2, 3600)), np.full((2, 3601), np.nan)):
        with pytest.raises(ValueError):
            evaluate(Model(), Normalizer(), raw, "batched")
    with pytest.raises(ValueError):
        evaluate(Model(), Normalizer(), np.zeros((2, 3601)), "unknown")


def test_hidden_state_is_not_silently_lost():
    class Recurrent(Model):
        def predict(self, obs, deterministic=True):
            return super().predict(obs)[0], np.ones(1)

    for mode in ("scalar", "batched"):
        with pytest.raises(RuntimeError, match="memory"):
            evaluate(Recurrent(), Normalizer(), np.zeros((2, 3601)), mode)


def test_all_preregistered_cases_are_preserved():
    import json

    from joxing.provenance import digest

    p = json.loads((ROOT / "experiments/v0.0.9/protocol.json").read_text())
    assert digest(p) == helpers["PROTOCOL_SHA"]
    cases = helpers["planned_cases"](p)
    assert len(cases) == 32 and len({r["case_id"] for r in cases}) == 32
    assert all(r["status"] == "not_run" for r in cases)
