"""Trace diagnostics preserve outputs, row identity and fault visibility."""

import json
import runpy

import numpy as np
import pytest
import torch

from joxing.provenance import ROOT, digest

helpers = runpy.run_path(str(ROOT / "scripts/diagnose_inference_numerics.py"))


def test_hooks_clone_before_downstream_mutation_and_always_remove():
    actor = torch.nn.Sequential(torch.nn.Identity(), torch.nn.ReLU(inplace=True))
    values = torch.tensor([[-1.0, 2.0]])
    with helpers["output_trace"](actor) as trace:
        result = actor(values.clone())
    assert torch.equal(result, actor(values.clone()))
    assert trace[0][1][0, 0] == -1 and trace[-1][1][0, 0] == 0
    assert all(not m._forward_hooks for m in actor.modules())
    with pytest.raises(RuntimeError), helpers["output_trace"](actor):
        raise RuntimeError("synthetic failure")
    assert all(not m._forward_hooks for m in actor.modules())


def test_trace_comparison_preserves_row_and_channel_identity():
    a = np.zeros((2, 3), dtype=np.float32)
    b = a.copy()
    b[1, 2] = 2e-6
    comparison = helpers["compare_traces"]([("linear#0", a)], [("linear#0", b)])[0]
    assert comparison["changed_per_row"] == [0, 1]
    assert comparison["worst_row_channel"] == [1, 2]
    assert comparison["channels_over_1e_6_per_row"] == [0, 1]
    reversed_rows = helpers["error_summary"](a[::-1], b[::-1])
    assert reversed_rows["changed_per_row"] == [1, 0]


def test_mismatched_calls_shapes_and_nonfinite_fail():
    with pytest.raises(RuntimeError, match="order"):
        helpers["compare_traces"]([("a", np.zeros((1, 3)))], [("b", np.zeros((1, 3)))])
    for bad in (np.zeros((1, 2)), np.full((1, 3), np.nan), np.zeros(3)):
        with pytest.raises(ValueError):
            helpers["error_summary"](np.zeros((1, 3)), bad)


def test_registered_protocol_retains_original_gate_and_both_rows():
    p = json.loads((ROOT / "experiments/v0.0.9/numerics-protocol.json").read_text())
    assert digest(p) == helpers["PROTOCOL_SHA"]
    assert p["original_h0_action_max_error"] == 1e-6
    assert p["caps"]["physics_steps"] == 0
    assert p["variants"] == ["default_float32", "mkldnn_disabled_float32"]
