"""Cross-host replay requires every contact row and an unchanged registered gate."""

import json
import runpy

import numpy as np
import pytest

from joxing.provenance import ROOT, digest

helpers = runpy.run_path(str(ROOT / "scripts/qualify_scaling_host.py"))


@pytest.mark.parametrize("mutate", [False, True])
def test_contact_tail_rebases_offsets_and_detects_tiny_force_change(tmp_path, mutate):
    expected = {
        "intervals_s": np.arange(6).reshape(3, 2),
        "loads_N": np.ones((3, 3)),
        "tilt_drop": np.zeros((3, 2)),
        "contact_offsets": np.array([0, 2, 2, 3]),
        "contact_geom_ids": np.arange(6).reshape(3, 2),
        "contact_role": np.arange(3),
        "contact_force_N": np.arange(9.0).reshape(3, 3),
        "contact_position_m": np.arange(9.0).reshape(3, 3),
    }
    actual = {
        k: v[1:].copy()
        if k in {"intervals_s", "loads_N", "tilt_drop"}
        else v[2:].copy()
        for k, v in expected.items()
    }
    actual["contact_offsets"] = np.array([0, 0, 1])
    if mutate:
        actual["contact_force_N"][0, 0] += 1e-9
    np.savez(tmp_path / "expected.npz", **expected)
    np.savez(tmp_path / "actual.npz", **actual)
    if mutate:
        with pytest.raises(RuntimeError, match="contact_force"):
            helpers["compare_contact_tail"](
                tmp_path / "expected.npz", tmp_path / "actual.npz", 1
            )
    else:
        assert helpers["compare_contact_tail"](
            tmp_path / "expected.npz", tmp_path / "actual.npz", 1
        )


def test_registered_new_attempt_preserves_original_scaling_and_fits_reserve():
    p = json.loads((ROOT / "experiments/v0.0.9/scaling-attempt-02.json").read_text())
    assert digest(p) == helpers["PROTOCOL_SHA"]
    original = json.loads(
        (ROOT / "experiments/v0.0.9/scaling-protocol.json").read_text()
    )
    assert digest(original) == p["scaling_protocol_sha256"]
    assert (
        p["scaling_caps_unchanged"]["reserve_bytes_per_worker"]
        == original["worker_memory_reservation_bytes"]
    )
    budget = p["mission_accounting"]
    assert (
        budget["previous_physics_steps"] + budget["conservative_additional_physics_cap"]
        < budget["full_H1_H2_H3_physics_cap"]
    )
    assert (
        budget["previous_formal_wall_s"] + budget["maximum_additional_formal_wall_s"]
        < budget["full_H1_H2_H3_wall_s"]
    )
