"""Reject changed budgets or learner settings before any physics."""

import copy

import pytest

from joxing.motor_pilot import SAC_PARAMETERS
from joxing.motor_protocol import validate
from joxing.provenance import digest


def fixture():
    return {
        "schema": "joxing-motor-pilot-D2-0.0.10",
        "training_seeds": [81011, 81029],
        "sac": copy.deepcopy(SAC_PARAMETERS),
        "transitions_per_seed": 8192,
        "evaluation_seeds": list(range(83000, 83008)),
        "caps": {"physics_steps": 550000, "wall_s": 1800, "raw_bytes": 1024**3},
        "audit_caps": {"physics_steps": 50000, "wall_s": 300},
        "raw_credits": {
            "parent_bytes": 192 * 1024**2,
            "worker_bytes": 32 * 1024**2,
            "evaluation_bytes": 12 * 1024**2,
            "replay_bytes": 12 * 1024**2,
        },
        "prior": {"physics_steps": 10394, "raw_charge_bytes": 32 * 1024**2},
    }


def test_fixed_allocation_and_sampling_contract():
    p = fixture()
    assert (
        validate(p, digest(p), source=False)["maximum_raw_credit_bytes"]
        == 888 * 1024**2
    )
    with pytest.raises(ValueError):
        validate(p, "0" * 64, source=False)
    for field, key, value in [
        ("caps", "physics_steps", 550001),
        ("sac", "gradient_steps", 2),
        ("raw_credits", "worker_bytes", 128 * 1024**2),
    ]:
        changed = copy.deepcopy(p)
        changed[field][key] = value
        with pytest.raises(ValueError):
            validate(changed, digest(changed), source=False)
