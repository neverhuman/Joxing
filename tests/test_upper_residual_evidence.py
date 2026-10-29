import json

import numpy as np
import pytest

from joxing.upper_residual import HeldUpperCommand, UpperResidualSpec, compose
from joxing.upper_residual_evidence import expected_command, inspect_snapshot


def fixture():
    spec = UpperResidualSpec(
        tuple(f"unit{i}" for i in range(700)),
        tuple(tuple(a.tolist()) for a in np.array_split(np.arange(700), 138)),
        tuple(range(100, 222)),
        (("fixture", "a" * 64),),
    )
    teacher = np.linspace(-1, 1, 700, dtype=np.float32)
    teacher[[0, 100, 221, 699]] = -0.0
    return spec, teacher


def test_scalar_oracle_and_candidate_agree_for_zero_extremes_and_distinct_groups():
    spec, teacher = fixture()
    for residual in (
        np.zeros(len(spec.active_groups), np.float32),
        np.ones(len(spec.active_groups), np.float32),
        -np.ones(len(spec.active_groups), np.float32),
        np.linspace(-1, 1, len(spec.active_groups), dtype=np.float32),
    ):
        assert (
            expected_command(teacher, residual, spec.descriptor()).tobytes()
            == compose(teacher, residual, spec).tobytes()
        )


def test_independent_record_check_and_corruption():
    spec, teacher = fixture()
    residual = np.ones(len(spec.active_groups), np.float32)
    record = HeldUpperCommand.create(
        teacher, residual, spec, actor="A", game_id="unit", sampled_tick=0
    )
    saved = json.loads(json.dumps(record.snapshot()))
    command = np.frombuffer(record.command_bytes, np.float32)
    inspect_snapshot(
        saved,
        spec.descriptor(),
        teacher,
        residual,
        command,
        actor="A",
        tick=0,
        game_id="unit",
    )
    saved["command"][0] = 0.5
    with pytest.raises(ValueError):
        inspect_snapshot(
            saved,
            spec.descriptor(),
            teacher,
            residual,
            command,
            actor="A",
            tick=0,
            game_id="unit",
        )


def test_registered_case_grid_keeps_both_actors_all_frames_and_each_arm_basis():
    from joxing.upper_residual_qualification import cases

    commands = np.zeros((176, 2, 700), np.float32)
    ticks = np.arange(0, 3520, 20, dtype=np.int64)
    rows = list(cases({"residual_seed": 93019}, commands, ticks, 46))
    keys = [r[0] for r in rows]
    assert len(rows) == len(set(keys)) == 2204
    assert sum(k.endswith("/zero") for k in keys) == 352
    assert keys[:2] == ["0/A/zero", "0/A/positive"]
    assert keys[-1] == "0/B/basis-45"
    for actor in "AB":
        basis = [r[4] for r in rows if r[1] == actor and "/basis-" in r[0]]
        assert np.array_equal(np.stack(basis), np.eye(46, dtype=np.float32))
