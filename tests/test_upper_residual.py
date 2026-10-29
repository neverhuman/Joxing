import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from joxing import upper_residual as upper
from joxing.motor_residual import AMPLITUDE, expand_groups
from joxing.provenance import ROOT


@pytest.fixture
def spec():
    return upper.UpperResidualSpec(
        tuple(f"fixture_muscle_{i}" for i in range(700)),
        tuple(tuple(g.tolist()) for g in np.array_split(np.arange(700), 138)),
        tuple(range(100, 222)),
        (("unit-fixture", "0" * 64),),
    )


@pytest.fixture
def teacher():
    a = np.linspace(-1, 1, 700, dtype=np.float32)
    a[[0, 100, 221, 699]] = np.float32(-0.0)
    a[3] = np.nextafter(np.float32(0), np.float32(1))
    return a


def test_zero_residual_preserves_all700_bits_and_does_not_alias(spec, teacher):
    for sign in (0.0, -0.0):
        result = upper.compose(
            teacher, np.full(len(spec.active_groups), sign, np.float32), spec
        )
        assert result.shape == (700,) and result.dtype == np.float32
        assert result.tobytes() == teacher.tobytes()
        result[0] = 0.5
        assert np.signbit(teacher[0])


def test_nonzero_residual_changes_only_selected_muscles_even_for_mixed_groups(
    spec, teacher
):
    assert spec.mixed_groups  # Fixture deliberately crosses an arm-mask boundary.
    residual = np.ones(len(spec.active_groups), np.float32)
    before = teacher.tobytes()
    result = upper.compose(teacher, residual, spec)
    assert teacher.tobytes() == before
    assert (
        result[list(spec.protected_ids)].tobytes()
        == teacher[list(spec.protected_ids)].tobytes()
    )
    assert np.max(abs(result - teacher)) <= np.float32(AMPLITUDE) + 6e-8
    assert np.any(result[list(spec.selected_ids)] != teacher[list(spec.selected_ids)])


def test_compact_scatter_matches_existing138_group_expansion(spec, teacher):
    residual = np.linspace(-1, 1, len(spec.active_groups), dtype=np.float32)
    latent = np.zeros(138, np.float32)
    latent[list(spec.active_groups)] = residual
    correction = expand_groups(latent, spec.groups)
    expected = teacher.copy()
    ids = list(spec.selected_ids)
    expected[ids] = np.clip(
        teacher[ids] + np.float32(AMPLITUDE) * correction[ids], -1, 1
    )
    # Compare to the legacy expansion on selected channels; protect other bits.
    assert upper.compose(teacher, residual, spec).tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "kind",
    [
        "teacher_dtype",
        "teacher_nan",
        "teacher_bound",
        "residual_dtype",
        "residual_shape",
        "residual_inf",
    ],
)
def test_invalid_or_ambiguous_numeric_inputs_rejected(spec, teacher, kind):
    residual = np.zeros(len(spec.active_groups), np.float32)
    if kind == "teacher_dtype":
        teacher = teacher.astype(np.float64)
    elif kind == "teacher_nan":
        teacher[0] = np.nan
    elif kind == "teacher_bound":
        teacher[0] = 1.01
    elif kind == "residual_dtype":
        residual = residual.astype(np.float64)
    elif kind == "residual_shape":
        residual = np.zeros(138, np.float32)
    else:
        residual[0] = np.inf
    with pytest.raises(ValueError, match="float32"):
        upper.compose(teacher, residual, spec)


def test_spec_owns_its_inputs_and_detects_forced_drift(spec, teacher):
    descriptor = spec.descriptor()
    descriptor["selected_ids"].clear()
    spec.check()
    object.__setattr__(spec, "selected_ids", tuple(range(99, 222)))
    with pytest.raises(ValueError, match="drift"):
        upper.compose(teacher, np.zeros(len(spec.active_groups), np.float32), spec)


def test_held_record_is_immutable_and_does_not_redecode_during_hold(
    spec, teacher, monkeypatch
):
    residual = np.zeros(len(spec.active_groups), np.float32)
    record = upper.HeldUpperCommand.create(
        teacher, residual, spec, actor="A", game_id="fixture", sampled_tick=20
    )
    original = teacher.tobytes()
    teacher[:] = 1

    def no_redecode(*args, **kwargs):
        pytest.fail("Held command re-decoded or called a policy")

    monkeypatch.setattr(upper, "compose", no_redecode)
    for tick in (20, 23, 39):
        held = record.command_at(actor="A", game_id="fixture", engine_tick=tick)
        assert held.tobytes() == original
        with pytest.raises(ValueError):
            held.flags.writeable = True


@pytest.mark.parametrize(
    "owner,game,tick",
    [
        ("B", "fixture", 23),
        ("A", "wrong", 23),
        ("A", "fixture", 19),
        ("A", "fixture", 40),
        ("A", "fixture", True),
    ],
)
def test_held_command_rejects_wrong_owner_game_or_time(
    spec, teacher, owner, game, tick
):
    record = upper.HeldUpperCommand.create(
        teacher,
        np.zeros(len(spec.active_groups), np.float32),
        spec,
        actor="A",
        game_id="fixture",
        sampled_tick=20,
    )
    with pytest.raises(ValueError, match="ownership/interval"):
        record.command_at(actor=owner, game_id=game, engine_tick=tick)


def test_snapshot_roundtrip_at_mid_hold_is_exact(spec, teacher):
    residual = np.linspace(-1, 1, len(spec.active_groups), dtype=np.float32)
    record = upper.HeldUpperCommand.create(
        teacher, residual, spec, actor="B", game_id="fixture", sampled_tick=0
    )
    saved = json.loads(json.dumps(record.snapshot()))
    restored = upper.HeldUpperCommand.restore(
        saved, spec, actor="B", game_id="fixture", sampled_tick=0
    )
    assert restored.snapshot() == saved
    assert (
        restored.command_at(actor="B", game_id="fixture", engine_tick=3).tobytes()
        == record.command_bytes
    )


@pytest.mark.parametrize("change", ["clock", "precision", "command", "spec"])
def test_snapshot_corruption_rejected_after_resigning(spec, teacher, change):
    record = upper.HeldUpperCommand.create(
        teacher,
        np.zeros(len(spec.active_groups), np.float32),
        spec,
        actor="A",
        game_id="fixture",
        sampled_tick=0,
    )
    saved = copy.deepcopy(record.snapshot())
    if change == "clock":
        saved["physics_dt_s"] = 0.002
    elif change == "precision":
        saved["teacher"][1] = 0.123456789123
    elif change == "spec":
        saved["spec_sha256"] = "f" * 64
    else:
        saved["command"][100] = 0.5
        saved["command_sha256"] = upper.array_sha(
            np.array(saved["command"], np.float32)
        )
    saved["sha256"] = upper.digest({k: v for k, v in saved.items() if k != "sha256"})
    with pytest.raises(ValueError):
        upper.HeldUpperCommand.restore(
            saved, spec, actor="A", game_id="fixture", sampled_tick=0
        )


def test_pair_composition_validates_both_without_mutating_teacher_commands(
    spec, teacher
):
    teachers = {"A": teacher, "B": -teacher}
    before = {a: x.tobytes() for a, x in teachers.items()}
    residuals = {a: np.zeros(len(spec.active_groups), np.float32) for a in "AB"}
    pair = upper.compose_pair(
        teachers, residuals, spec, game_id="paired", sampled_tick=0
    )
    assert {a: r.command_bytes for a, r in pair.items()} == before
    residuals["B"] = np.zeros(1, np.float32)
    with pytest.raises(ValueError):
        upper.compose_pair(teachers, residuals, spec, game_id="paired", sampled_tick=0)
    assert {a: x.tobytes() for a, x in teachers.items()} == before


def test_real_native_mask_is122_arms_578_protected_and46_compact_groups():
    import mujoco

    source = ROOT / "third_party/ms-human-700"
    model = mujoco.MjModel.from_xml_path(str(source / "MS-Human-700.xml"))
    cfg = json.loads(
        (ROOT / "third_party/msgym/DynSyn-SAC/configs/locomotionFull.json").read_text()
    )
    spec = upper.UpperResidualSpec.from_native_model(
        model, cfg["agent_kwargs"]["policy_kwargs"]["dynsyn"], source
    )
    assert len(spec.selected_ids) == 122 and len(spec.protected_ids) == 578
    assert len(spec.active_groups) == 46 and not spec.mixed_groups
    assert spec.selected_ids == tuple(range(100, 222))
    assert model.nu == model.na == 700  # No actuator is removed by a command mask.


def test_incompatible_manipulation_embodiment_cannot_be_used_as_native_teacher():
    with pytest.raises(ValueError, match="intact native700"):
        upper.UpperResidualSpec.from_native_model(
            SimpleNamespace(nq=12, nv=12, nu=81, na=81), (), "unused"
        )


def test_changed_mask_cannot_restore_an_old_held_record(spec, teacher):
    record = upper.HeldUpperCommand.create(
        teacher,
        np.zeros(len(spec.active_groups), np.float32),
        spec,
        actor="A",
        game_id="fixture",
        sampled_tick=0,
    )
    changed = replace(spec, selected_ids=tuple(range(99, 222)))
    with pytest.raises(ValueError, match="identity"):
        upper.HeldUpperCommand.restore(
            record.snapshot(), changed, actor="A", game_id="fixture", sampled_tick=0
        )
