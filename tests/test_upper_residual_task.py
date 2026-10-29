import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from joxing.provenance import ROOT, digest
from joxing.shared_motor_inputs import FrozenPairTeacher
from joxing.shared_motor_task import (
    NativeReferenceDriver,
    SharedMotorTask,
    build_motor_scene,
)
from joxing.upper_residual import UpperResidualSpec
from joxing.upper_residual_task import (
    DiagnosticArmPolicy,
    UpperPairTeacher,
    UpperResidualTask,
)
from joxing.upstream_motor import NativeTask


class FixtureNormalizer:
    def check(self):
        pass

    def apply(self, values):
        return values.copy()


class FixtureActor:
    """Actual native engine, explicit synthetic actor; no released weights."""

    def __init__(self):
        self.calls = []

    def predict(self, values, *, deterministic):
        assert deterministic
        self.calls.append(values.tobytes())
        output = (-0.2 - 0.01 * np.tanh(values[85:170])[np.arange(700) % 85]).astype(
            np.float32
        )
        output[0] = np.float32(-0.0)  # A protected channel exercises signed bits.
        return output, None


@pytest.fixture(scope="module")
def inputs():
    config = json.loads((ROOT / "experiments/v0.0.6/protocol.json").read_text())[
        "environment"
    ]
    donors = {a: NativeTask(config) for a in "AB"}
    groups = json.loads(
        (ROOT / "third_party/msgym/DynSyn-SAC/configs/locomotionFull.json").read_text()
    )["agent_kwargs"]["policy_kwargs"]["dynsyn"]
    spec = UpperResidualSpec.from_native_model(
        donors["A"].model, groups, ROOT / "third_party/ms-human-700"
    )
    return build_motor_scene(), donors, spec


def make_task(inputs, *, amplitude=0.0, horizon=240, seed=93901):
    scene, donors, spec = inputs
    drivers = {a: NativeReferenceDriver(donors[a], 71000) for a in "AB"}
    actor = FixtureActor()
    teacher = FrozenPairTeacher(actor, FixtureNormalizer())
    policy = DiagnosticArmPolicy.start(seed=seed, dimension=46, amplitude=amplitude)
    task = UpperResidualTask(
        scene,
        drivers,
        teacher,
        policy,
        spec,
        horizon_ticks=horizon,
        teacher_bundle={"explicit-unit-fixture-no-weights": "0" * 64},
    )
    return task, actor


def contacts_sha(contacts):
    return digest(
        [
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in asdict(c).items()
            }
            for c in contacts
        ]
    )


def step(task):
    if not task.base.remaining:
        task.begin_control()
    contacts = task.advance_substep(lambda n: None)
    return task.base.state().tobytes(), contacts_sha(contacts)


def test_zero_arm_residual_matches_unchanged_shared_task_every_physical_interval(
    inputs,
):
    wrapped, actor = make_task(inputs, horizon=60)
    scene, donors, _ = inputs
    original_actor = FixtureActor()
    original = SharedMotorTask(
        scene,
        {a: NativeReferenceDriver(donors[a], 71000) for a in "AB"},
        FrozenPairTeacher(original_actor, FixtureNormalizer()),
        horizon_ticks=60,
        bundle_hash="explicit-reference-fixture",
    )
    assert (wrapped.base.model.nu, wrapped.base.model.na) == (1400, 1400)
    assert wrapped.base.state().tobytes() == original.state().tobytes()
    for tick in range(60):
        if not original.remaining:
            before = original.begin_control()
            after = wrapped.begin_control()
            assert all(before[a].tobytes() == after[a].tobytes() for a in "AB")
        expected = original.advance_substep(lambda n: None)
        actual = wrapped.advance_substep(lambda n: None)
        assert original.tick == wrapped.base.tick == tick + 1
        assert original.state().tobytes() == wrapped.base.state().tobytes()
        assert contacts_sha(expected) == contacts_sha(actual)
        assert {a: d.snapshot() for a, d in original.drivers.items()} == {
            a: d.snapshot() for a, d in wrapped.base.drivers.items()
        }
    assert actor.calls == original_actor.calls
    assert len(actor.calls) == 6


def test_nonzero_private_arm_policy_replays_tick203_held17_and_next_decision_exactly(
    inputs,
):
    task, actor = make_task(inputs, amplitude=0.25)
    initial = task.snapshot()
    source, saved = [], None
    for _ in range(240):
        source.append(step(task))
        if task.base.tick == 203:
            saved = task.snapshot()
    final = task.snapshot()
    assert saved["base"]["remaining"] == 17
    assert len(actor.calls) == 24
    assert final["policy"]["decisions"] == 12
    protected = list(task.spec.protected_ids)
    for record in task.records.values():
        teacher = np.frombuffer(record.teacher_bytes, np.float32)
        command = np.frombuffer(record.command_bytes, np.float32)
        assert teacher[protected].tobytes() == command[protected].tobytes()
        assert np.any(teacher != command)
    assert task.records["A"].residual_bytes != task.records["B"].residual_bytes
    calls = len(actor.calls)
    task.restore(json.loads(json.dumps(saved)))
    for i in range(203, 220):
        assert step(task) == source[i]
        assert len(actor.calls) == calls
    for i in range(220, 240):
        assert step(task) == source[i]
    assert len(actor.calls) == calls + 2
    assert task.snapshot() == final
    assert (
        task.base.physics_total == 277
    )  # Restore never rewinds aggregate entry credit.
    task.restore(initial)
    assert task.snapshot() == initial


@pytest.mark.parametrize("tick", [0, 20, 40])
def test_boundary_and_terminal_restore_keeps_historical_record_without_resampling(
    inputs, tick
):
    task, actor = make_task(inputs, horizon=40, amplitude=0.1)
    for _ in range(tick):
        step(task)
    saved = task.snapshot()
    calls = len(actor.calls)
    task.restore(json.loads(json.dumps(saved)))
    assert task.snapshot() == saved and len(actor.calls) == calls
    if tick == 40:
        with pytest.raises(RuntimeError):
            task.begin_control()


@pytest.fixture(scope="module")
def held(inputs):
    task, actor = make_task(inputs, amplitude=0.2, horizon=60)
    for _ in range(23):
        step(task)
    return task, actor, task.snapshot()


def resign(value):
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})


@pytest.mark.parametrize(
    "fault",
    [
        "bundle",
        "policy_behavior",
        "policy_rng",
        "policy_clock",
        "record_owner",
        "record_clock",
        "record_bits",
        "record_input",
        "base_command",
        "base_time",
        "base_reference",
        "remaining",
        "signed_zero",
    ],
)
def test_resigned_corruption_rejected_without_engine_policy_or_reference_mutation(
    held, fault
):
    task, actor, original = held
    task.restore(original)
    bad = copy.deepcopy(original)
    if fault == "bundle":
        bad["contract_hash"] = "0" * 64
    elif fault == "policy_behavior":
        bad["policy"]["contract"]["amplitude"] = 0.8
        resign(bad["policy"])
    elif fault == "policy_rng":
        bad["policy"]["rng"]["A"]["state"]["state"] ^= 1
        resign(bad["policy"])
    elif fault == "policy_clock":
        bad["policy"]["last_tick"] = 0
        resign(bad["policy"])
    elif fault.startswith("record_"):
        r = bad["records"]["B"]
        if fault == "record_owner":
            r["held"]["actor"] = "A"
            resign(r["held"])
        elif fault == "record_clock":
            r["held"]["sampled_tick"] = 0
            resign(r["held"])
        elif fault == "record_bits":
            r["held"]["command"][0] += 0.125
            resign(r["held"])
        else:
            r["input"]["reference_sha256"] = "f" * 64
    elif fault == "base_command":
        bad["base"]["commands"]["B"][0] += 0.125
    elif fault == "base_time":
        bad["base"]["engine"][0] += 0.001
    elif fault == "base_reference":
        bad["base"]["references"]["B"]["init_time"] += 0.02
    elif fault == "signed_zero":
        bad["base"]["commands"]["B"][0] = 0.0
    else:
        bad["base"]["remaining"] = 16
    resign(bad["base"])
    resign(bad)
    before_calls, before_physics = len(actor.calls), task.base.physics_total
    with pytest.raises(ValueError):
        task.restore(bad)
    assert task.snapshot() == original
    assert (
        len(actor.calls) == before_calls and task.base.physics_total == before_physics
    )


def test_external_assistance_and_command_drift_stop_before_any_engine_entry(held):
    task, _actor, original = held
    task.restore(original)
    before = task.base.physics_total
    task.base.data.xfrc_applied[1, 0] = 1
    with pytest.raises(ValueError, match="assistance"):
        task.advance_substep(lambda n: pytest.fail("Physics guard must not be entered"))
    task.base.data.xfrc_applied[:] = 0
    assert task.base.physics_total == before
    command = task.base.commands["B"].copy()
    task.base.commands["B"][0] += 0.1
    with pytest.raises(ValueError, match="record drift"):
        task.advance_substep(lambda n: pytest.fail("Physics guard must not be entered"))
    task.base.commands["B"] = command
    assert task.snapshot() == original


def test_pair_staging_rejects_bad_b_before_calling_a_teacher(held):
    task, actor, _ = held
    adapter = UpperPairTeacher(task.teacher, task.spec)
    actions = {a: np.zeros(46, np.float32) for a in "AB"}
    actions["B"][0] = np.nan
    before = len(actor.calls)
    with pytest.raises(ValueError):
        adapter.stage(actions, game_id=task.base.game_id, tick=20)
    assert adapter.pending is None and len(actor.calls) == before
    actions["B"][0] = 0
    adapter.stage(actions, game_id=task.base.game_id, tick=20)
    adapter.infer_pair(task.base.inputs, game_id=task.base.game_id, tick=20)
    with pytest.raises(ValueError, match="unarmed"):
        adapter.infer_pair(task.base.inputs, game_id=task.base.game_id, tick=20)
    assert len(actor.calls) == before + 2


def test_private_policy_proposals_do_not_mutate_prior_state_or_follow_global_rng(held):
    task, _, original = held
    task.restore(original)
    policy = task.policy
    saved = policy.snapshot()
    pair = {a: task.base.inputs[a] for a in "AB"}
    # This pair is the original tick20 sample; create an explicit fresh policy
    # at tick0 on an independent task instead of feeding stale future metadata.
    initial = DiagnosticArmPolicy.start(seed=93903, dimension=46, amplitude=0.25)
    from dataclasses import replace

    pair = {a: replace(p, sampled_tick=0) for a, p in pair.items()}
    state = np.random.get_state()
    try:
        np.random.seed(1)
        first, one = initial.propose(pair, game_id=task.base.game_id, tick=0)
        np.random.seed(989)
        second, two = initial.propose(pair, game_id=task.base.game_id, tick=0)
    finally:
        np.random.set_state(state)
    assert all(first[a].tobytes() == second[a].tobytes() for a in "AB")
    assert one.snapshot() == two.snapshot()
    assert initial.snapshot()["decisions"] == 0 and policy.snapshot() == saved


def test_just_sampled_boundary_record_restores_without_an_extra_pair_call(inputs):
    task, actor = make_task(inputs, amplitude=0.1, horizon=60)
    for _ in range(20):
        step(task)
    task.begin_control()
    saved = task.snapshot()
    assert saved["base"]["tick"] == 20 and saved["base"]["remaining"] == 20
    calls = len(actor.calls)
    expected = [step(task) for _ in range(20)]
    task.restore(json.loads(json.dumps(saved)))
    assert [step(task) for _ in range(20)] == expected
    assert len(actor.calls) == calls
