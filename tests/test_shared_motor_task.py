import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from joxing.provenance import digest
from joxing.shared_motor_task import (
    NativeReferenceDriver,
    SharedMotorTask,
    build_motor_scene,
)
from joxing.upstream_motor import NativeTask


class FixtureTeacher:
    """Deterministic fixture only; these checks never load trained weights."""

    def __init__(self):
        self.calls = []

    def infer_pair(self, pair, *, game_id, tick):
        self.calls.append((tick, {p: v.values.tobytes() for p, v in pair.items()}))
        return {
            p: (-0.2 - 0.01 * np.tanh(v.values[85:170])[np.arange(700) % 85]).astype(
                np.float32
            )
            for p, v in pair.items()
        }


@pytest.fixture(scope="module")
def task():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "experiments/v0.0.6/protocol.json").read_text())[
        "environment"
    ]
    scene = build_motor_scene()
    drivers = {p: NativeReferenceDriver(NativeTask(config), 71000) for p in ["A", "B"]}
    teacher = FixtureTeacher()
    return SharedMotorTask(
        scene,
        drivers,
        teacher,
        horizon_ticks=60,
        bundle_hash="fixture-no-trained-weights",
    )


def collect(task, checkpoint=None):
    hashes, contacts, commands = [], [], []
    saved = None
    while not task.done:
        if not task.remaining:
            commands.append(task.begin_control())
        row = task.advance_substep(lambda n: None)
        hashes.append(hashlib.sha256(task.state().tobytes()).hexdigest())
        contacts.append(
            [
                {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in asdict(c).items()
                }
                for c in row
            ]
        )
        if task.tick == checkpoint:
            saved = task.snapshot()
    return hashes, contacts, commands, saved


def test_parallel_variant_preserves_complete_muscles_and_reference_reset(task):
    assert (task.model.nq, task.model.nv, task.model.nu, task.model.na) == (
        170,
        170,
        1400,
        1400,
    )
    assert not np.any(task.data.ctrl) and not np.any(task.data.act)
    assert task.tick == task.physics_total == 0 and task.commands is None
    for p, mapping in task.maps.items():
        assert np.array_equal(
            task.data.qpos[mapping.qpos_ids], task.drivers[p].initial["qpos"]
        )
        assert task.drivers[p].task.physics_total == 0
        assert task.inputs[p].values.shape == (3601,)
    a, b = task.model.body("A_pelvis").id, task.model.body("B_pelvis").id
    assert np.allclose(
        task.data.xpos[b] - task.data.xpos[a], [0, 2.5, 0], rtol=0, atol=1e-12
    )
    assert not set(task.maps["A"].actuator_ids) & set(task.maps["B"].actuator_ids)


def test_real_shared_engine_mid_hold_replays_inputs_commands_contacts_without_new_decision(
    task,
):
    initial = task.snapshot()
    hashes, contacts, commands, saved = collect(task, checkpoint=23)
    assert task.physics_total == 60 and len(task.teacher.calls) == 3
    assert saved["remaining"] == 17 and saved["references"]["A"]["tick"] == 20
    final = task.snapshot()
    before_calls = len(task.teacher.calls)
    task.restore(json.loads(json.dumps(saved)))
    while task.remaining:
        task.advance_substep(lambda n: None)
    assert len(task.teacher.calls) == before_calls  # held17steps, no actor call
    h, c, cmds, _ = collect(task)
    assert h == hashes[40:] and c == contacts[40:]
    assert all(np.array_equal(cmds[0][p], commands[2][p]) for p in ["A", "B"])
    assert task.snapshot() == final and task.physics_total == 97
    assert len(task.teacher.calls) == before_calls + 1
    with pytest.raises(RuntimeError):
        task.begin_control()
    task.restore(initial)


@pytest.mark.parametrize(
    "fault",
    [
        "time",
        "remaining",
        "ending",
        "engine",
        "reference",
        "input",
        "command",
        "bundle",
    ],
)
def test_motor_snapshot_corruption_is_atomic(task, fault):
    original = task.snapshot()
    bad = copy.deepcopy(original)
    if fault == "time":
        bad["tick"] = 1
    elif fault == "remaining":
        bad["remaining"] = 3
    elif fault == "ending":
        bad["done"] = True
    elif fault == "engine":
        bad["engine"][0] = 0.02
    elif fault == "reference":
        bad["references"]["A"]["init_time"] += 0.02
    elif fault == "input":
        bad["inputs"]["B"]["values"][3073] += 1
    elif fault == "command":
        bad["commands"] = {"A": [0.0] * 700, "B": [0.0] * 700}
    else:
        bad["contract_hash"] = "invented"
    bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError):
        task.restore(bad)
    assert task.snapshot() == original


def test_motor_control_is_staged_and_external_assistance_fails_before_engine(task):
    task.data.xfrc_applied[1, 0] = 1
    before = task.physics_total
    with pytest.raises(ValueError, match="assistance"):
        task.begin_control()
    task.data.xfrc_applied[:] = 0
    assert task.physics_total == before
