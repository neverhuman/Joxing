import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from joxing import deferred_qualification as dq


def test_exact_array_hash_and_no_rng_consumption():
    state = np.random.get_state()
    a = np.array([1.0, -0.0], dtype=np.float64)
    assert dq.value_digest(a) == dq.value_digest(a.copy())
    assert dq.value_digest(a) != dq.value_digest(a.astype(np.float32))
    assert dq.value_digest(a) != dq.value_digest(a.reshape(1, 2))
    assert dq.value_digest(a) != dq.value_digest(np.array([1.0, 0.0]))
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    with pytest.raises(ValueError):
        dq.value_digest(np.array([np.nan]))
    with pytest.raises(ValueError):
        dq.value_digest(float("inf"))


def test_only_monitor_wall_time_excluded():
    a = {"episode": {"t": 1.0, "r": 2.0, "l": 3}, "elapsed_sim_s": 4.0}
    assert dq.semantic_info(a) == {"episode": {"r": 2.0, "l": 3}, "elapsed_sim_s": 4.0}
    assert "t" in a["episode"]


def test_static_worst_case_includes_first_vector_reset():
    cfg = {"dt_s": 0.001, "control_s": 0.01, "episode_s": 20.0, "preparation_s": 0.2}
    bound = dq.physics_upper_bound(cfg)
    assert bound == 221760 and bound < 240000


def test_stdlib_import_before_readiness_and_timeout():
    code = "import sys; from joxing.deferred_qualification import wait_ready; assert not {'numpy','torch','mujoco'}&sys.modules.keys(); wait_ready('/nonexistent-joxing-ready',.02)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0 and "No qualified watcher readiness" in result.stderr


def test_fresh_spawn_installs_native_hooks_before_constructor(tmp_path):
    code = """
import contextlib,sys
from pathlib import Path
from types import SimpleNamespace
from joxing.deferred_qualification import install_physics_hooks
from joxing.physics_credit import PhysicsCredit
from joxing.balance import BalanceModel
from joxing.stance_env import StanceEnv
path=Path(sys.argv[1]);PhysicsCredit.create(path/'credit.json','fixed',50)
calls=[]
original=BalanceModel.prepare
BalanceModel.prepare=lambda self,*a,**kw:calls.append('original') or ('data','audit')
try:
 with contextlib.ExitStack() as stack:
  install_physics_hooks(stack,path,path/'credit.json','fixed','A','native-fixture')
  rig=SimpleNamespace(config=SimpleNamespace(preparation_s=.02,dt_s=.001))
  assert BalanceModel.prepare(rig)==('data','audit')
  assert PhysicsCredit(path/'credit.json','fixed').snapshot()['used']==20
 assert calls==['original']
finally:BalanceModel.prepare=original
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, timeout=15)


def complete_fixture(folder):
    folder.mkdir()
    (folder / "traces").mkdir()
    summary = {
        "training_transitions": 64,
        "updates": 12,
        "scheduled_training_complete": True,
        "source_unchanged": True,
        "initial_actor_hash": "initial",
        "final_actor_hash": "final",
    }
    (folder / "semantic-result.json").write_text(json.dumps(summary))
    learner = [{"phase": "model_initial"}, {"phase": "learn_before"}]
    for i in range(12):
        learner += [
            {"phase": "update_before", "updates": i},
            {"phase": "update_after", "updates": i + 1},
        ]
    learner += [{"phase": "learn_after"}] + [
        {"phase": p}
        for _ in range(2)
        for p in ("evaluation_before", "evaluation_after")
    ]
    rows = {
        "learner": learner,
        "native-evaluation": [{"phase": "reset"}] * 3 + [{"phase": "step"}] * 2,
    }
    for i in range(4):
        rows[f"worker-{i}"] = (
            [{"phase": "reset"}] + [{"phase": "step"}] * 16 + [{"phase": "close"}]
        )
        rows[f"native-world-{i}"] = [{"phase": "reset"}] * 2 + [{"phase": "step"}] * 16
    for name, values in rows.items():
        (folder / "traces" / (name + ".jsonl")).write_text(
            "".join(json.dumps({"semantic": v}) + "\n" for v in values)
        )


@pytest.mark.parametrize(
    "defect", ["none", "missing", "empty", "changed", "missing_update"]
)
def test_comparison_requires_complete_exact_histories(tmp_path, defect):
    folders = {
        name: str(tmp_path / name)
        for name in ("legacyA", "candidateA", "legacyB", "candidateB")
    }
    for folder in folders.values():
        complete_fixture(Path(folder))
    target = Path(folders["candidateA"]) / "traces/learner.jsonl"
    if defect == "missing":
        target.unlink()
    elif defect == "empty":
        target.write_text("")
    elif defect == "changed":
        target.write_text(target.read_text().replace("model_initial", "wrong"))
    elif defect == "missing_update":
        for folder in ("candidateA", "legacyA"):
            p = Path(folders[folder]) / "traces/learner.jsonl"
            p.write_text("\n".join(p.read_text().splitlines()[1:]) + "\n")
    if defect == "none":
        assert dq.compare_slots(folders)["qualified"]
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            dq.compare_slots(folders)
