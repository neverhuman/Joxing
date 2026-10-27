import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def helper():
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "joxing_process_ownership", root / "scripts/joxing_process_ownership.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "kind",
    [
        "existing",
        "new_child",
        "new_grandchild",
        "foreign",
        "orphan",
        "reused_root",
        "vanished",
        "cycle",
    ],
)
def test_ancestry_ownership_handles_new_children_without_inventory_race(helper, kind):
    root = {"pid": 100, "start_ticks": 1000, "parent_pid": 1}
    rows = {100: root, 101: {"pid": 101, "start_ticks": 1001, "parent_pid": 100}}
    candidate = 101
    expected = True
    if kind == "new_child":
        rows[102] = {"pid": 102, "start_ticks": 1002, "parent_pid": 101}
        candidate = 102
    elif kind == "new_grandchild":
        rows[102] = {"pid": 102, "start_ticks": 1002, "parent_pid": 101}
        rows[103] = {"pid": 103, "start_ticks": 1003, "parent_pid": 102}
        candidate = 103
    elif kind == "foreign":
        rows[101]["parent_pid"] = 200
        rows[200] = {"pid": 200, "start_ticks": 2, "parent_pid": 1}
        expected = False
    elif kind == "orphan":
        rows[101]["parent_pid"] = 1
        expected = False
    elif kind == "reused_root":
        rows[100] = root | {"start_ticks": 9999}
        expected = False
    elif kind == "vanished":
        del rows[100]
        expected = None
    elif kind == "cycle":
        rows[101]["parent_pid"] = 101
        expected = False

    def read(pid):
        if pid not in rows:
            raise ProcessLookupError(pid)
        return rows[pid]

    assert helper.belongs_to_supervisor(candidate, root, read_identity=read) is expected


@pytest.mark.skipif(
    sys.platform != "linux", reason="Real /proc ancestry requires source host"
)
def test_real_new_child_and_grandchild_are_owned_after_initial_inventory(helper):
    supervisor = helper.identity(os.getpid())
    previous_inventory = {os.getpid()}
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)']); print(p.pid,flush=True); time.sleep(5)"
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        grandchild = int(child.stdout.readline().strip())
        assert (
            child.pid not in previous_inventory and grandchild not in previous_inventory
        )
        assert helper.belongs_to_supervisor(child.pid, supervisor)
        assert helper.belongs_to_supervisor(grandchild, supervisor)
        assert not helper.belongs_to_supervisor(
            grandchild, supervisor | {"start_ticks": supervisor["start_ticks"] + 1}
        )
    finally:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)
