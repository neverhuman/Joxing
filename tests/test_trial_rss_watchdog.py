"""Bound identities, lightweight discovery and bounded real Linux growth stops."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def watchdog():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "trial_rss_watchdog", scripts / "trial_rss_watchdog.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def tree(tmp_path, entries):
    for pid, children in entries.items():
        folder = tmp_path / str(pid) / "task" / str(pid)
        folder.mkdir(parents=True)
        (folder / "children").write_text(" ".join(map(str, children)))


@pytest.mark.parametrize(
    "kind", ["children", "all_threads", "foreign", "reused", "zombie"]
)
def test_fast_tree_only_current_descendants(watchdog, tmp_path, kind):
    root = {"pid": 100, "start_ticks": 10, "parent_pid": 1, "state": "R"}
    rows = {
        100: root,
        101: {"pid": 101, "start_ticks": 11, "parent_pid": 100, "state": "R"},
        102: {"pid": 102, "start_ticks": 12, "parent_pid": 101, "state": "R"},
    }
    tree(tmp_path, {100: [101], 101: [102], 102: []})
    expected = {100, 101, 102}
    if kind == "all_threads":
        (tmp_path / "100/task/100/children").write_text("")
        thread = tmp_path / "100/task/500"
        thread.mkdir()
        (thread / "children").write_text("101")
    elif kind == "foreign":
        rows[101] = rows[101] | {"parent_pid": 77}
        expected = {100}
    elif kind == "reused":
        rows[100] = root | {"start_ticks": 999}
        expected = set()
    elif kind == "zombie":
        rows[101] = rows[101] | {"state": "Z"}
        expected = {100}
    result = watchdog.fast_tree(
        root, proc_root=tmp_path, read_identity=rows.__getitem__
    )
    assert {r["pid"] for r in result.values()} == expected


@pytest.mark.parametrize("kind", ["normal", "reused", "missing_rss", "zombie"])
def test_bound_status_fail_closed(watchdog, tmp_path, kind):
    bound = {"pid": 17, "start_ticks": 10, "state": "R"}
    current = bound.copy()
    folder = tmp_path / "17"
    folder.mkdir()
    (folder / "status").write_text("VmRSS: 123 kB\n")
    if kind == "reused":
        current["start_ticks"] = 11
    elif kind == "missing_rss":
        (folder / "status").write_text("Name: live\n")
    elif kind == "zombie":
        current["state"] = "Z"
    if kind == "missing_rss":
        with pytest.raises(RuntimeError, match="Unreadable live"):
            watchdog.bound_rss(
                bound, proc_root=tmp_path, read_identity=lambda _: current
            )
    else:
        value = watchdog.bound_rss(
            bound, proc_root=tmp_path, read_identity=lambda _: current
        )
        assert value == (123 * 1024 if kind == "normal" else None)


def test_transient_missing_status_requires_later_real_rss(
    watchdog, tmp_path, monkeypatch
):
    bound = {"pid": 17, "start_ticks": 10, "state": "R"}
    folder = tmp_path / "17"
    folder.mkdir()
    status = folder / "status"
    status.write_text("VmRSS: 123 kB\n")
    original = Path.read_text
    calls = []

    def read(path, *args, **kwargs):
        if path == status:
            calls.append(1)
            if len(calls) == 1:
                return "Name: switching\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert (
        watchdog.bound_rss(bound, proc_root=tmp_path, read_identity=lambda _: bound)
        == 123 * 1024
    )
    assert len(calls) == 2


def test_persistent_missing_status_fault_names_bound_pid(watchdog, tmp_path):
    bound = {"pid": 17, "start_ticks": 10, "state": "R"}
    folder = tmp_path / "17"
    folder.mkdir()
    (folder / "status").write_text("Name: unreadable\n")
    with pytest.raises(RuntimeError, match="pid=17 start=10 after3reads"):
        watchdog.bound_rss(bound, proc_root=tmp_path, read_identity=lambda _: bound)


@pytest.mark.parametrize("case", ["normal_exit", "grandchild_growth"])
@pytest.mark.skipif(
    sys.platform != "linux", reason="Real /proc watchdog requires Linux"
)
def test_actual_watchdog_stops_owned_growth_only(watchdog, tmp_path, case):
    report = tmp_path / "watchdog.json"
    ready = report.with_suffix(".ready")
    # Registered test-owned foreign process must survive the watchdog stop.
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(10)"], start_new_session=True
    )
    code = """
import json,sys,time,subprocess
from pathlib import Path
ready,case=Path(sys.argv[1]),sys.argv[2]
deadline=time.monotonic()+5
while not ready.exists():
 if time.monotonic()>deadline:raise RuntimeError('No watchdog readiness')
 time.sleep(.005)
if case=='normal_exit':time.sleep(.05)
else:
 child=subprocess.Popen([sys.executable,'-c','import time; blocks=[]; [(blocks.append(bytearray(16*1024**2)),time.sleep(.03)) for _ in range(12)];time.sleep(2)'])
 print(child.pid,flush=True);child.wait(timeout=5)
"""
    root = subprocess.Popen(
        [sys.executable, "-c", code, str(ready), case],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    own = watchdog.identity(os.getpid())
    bound = watchdog.identity(root.pid)
    baseline = watchdog.bound_rss(own)
    cmd = [
        sys.executable,
        str(Path(watchdog.__file__)),
        "--root-pid",
        str(root.pid),
        "--root-start",
        str(bound["start_ticks"]),
        "--supervisor-pid",
        str(own["pid"]),
        "--supervisor-start",
        str(own["start_ticks"]),
        "--output",
        str(report),
        "--planned-rss",
        str(baseline + 96 * 1024**2),
        "--hard-rss",
        str(baseline + 192 * 1024**2),
        "--process-cap",
        "8",
        "--wall-s",
        "6",
        "--max-gap-s",
        "0.1",
    ]
    guard = None
    try:
        guard = subprocess.Popen(cmd)
        guard.wait(timeout=8)
        root.wait(timeout=2)
        r = json.loads(report.read_text())
        assert r["status"] == ("exited" if case == "normal_exit" else "stopped"), r
        assert r["reason"] == (
            "all_owned_exited" if case == "normal_exit" else "planned_rss"
        ), r
        assert r["peak_sampled_rss_bytes"] <= r["limits"]["hard_rss"]
        assert r["max_sample_gap_s"] <= 0.1 and r["max_probe_s"] <= 0.1
        assert not r["final_live"] and foreign.poll() is None
        if case == "grandchild_growth":
            assert len(r["owned_identities"]) >= 2
    finally:
        for process in [root, guard, foreign]:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=2)
