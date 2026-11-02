import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def helper():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "terminal_rss_reader", scripts / "terminal_rss_reader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proc(root, *, flags=4, start=10, statm="0 0 0 0 0 0 0", extra=False):
    folder = root / "17"
    task = folder / "task/17"
    task.mkdir(parents=True)
    fields = ["R", "2", "0", "0", "0", "0", str(flags)] + ["0"] * 12 + [str(start)]
    for path in [folder / "stat", task / "stat"]:
        path.write_text("17 (fake) " + " ".join(fields))
    (folder / "status").write_text("Threads: 1\n")
    (folder / "statm").write_text(statm)
    if extra:
        (folder / "task/99").mkdir()
    return {"pid": 17, "start_ticks": 10}


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "live_flag",
        "reused",
        "nonzero",
        "malformed",
        "mixed_threads",
        "missing_size_field",
        "missing_statm",
    ],
)
def test_terminal_evidence_requires_complete_bound_single_thread(
    helper, tmp_path, case
):
    bound = proc(
        tmp_path,
        flags=0 if case == "live_flag" else 4,
        start=11 if case == "reused" else 10,
        statm="0 1 0 0 0 0 0"
        if case == "nonzero"
        else "0 0"
        if case == "malformed"
        else "0 0 0 0 0 0 0",
        extra=case == "mixed_threads",
    )
    if case == "missing_size_field":
        (tmp_path / "17/status").write_text("Threads: 1\nVmSize: 12 kB\n")
    if case == "missing_statm":
        (tmp_path / "17/statm").unlink()
    if case == "valid":
        assert helper.terminal_evidence(bound, tmp_path)["statm_pages"] == [0] * 7
    else:
        with pytest.raises((RuntimeError, FileNotFoundError)):
            helper.terminal_evidence(bound, tmp_path)


def test_nonzero_charge_held_and_released_without_zero_substitution(helper):
    bound = {"pid": 17, "start_ticks": 10}
    clock = [1.0]
    readings = iter([123, None])
    state = {"retire": False}

    def read(_):
        if state["retire"]:
            raise RuntimeError("missing")
        return next(readings)

    reader = helper.TerminalReader(
        read, evidence=lambda _: {"after": {"state": "R"}}, clock=lambda: clock[0]
    )
    assert reader.sample(bound)["rss_bytes"] == 123
    state["retire"] = True
    held = reader.sample(bound)
    assert (
        held["rss_bytes"] is None
        and held["capacity_charge_bytes"] == held["retiring_rss_charge_bytes"] == 123
    )
    assert helper.aggregate(
        [held, {"rss_bytes": 400, "retiring_rss_charge_bytes": 0}]
    ) == (400, 123, 523)
    state["retire"] = False
    clock[0] += 0.05
    assert reader.sample(bound) is None and not reader.retiring
    assert reader.records["17:10"]["released"]


def test_missing_live_no_baseline_and_expired_hold_fail(helper):
    bound = {"pid": 17, "start_ticks": 10}
    clock = [0.0]

    def missing(_):
        raise RuntimeError("missing")

    reader = helper.TerminalReader(
        missing, evidence=lambda _: {"after": {"state": "R"}}, clock=lambda: clock[0]
    )
    with pytest.raises(RuntimeError, match="prior observation"):
        reader.sample(bound)
    reader.highwater[(17, 10)] = 100
    reader.sample(bound)
    clock[0] = 0.101
    with pytest.raises(RuntimeError, match="hold first limit"):
        reader.sample(bound)


@pytest.mark.parametrize("reader", ["strict-v2", "terminal-mm-reserve-v3"])
def test_live_missing_memory_still_faults_in_both_modes(helper, reader, tmp_path):
    bound = proc(tmp_path, flags=0)

    def missing(_):
        raise RuntimeError("live missing")

    if reader == "strict-v2":
        with pytest.raises(RuntimeError):
            missing(bound)
    else:
        r = helper.TerminalReader(
            missing, evidence=lambda b: helper.terminal_evidence(b, tmp_path)
        )
        r.highwater[(17, 10)] = 123
        with pytest.raises(RuntimeError):
            r.sample(bound)


@pytest.mark.skipif(
    sys.platform != "linux", reason="Actual Linux heavy exiting address space"
)
def test_actual_exit_uses_nonzero_retirement_charge(helper, tmp_path):
    from supervise_motor_pilot import identity
    from trial_rss_watchdog import bound_rss

    scripts = Path(helper.__file__).parent
    report = tmp_path / "watchdog.json"
    ready = report.with_suffix(".ready")
    release = tmp_path / "release"
    code = "import os,sys,time;from pathlib import Path;r=Path(sys.argv[1]);x=Path(sys.argv[2]);\nwhile not r.exists():time.sleep(.005)\na=bytearray(512*1024**2);x.write_text('allocated');time.sleep(.1);os._exit(0)"
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(ready), str(release)], start_new_session=True
    )
    bound = identity(child.pid)
    parent = identity(os.getpid())
    baseline = bound_rss(parent)
    cmd = [
        sys.executable,
        str(scripts / "trial_rss_watchdog.py"),
        "--root-pid",
        str(child.pid),
        "--root-start",
        str(bound["start_ticks"]),
        "--supervisor-pid",
        str(parent["pid"]),
        "--supervisor-start",
        str(parent["start_ticks"]),
        "--output",
        str(report),
        "--planned-rss",
        str(baseline + 768 * 1024**2),
        "--hard-rss",
        str(baseline + 1024**3),
        "--process-cap",
        "8",
        "--wall-s",
        "5",
        "--rss-reader",
        "terminal-mm-reserve-v3",
    ]
    watcher = None
    try:
        watcher = subprocess.Popen(cmd)
        watcher.wait(timeout=7)
        child.wait(timeout=2)
        data = json.loads(report.read_text())
        assert watcher.returncode == child.returncode == 0, data
        assert (
            data["status"] == "exited"
            and not data["pending_retirements"]
            and not data["final_live"]
        )
        assert data["peak_capacity_charge_bytes"] <= data["limits"]["hard_rss"]
        assert max(data["max_probe_s"], data["max_sample_gap_s"]) <= 0.1
        assert data["retirement_records"], (
            "Retirement path was not observed; do not qualify it"
        )
        assert all(
            r["highwater_charge_bytes"] > 0 and r["released"] and r["max_hold_s"] <= 0.1
            for r in data["retirement_records"].values()
        )
    finally:
        for p in (child, watcher):
            if p is not None and p.poll() is None:
                p.kill()
                p.wait(timeout=2)
