import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from watchdog_publication import PhaseTiming, ReportScope, tmpfs_proof


def scope(tmp_path):
    base = tmp_path / "fake-ram"
    base.mkdir()
    return ReportScope(
        tmp_path / "watchdog.json",
        base=base,
        mount_check=lambda p: {"development_fake": str(p)},
    )


@pytest.mark.parametrize(
    "case",
    ["exact", "active", "symlink", "mode", "size", "unknown", "corrupt", "owner"],
)
def test_private_scope_ownership_bounds_and_exact_fault_copy(
    tmp_path, monkeypatch, case
):
    import watchdog_publication as module

    s = scope(tmp_path)
    raw = b'{"status":"exited","value":-0.0}\n'
    s.path.write_bytes(raw)
    s.path.with_suffix(".ready").touch()
    if case == "active":
        with pytest.raises(RuntimeError, match="active"):
            s.finish(task_live=True, guard_live=False)
        assert s.path.read_bytes() == raw and not s.canonical.exists()
        return
    if case == "symlink":
        s.path.unlink()
        s.path.symlink_to(tmp_path / "foreign")
    if case == "mode":
        s.directory.chmod(0o755)
    if case == "size":
        s.path.write_bytes(b"x" * (256 * 1024 + 1))
    if case == "unknown":
        (s.directory / "unregistered").write_text("x")
    if case == "owner":
        monkeypatch.setattr(
            module, "own_identity", lambda: {"pid": -1, "start": "foreign"}
        )
    if case in ("symlink", "mode", "size", "unknown", "owner"):
        with pytest.raises(RuntimeError):
            s._validate()
        return
    if case == "corrupt":

        def copier(source, target):
            target.write_bytes(b"corrupt")

        with pytest.raises(RuntimeError, match="mismatch"):
            s.finish(task_live=False, guard_live=False, copier=copier)
        assert s.path.read_bytes() == raw
        receipt = s.receipt.read_bytes()
        with pytest.raises(RuntimeError, match="Previous"):
            s.finish(task_live=False, guard_live=False)
        assert s.receipt.read_bytes() == receipt
    else:
        s.finish(task_live=False, guard_live=False)
        assert not s.directory.exists() and s.canonical.read_bytes() == raw
        r = json.loads(s.receipt.read_text())
        assert r["status"] == "copied_and_removed"
        assert r["copies"][0]["sha256"] == hashlib.sha256(raw).hexdigest()


def test_missing_tmpfs_rejected_before_scope_creation(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        tmpfs_proof(tmp_path / "not-a-mount")


def test_capacity_and_existing_output_reject_before_allocation(tmp_path, monkeypatch):
    import watchdog_publication as module

    base = tmp_path / "ram"
    base.mkdir()
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda p: type("D", (), {"free": 1})()
    )
    with pytest.raises(ValueError):
        ReportScope(tmp_path / "out.json", base=base, mount_check=lambda p: {})
    assert not list(base.iterdir())
    (tmp_path / "out.json").write_text("retained")
    with pytest.raises(ValueError):
        ReportScope(tmp_path / "out.json", base=base, mount_check=lambda p: {})


def test_phase_metadata_and_compact_json_preserve_float_values():
    import math

    timing = PhaseTiming()
    b = timing.begin()
    time.sleep(0.001)
    timing.end("sleep", b, requested_s=0.0001)
    assert (
        timing.data["max_sleep_excess_s"] > 0 and timing.data["recent"][0]["cpu_s"] >= 0
    )
    values = {
        "minus_zero": -0.0,
        "tiny": 1.2345678901234567e-270,
        "timing": timing.data,
    }
    decoded = json.loads(json.dumps(values, separators=(",", ":"), allow_nan=False))
    assert decoded["tiny"].hex() == values["tiny"].hex()
    assert math.copysign(1, decoded["minus_zero"]) == -1


def test_launcher_ready_path_matches_guard_output_and_default_stays_disk(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "qualify_deferred_trial", SCRIPTS / "qualify_deferred_trial.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    canonical = tmp_path / "run-watchdog.json"
    assert m.watchdog_paths(canonical) == (canonical, canonical.with_suffix(".ready"))

    class Pub:
        path = tmp_path / "private-ram/report.json"

    assert m.watchdog_paths(canonical, Pub()) == (
        Pub.path,
        Pub.path.with_suffix(".ready"),
    )
    assert m.runner_module("joxing-watch-publication-v1") == "joxing.process_exit_probe"
    assert (
        m.runner_module("joxing-process-exit-probe-v1") == "joxing.process_exit_probe"
    )


@pytest.mark.parametrize("backend", ["disk", "private-tmpfs-v1"])
@pytest.mark.skipif(
    sys.platform != "linux", reason="Actual Linux report IPC and owned monitor"
)
def test_injected_disk_delay_cannot_block_ram_sampler(tmp_path, backend):
    from supervise_motor_pilot import identity
    from trial_rss_watchdog import bound_rss

    canonical = tmp_path / "watchdog.json"
    pub = ReportScope(canonical) if backend != "disk" else None
    report = pub.path if pub else canonical
    ready = report.with_suffix(".ready")
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
    )
    root = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from pathlib import Path;import sys,time;r=Path(sys.argv[1]);deadline=time.monotonic()+5\nwhile not r.exists():\n if time.monotonic()>deadline:raise RuntimeError('No readiness')\n time.sleep(.005)\nx=bytearray(16*1024**2);time.sleep(.65)",
            str(ready),
        ],
        start_new_session=True,
    )
    own, bound = identity(os.getpid()), identity(root.pid)
    baseline = bound_rss(own)
    code = r"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import trial_rss_watchdog as w
c=json.loads(sys.argv[2]);original=Path.write_text;calls=[0]
def write(path,*a,**kw):
 if c['backend']=='disk' and path==Path(c['output']).with_suffix('.tmp'):
  calls[0]+=1
  if calls[0]==2:time.sleep(.15)
 return original(path,*a,**kw)
Path.write_text=write
r=w.monitor(c['root'],c['supervisor'],Path(c['output']),planned_rss=c['baseline']+64*1024**2,hard_rss=c['baseline']+128*1024**2,process_cap=8,wall_s=5,rss_reader='terminal-mm-reserve-v3',phase_timing=True)
assert r['status']==('stopped' if c['backend']=='disk' else 'exited'),r
"""
    settings = {
        "root": bound,
        "supervisor": own,
        "baseline": baseline,
        "output": str(report),
        "backend": backend,
    }
    guard = None
    try:
        guard = subprocess.Popen(
            [sys.executable, "-c", code, str(SCRIPTS), json.dumps(settings)]
        )
        guard.wait(timeout=7)
        root.wait(timeout=2)
        assert guard.returncode == 0
        if pub:

            def delayed_copy(source, target):
                assert root.poll() is not None and guard.poll() is not None
                time.sleep(0.15)
                target.write_bytes(source.read_bytes())

            pub.finish(task_live=False, guard_live=False, copier=delayed_copy)
            r = json.loads(pub.receipt.read_text())
            assert (
                r["copy_end_s"] - r["copy_begin_s"] >= 0.15
                and not pub.directory.exists()
            )
        w = json.loads(canonical.read_text())
        assert (
            w["peak_capacity_charge_bytes"] <= w["limits"]["hard_rss"]
            and not w["final_live"]
        )
        if backend == "disk":
            assert w["reason"] == "stale_monitor" and w["max_sample_gap_s"] > 0.1
            assert w["phase_timing"]["max_wall_s"]["report_write"] >= 0.15
        else:
            assert max(w["max_sample_gap_s"], w["max_probe_s"]) <= 0.1
            assert w["status"] == "exited" and not w["pending_retirements"]
        assert foreign.poll() is None
        (tmp_path / "foreign.json").write_text(
            json.dumps({"identity": identity(foreign.pid), "survived_guard": True})
            + "\n"
        )
        (tmp_path / "publication-control.json").write_text(
            json.dumps(
                {
                    "backend": backend,
                    "expected_status": "stopped" if backend == "disk" else "exited",
                    "declared_disk_delay_s": 0.15,
                    "actual_status": w["status"],
                    "max_gap_s": w["max_sample_gap_s"],
                    "physics": 0,
                }
            )
            + "\n"
        )
    finally:
        for p in (root, guard, foreign):
            if p is not None and p.poll() is None:
                p.kill()
                p.wait(timeout=2)
