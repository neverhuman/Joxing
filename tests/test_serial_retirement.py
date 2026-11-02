import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from joxing.serial_retirement import serial_vec_class


def test_stdlib_adapter_and_runner_until_ready():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;import joxing.serial_native_teardown;"
                "assert not {'numpy','torch','mujoco'}&sys.modules.keys()"
            ),
        ],
        check=True,
        timeout=5,
    )


@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "waiting",
        "repeat",
        "missing",
        "reused",
        "dead",
        "pipe",
        "join_timeout",
        "nonzero",
        "present_after_join",
        "pending_timeout",
        "pending_error",
        "counts",
        "duplicate",
        "overall_timeout",
    ],
)
def test_serial_close_order_and_fail_closed(tmp_path, case):
    calls, processes, clock = [], [], [0.0]

    class Process:
        def __init__(self, i):
            self.i, self.pid, self.running, self.exitcode = i, 100 + i, True, None

        def is_alive(self):
            return self.running

        def join(self, timeout):
            assert 0 < timeout <= 2
            calls.append(("join", self.i))
            if case == "join_timeout":
                return
            self.running = False
            self.exitcode = 3 if case == "nonzero" else 0
            clock[0] += 3.0 if case == "overall_timeout" else 0.02

    class Remote:
        def __init__(self, i):
            self.i = i

        def send(self, msg):
            assert msg == ("close", None)
            assert all(not p.running for p in processes[: self.i])
            calls.append(("send", self.i))
            if case == "pipe":
                raise BrokenPipeError("fixture")

        def poll(self, timeout):
            assert timeout > 0
            return case != "pending_timeout"

        def recv(self):
            calls.append(("recv", self.i))
            if case == "pending_error":
                raise RuntimeError("Pending response failed")

        def close(self):
            calls.append(("endpoint", self.i))

    class Base:
        def __init__(self):
            self.closed, self.waiting = (
                False,
                case in ("waiting", "pending_timeout", "pending_error"),
            )
            self.processes = [Process(i) for i in range(4)]
            processes.extend(self.processes)
            if case == "duplicate":
                processes[1].pid = processes[0].pid
            self.remotes = [Remote(i) for i in range(3 if case == "counts" else 4)]

        def reset(self):
            pytest.fail("No reset during retirement")

        def step(self, action):
            pytest.fail("No step during retirement")

    def identity(pid):
        p = next(p for p in processes if p.pid == pid)
        if not p.running and case != "present_after_join":
            return None
        return {"pid": pid, "start": str(pid + 10), "state": "R"}

    path = tmp_path / "serial.jsonl"
    cls = serial_vec_class(Base, path, identity_reader=identity, clock=lambda: clock[0])
    if case in ("counts", "duplicate"):
        with pytest.raises(ValueError):
            cls()
        assert not calls
        return
    vec = cls()
    if case in ("missing", "reused", "dead"):
        original = identity

        def identity(pid):
            return (
                None
                if case == "missing"
                else original(pid) | {"start": "new"}
                if case == "reused"
                else original(pid) | {"state": "Z"}
            )

        # A second class binds the ordinary identities; change the captured reader
        # through a mutable callable, rather than allowing identity to float.
        reader = [original]
        cls = serial_vec_class(
            Base, path, identity_reader=lambda p: reader[0](p), clock=lambda: clock[0]
        )
        processes.clear()
        vec = cls()
        reader[0] = identity
    if case in ("normal", "waiting", "repeat"):
        vec.close()
        if case == "repeat":
            prior = calls.copy()
            vec.close()
            assert calls == prior
        expected = [
            (kind, i) for i in range(4) for kind in ("send", "join", "endpoint")
        ]
        assert (
            calls
            == ([("recv", i) for i in range(4)] if case == "waiting" else []) + expected
        )
        assert vec.closed and not vec.waiting
    else:
        with pytest.raises((RuntimeError, TimeoutError, BrokenPipeError)):
            vec.close()
        assert not vec.closed
        before = calls.copy()
        with pytest.raises(RuntimeError, match="previously failed"):
            vec.close()
        assert calls == before and ("send", 1) not in calls
        assert json.loads(path.read_text().splitlines()[-1])["phase"] == "fault"


@pytest.mark.skipif(
    sys.platform != "linux", reason="Actual owned Linux heap retirement"
)
def test_actual_serial_heaps_guard_and_foreign_survival(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from supervise_motor_pilot import identity
    from trial_rss_watchdog import bound_rss

    ready = tmp_path / "watchdog.ready"
    report = tmp_path / "watchdog.json"
    events = tmp_path / "serial.jsonl"
    code = r"""
import subprocess,sys,time
from pathlib import Path
from joxing.serial_retirement import serial_vec_class
ready=Path(sys.argv[1]);deadline=time.monotonic()+5
while not ready.exists():
 if time.monotonic()>deadline:raise RuntimeError('No watchdog')
 time.sleep(.005)
class Process:
 def __init__(self):
  self.child=subprocess.Popen([sys.executable,'-c',"import os,sys;x=bytearray(512*1024**2);print('allocated',flush=True);sys.stdin.readline();os._exit(0)"],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
  self.pid=self.child.pid
 def is_alive(self):return self.child.poll() is None
 def join(self,timeout):
  try:self.child.wait(timeout=timeout)
  except subprocess.TimeoutExpired:pass
 @property
 def exitcode(self):return self.child.returncode
class Remote:
 def __init__(self,p):self.p=p
 def send(self,msg):
  assert msg==('close',None);self.p.child.stdin.write(b'close\n');self.p.child.stdin.flush()
 def close(self):self.p.child.stdin.close();self.p.child.stdout.close()
class Base:
 def __init__(self):
  self.closed=self.waiting=False
  self.processes=[Process() for i in range(4)]
  self.remotes=[Remote(p) for p in self.processes]
  for p in self.processes:assert p.child.stdout.readline()==b'allocated\n'
vec=serial_vec_class(Base,sys.argv[2])()
vec.close()
assert vec.closed
"""
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
    )
    root = subprocess.Popen(
        [sys.executable, "-c", code, str(ready), str(events)], start_new_session=True
    )
    own, bound = identity(os.getpid()), identity(root.pid)
    baseline = bound_rss(own)
    cmd = [
        sys.executable,
        str(scripts / "trial_rss_watchdog.py"),
        "--root-pid",
        str(bound["pid"]),
        "--root-start",
        str(bound["start_ticks"]),
        "--supervisor-pid",
        str(own["pid"]),
        "--supervisor-start",
        str(own["start_ticks"]),
        "--output",
        str(report),
        "--planned-rss",
        str(baseline + 3 * 1024**3),
        "--hard-rss",
        str(baseline + 4 * 1024**3),
        "--process-cap",
        "8",
        "--wall-s",
        "10",
        "--rss-reader",
        "terminal-mm-reserve-v3",
    ]
    watcher = None
    try:
        watcher = subprocess.Popen(cmd)
        watcher.wait(timeout=12)
        root.wait(timeout=2)
        w = json.loads(report.read_text())
        assert watcher.returncode == root.returncode == 0, w
        assert (
            w["status"] == "exited"
            and not w["pending_retirements"]
            and not w["final_live"]
        )
        assert max(w["max_sample_gap_s"], w["max_probe_s"]) <= 0.1
        assert w["retirement_records"], "No retirement observed; do not qualify"
        assert all(
            r["released"] and r["max_hold_s"] <= 0.1 and r["highwater_charge_bytes"] > 0
            for r in w["retirement_records"].values()
        )
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        assert [(r["phase"], r.get("worker")) for r in rows[1:-1]] == [
            (phase, i) for i in range(4) for phase in ("close_send", "joined")
        ]
        assert foreign.poll() is None
        (tmp_path / "foreign.json").write_text(
            json.dumps({"identity": identity(foreign.pid), "survived_guard": True})
            + "\n"
        )
    finally:
        for p in [root, watcher, foreign]:
            if p is not None and p.poll() is None:
                p.kill()
                p.wait(timeout=2)
