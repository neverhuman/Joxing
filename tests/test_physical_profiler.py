"""Profiler accounting and failure retention without a physical sweep."""

import cProfile
import json
import runpy

from joxing.provenance import ROOT, digest

helpers = runpy.run_path(str(ROOT / "scripts/profile_physical_pipeline.py"))


def test_all_six_cases_and_alternating_order_registered():
    p = json.loads((ROOT / "experiments/v0.0.9/profile-protocol.json").read_text())
    assert digest(p) == helpers["PROTOCOL_SHA"]
    cases = helpers["planned_cases"](p)
    assert len(cases) == len({row["case_id"] for row in cases}) == 6
    assert [row["mode"] for row in cases] == [
        "unprofiled",
        "profiled",
        "profiled",
        "unprofiled",
        "unprofiled",
        "profiled",
    ]
    assert p["control_steps"] * 6 * 10 <= p["caps"]["physics_steps"]


def test_profiler_keeps_own_and_overlapping_cumulative_times_distinct():
    profiler = cProfile.Profile()
    profiler.runcall(lambda: sum(range(100)))
    rows = helpers["profile_rows"](profiler)
    assert rows and all(row["own_s"] <= row["cumulative_s"] for row in rows)
    assert rows == sorted(rows, key=lambda r: -r["own_s"])


def test_missing_checkpoint_preserves_all_future_cases_and_zero_physics(tmp_path):
    result = helpers["run"](
        ROOT / "experiments/v0.0.9/profile-protocol.json",
        tmp_path / "missing",
        tmp_path / "out",
    )
    assert result["status"] == "halted" and result["physics_steps"] == 0
    assert len(result["cases"]) == 6 and all(
        row["status"] == "not_run" for row in result["cases"]
    )
    assert (tmp_path / "out/result.json").exists()


def test_ipc_probe_verifies_content_and_terminates_on_empty_payload():
    class Connection:
        def __init__(self):
            self.values = iter([b"snapshot payload", b""])
            self.sent = []
            self.closed = False

        def recv_bytes(self):
            return next(self.values)

        def send_bytes(self, value):
            self.sent.append(value)

        def close(self):
            self.closed = True

    import hashlib

    connection = Connection()
    helpers["echo_worker"](connection)
    assert connection.sent == [hashlib.sha256(b"snapshot payload").digest()]
    assert connection.closed
