import json
import subprocess
import sys

import pytest

from joxing.process_exit_probe import (
    SLOTS,
    conclude,
    facts,
    recv,
    validate_phases,
    validate_server,
)


@pytest.mark.parametrize("case", ["stdlib", "mixed", "heavy", "timeout"])
def test_fresh_worker_cannot_import_before_release(tmp_path, case):
    code = """
import sys
import joxing.process_exit_probe as p
assert not {'numpy','torch','mujoco','stable_baselines3'}&sys.modules.keys()
p.mp.get_start_method=lambda:'spawn'
class Conn:
 def send(self,x):pass
 def poll(self,t):return False
 def recv(self):raise AssertionError('Not ready')
"""
    if case == "mixed":
        code += f"\ntry:p.worker(Conn(),{str(tmp_path)!r},'forkserver',1097)\nexcept RuntimeError as e:assert 'Mixed' in str(e)\nelse:raise AssertionError('Mixed context accepted')"
    if case == "heavy":
        code += f"\nsys.modules['torch']=object()\ntry:p.worker(Conn(),{str(tmp_path)!r},'spawn',1097)\nexcept RuntimeError as e:assert 'Heavy' in str(e)\nelse:raise AssertionError('Heavy preload accepted')"
    if case == "timeout":
        code += f"\ntry:p.worker(Conn(),{str(tmp_path)!r},'spawn',1097)\nexcept TimeoutError:pass\nelse:raise AssertionError('No readiness deadline')"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5)


@pytest.mark.parametrize(
    "case",
    ["clean", "threads", "after_threads", "heavy", "foreign_server", "foreign_worker"],
)
def test_clean_server_and_ancestry_are_required(case):
    server = {
        "identity": {"pid": 17, "start": "10"},
        "threads": 1,
        "after_threads": 1,
        "heavy_mappings": [],
        "parent_pid": 11,
    }
    worker = {"parent_pid": 17}
    if case == "threads":
        server["threads"] = 2
    if case == "after_threads":
        server["after_threads"] = 2
    if case == "heavy":
        server["heavy_mappings"] = ["libtorch"]
    if case == "foreign_server":
        server["parent_pid"] = 99
    if case == "foreign_worker":
        worker["parent_pid"] = 99
    if case == "clean":
        validate_server(server, worker, 11)
    else:
        with pytest.raises(RuntimeError):
            validate_server(server, worker, 11)


@pytest.mark.parametrize("case", ["clean", "reused", "exiting"])
def test_bound_proc_proof_rejects_foreign_memory(tmp_path, case):
    d = tmp_path / "17"
    d.mkdir()
    fields = (
        ["R", "11", "0", "0", "0", "0", "4" if case == "exiting" else "0"]
        + ["0"] * 10
        + ["1", "0", "11" if case == "reused" else "10"]
    )
    assert len(fields) == 20
    (d / "stat").write_text("17 (fixture) " + " ".join(fields))
    (d / "status").write_text("Threads: 1\n")
    (d / "maps").write_text("python\n")
    if case == "clean":
        assert facts({"pid": 17, "start": "10"}, tmp_path)["heavy_mappings"] == []
    else:
        (d / "maps").unlink()  # Identity rejection must precede any foreign maps read.
        with pytest.raises(RuntimeError):
            facts({"pid": 17, "start": "10"}, tmp_path)


@pytest.mark.parametrize(
    "case", ["valid", "missing_finalizer", "foreign", "wrong_order"]
)
def test_exit_receipts_require_complete_managed_cleanup(case):
    own = {"pid": 17, "start": "10"}
    names = [
        "waiting_release",
        "import_begin",
        "ready",
        "close_received",
        "before_return",
        "managed_finalizer",
    ]
    rows = [
        {"phase": p, "identity": own, "monotonic_s": i} for i, p in enumerate(names)
    ]
    if case == "missing_finalizer":
        rows.pop()
    if case == "foreign":
        rows[1]["identity"] = {"pid": 18, "start": "99"}
    if case == "wrong_order":
        rows[-1]["monotonic_s"] = 0
    if case == "valid":
        assert validate_phases("forkserver", rows, own) == names
    else:
        with pytest.raises(ValueError):
            validate_phases("forkserver", rows, own)


@pytest.mark.parametrize(
    "case", ["supported", "negative", "mismatch", "missing", "work", "slow", "nan"]
)
def test_paired_conclusion_keeps_negative_and_invalid_denominators(tmp_path, case):
    folders = {}
    for i, (name, method, seed) in enumerate(SLOTS):
        d = tmp_path / name
        d.mkdir()
        folders[name] = str(d)
        r = {
            "method": method,
            "seed": seed,
            "close_to_join_s": 0.8 if method == "spawn" else 0.1,
            "payload_sha256": str(seed),
            "physics": 0,
            "actors": 0,
            "updates": 0,
            "inference": 0,
        }
        if case == "negative" and method == "forkserver":
            r["close_to_join_s"] = 0.9
        if case == "mismatch" and i == 1:
            r["payload_sha256"] = "changed"
        if case == "work" and i == 1:
            r["updates"] = 1
        if case == "slow" and i == 1:
            r["close_to_join_s"] = 2.1
        if case == "nan" and i == 1:
            r["close_to_join_s"] = float("nan")
        if case != "missing" or i != 1:
            (d / "probe-result.json").write_text(json.dumps(r))
    if case in ("supported", "negative"):
        r = conclude(folders)
        assert len(r["pairs"]) == 3
        assert r["scientific_verdict"] == (
            "supported_import_only"
            if case == "supported"
            else "not_supported_import_only"
        )
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            conclude(folders)


def test_bounded_receive_preserves_pipe_fault():
    class C:
        def poll(self, t):
            return False

        def recv(self):
            pytest.fail("No read without ready")

    with pytest.raises(TimeoutError):
        recv(C(), 0.01)

    class Broken(C):
        def poll(self, t):
            return True

        def recv(self):
            raise EOFError("original")

    with pytest.raises(EOFError, match="original"):
        recv(Broken(), 0.01)
