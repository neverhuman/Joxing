import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from joxing import native_context as nc


def test_import_stays_stdlib_before_readiness():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;import joxing.context_native_teardown;assert not {'numpy','torch','mujoco','stable_baselines3'}&sys.modules.keys()",
        ],
        check=True,
        timeout=5,
    )


@pytest.mark.parametrize("method", ["fork", "unknown", "forkserver"])
def test_unproved_context_rejected_before_readiness(monkeypatch, method):
    from joxing import release_native_teardown as nt

    monkeypatch.setattr(
        nt,
        "wait_ready",
        lambda *a: pytest.fail("Must reject before imports/native work"),
    )
    with pytest.raises(ValueError, match="proof required"):
        nt.run(None, None, None, None, None, None, start_method=method)


@pytest.mark.parametrize(
    "case",
    [
        "clean",
        "loaded",
        "existing",
        "root_threads",
        "root_heavy",
        "server_threads",
        "server_heavy",
        "foreign_server",
        "preload",
    ],
)
def test_prepare_refuses_dirty_or_foreign_server(tmp_path, monkeypatch, case):
    import multiprocessing

    monkeypatch.setattr(nc, "HEAVY", ("joxing_fake_heavy",))
    if case == "loaded":
        monkeypatch.setitem(sys.modules, "joxing_fake_heavy", object())
    handle = SimpleNamespace(
        _forkserver_pid=17 if case == "existing" else None,
        _preload_modules=["__main__"],
    )

    def preload(value):
        handle._preload_modules = ["bad"] if case == "preload" else value

    def ensure():
        handle._forkserver_pid = 17

    monkeypatch.setattr(
        multiprocessing,
        "forkserver",
        SimpleNamespace(
            _forkserver=handle, set_forkserver_preload=preload, ensure_running=ensure
        ),
        raising=False,
    )
    monkeypatch.setattr(nc, "process_identity", lambda: {"pid": 11, "start": "1"})
    monkeypatch.setattr(nc.os, "getpid", lambda: 11)
    monkeypatch.setattr(nc, "bound_process", lambda pid: {"pid": pid, "start": "2"})

    def fake(bound, **kwargs):
        root = bound["pid"] == 11
        return {
            "identity": bound,
            "threads": 2
            if case == ("root_threads" if root else "server_threads")
            else 1,
            "after_threads": 1,
            "heavy_mappings": ["libtorch"]
            if case == ("root_heavy" if root else "server_heavy")
            else [],
            "parent_pid": 99 if case == "foreign_server" else 11,
        }

    monkeypatch.setattr(nc, "facts", fake)
    proof = nc.NativeContextProof(tmp_path / "native-context")
    if case == "clean":
        proof.prepare()
        assert proof.proof["preload"] == [] and proof.server == {
            "pid": 17,
            "start": "2",
        }
        assert (proof.folder / "before.json").exists()
    else:
        with pytest.raises(RuntimeError):
            proof.prepare()


@pytest.mark.parametrize("case", ["clean", "method", "parent", "threads"])
def test_child_witness_before_native_constructor(tmp_path, monkeypatch, case):
    monkeypatch.setattr(
        nc.mp, "get_start_method", lambda: "spawn" if case == "method" else "forkserver"
    )
    monkeypatch.setattr(nc, "process_identity", lambda: {"pid": 21, "start": "3"})
    monkeypatch.setattr(
        nc,
        "facts",
        lambda *a, **k: {
            "parent_pid": 99 if case == "parent" else 17,
            "threads": 2 if case == "threads" else 1,
            "after_threads": 1,
        },
    )
    if case == "clean":
        assert (
            nc.worker_witness(tmp_path, {"pid": 17, "start": "2"})["worker"]["pid"]
            == 21
        )
    else:
        with pytest.raises(RuntimeError):
            nc.worker_witness(tmp_path, {"pid": 17, "start": "2"})
    assert (tmp_path / "context-worker-before.json").exists()


@pytest.mark.parametrize(
    "case", ["valid", "missing", "foreign", "early", "worker_drift"]
)
def test_managed_finalizer_and_result_identity_required(tmp_path, case):
    bound = {"pid": 21, "start": "3"}
    proof = nc.NativeContextProof(tmp_path / "context")
    proof.proof = {"worker": bound}
    result = {
        "worker": {"pid": 22, "start": "3"} if case == "worker_drift" else bound,
        "release_phases": [{"monotonic_s": 2}],
    }
    rows = (
        []
        if case == "missing"
        else [
            {
                "phase": "managed_finalizer",
                "identity": {"pid": 99, "start": "9"} if case == "foreign" else bound,
                "monotonic_s": 1 if case == "early" else 3,
            }
        ]
    )
    (tmp_path / "context-finalizer.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    if case == "valid":
        assert proof.qualify(result, tmp_path)["finalizer"] == rows[0]
    else:
        with pytest.raises(ValueError):
            proof.qualify(result, tmp_path)


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "worker_reused",
        "server_reused",
        "method",
        "threads",
        "server_heavy",
        "parent",
        "two_workers",
    ],
)
def test_post_constructor_proof_rejects_context_drift(tmp_path, monkeypatch, case):
    native = tmp_path / "native"
    native.mkdir()
    worker, server = {"pid": 21, "start": "3"}, {"pid": 17, "start": "2"}
    proof = nc.NativeContextProof(tmp_path / "native-context")
    proof.server, proof.proof = server, {"root": {"pid": 11, "start": "1"}}
    witness = {
        "method": "spawn" if case == "method" else "forkserver",
        "worker": worker,
        "server": {"pid": 17, "start": "9"} if case == "server_reused" else server,
    }
    (native / "context-worker-before.json").write_text(json.dumps(witness))
    monkeypatch.setattr(
        nc,
        "bound_process",
        lambda pid: {"pid": pid, "start": "9"} if case == "worker_reused" else worker,
    )

    def capture(bound, **kwargs):
        return {
            "identity": bound,
            "threads": 2 if case == "threads" else 1,
            "after_threads": 1,
            "heavy_mappings": ["libtorch"] if case == "server_heavy" else [],
            "parent_pid": 99 if case == "parent" else (11 if bound == server else 17),
        }

    monkeypatch.setattr(nc, "facts", capture)
    vec = SimpleNamespace(
        processes=[SimpleNamespace(pid=21)] * (2 if case == "two_workers" else 1)
    )
    if case == "valid":
        proof.after_construct(vec)
        assert proof.proof["worker"] == worker
    else:
        with pytest.raises(RuntimeError):
            proof.after_construct(vec)


def test_supervisor_constructor_route_is_explicit():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "qualify_context", scripts / "qualify_deferred_trial.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.runner_module("joxing-native-context-transfer-v1")
        == "joxing.context_native_teardown"
    )
    assert (
        module.runner_module("joxing-release-native-teardown-v1")
        == "joxing.release_native_teardown"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Actual Linux proc/context proof")
def test_actual_clean_context_and_managed_exit_zero_native(tmp_path):
    module = tmp_path / "fixture_context_worker.py"
    module.write_text("""from joxing.native_context import worker_witness,finalizer_witness
from multiprocessing.util import Finalize
def child(conn,folder,server):
 w=worker_witness(folder,server)
 Finalize(None,finalizer_witness,args=(folder,w['worker']),exitpriority=5)
 conn.send(w['worker'])
 assert conn.recv()=='close'
 conn.close()
""")
    code = f"""
import json,multiprocessing as mp,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,{str(tmp_path)!r})
from fixture_context_worker import child
from joxing.native_context import NativeContextProof
root=Path({str(tmp_path)!r}); native=root/'native';native.mkdir()
p=NativeContextProof(root/'native-context');p.prepare()
ctx=mp.get_context('forkserver');a,b=ctx.Pipe()
worker=ctx.Process(target=child,args=(b,str(native),p.server));worker.start();b.close()
try:
 assert a.poll(4);bound=a.recv()
 p.after_construct(SimpleNamespace(processes=[worker]))
 a.send('close');worker.join(timeout=2);assert worker.exitcode==0
 result=p.qualify({{'worker':bound,'release_phases':[{{'monotonic_s':0}}]}},native)
 (root/'context-control.json').write_text(json.dumps({{'proof':result,'physics':0,'scope':'stdlib toy child; no native model'}}))
finally:
 a.close()
 if worker.is_alive():worker.terminate();worker.join(timeout=2)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-B", "-c", code], env=env, check=True, timeout=12)
    value = json.loads((tmp_path / "context-control.json").read_text())
    assert (
        value["physics"] == 0 and value["proof"]["server_after"]["heavy_mappings"] == []
    )
