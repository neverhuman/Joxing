import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from joxing import lean_worker_probe as probe


def test_portable_preserves_signed_zero_dtype_shape_and_snapshot_ownership():
    env = probe.make_fake(1097)
    env.reset(seed=1097)
    before = env.unwrapped.snapshot()
    one = probe.portable(before)
    before["observation"]["state"][0] = 99
    before["rng"]["state"]["state"] = 0
    assert one == probe.portable(env.unwrapped.snapshot())
    assert probe.portable(-0.0) != probe.portable(0.0)
    assert probe.portable(np.array([1.0], dtype=np.float32)) != probe.portable(
        np.array([1.0], dtype=np.float64)
    )
    assert probe.portable(np.array([[1.0]])) != probe.portable(np.array([1.0]))


@pytest.mark.parametrize("value", [np.array([np.nan]), float("inf"), object()])
def test_nonfinite_or_unsupported_semantics_fail_closed(value):
    with pytest.raises((TypeError, ValueError)):
        probe.portable(value)


def test_full_fresh_forkserver_pair_matches_and_candidate_import_isolation(tmp_path):
    result = tmp_path / "proof.json"
    code = r"""
import multiprocessing as mp,json,sys
from pathlib import Path
from joxing.lean_worker_probe import make_fake,exercise,digest,transact
from joxing.lean_rollout_worker import lean_pipe_worker,LeanCloudpickleWrapper
from functools import partial
mp.set_forkserver_preload([])
from multiprocessing import forkserver
forkserver.ensure_running()
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
from stable_baselines3.common.vec_env.subproc_vec_env import _worker
ctx=mp.get_context('forkserver');results=[];imports=[]
for target,wrapper in ((_worker,CloudpickleWrapper),(lean_pipe_worker,LeanCloudpickleWrapper)):
 a,b=ctx.Pipe();p=ctx.Process(target=target,args=(b,a,wrapper(partial(make_fake,1097))));p.start();b.close()
 try:
  rows=exercise(a,1097);results.append(rows)
  imports.append(transact(a,('env_method',('import_snapshot',(),{}))))
  a.send(('close',None));p.join(timeout=3);assert p.exitcode==0
 finally:
  a.close()
  if p.is_alive():p.terminate();p.join(timeout=1)
assert results[0]==results[1] and digest(results[0])==digest(results[1])
assert imports[0]['torch'] and imports[0]['stable_baselines3'] and not imports[0]['mujoco']
assert not any(imports[1].values())
Path(sys.argv[1]).write_text(json.dumps({'rows':results[1],'imports':imports,'native_physics':0}))
"""
    env = os.environ | {
        k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    subprocess.run(
        [sys.executable, "-c", code, str(result)], env=env, check=True, timeout=15
    )
    v = json.loads(result.read_text())
    rows = v["rows"]
    assert len(rows) == 26

    # Explicit known output oracle, beyond equality with another implementation.
    # Initial12commands; each mode then reset,step,terminal-step,post-autoreset-step.
    def pairs(encoded):
        return {k: value for k, value in encoded["dict"]}

    for offset, mode in ((12, "terminated"), (16, "truncated"), (20, "both")):
        first = rows[offset + 1]["response"]["tuple"]
        last = rows[offset + 2]["response"]["tuple"]
        following = rows[offset + 3]["response"]["tuple"]
        assert first[2] is False and last[2] is True and following[2] is False
        info = pairs(last[3])
        assert info["TimeLimit.truncated"] is (mode == "truncated")
        assert "terminal_observation" in info and "terminal_observation" not in pairs(
            following[3]
        )
        assert last[4] == following[4]
        obs = pairs(info["terminal_observation"])
        raw = bytes.fromhex(obs["state"]["array"]["hex"])
        assert np.frombuffer(raw, dtype=np.float32)[0] == 2
    assert rows[4]["response"] == 3 and rows[6]["response"] == 17
    assert rows[2]["response"] is True and rows[3]["response"] is False
    assert rows[8]["response"] is True and rows[9]["response"] is False
    assert v["native_physics"] == 0
    from audit_lean_worker_probe import check_semantics

    assert check_semantics(rows)["commands"] == 26
    for fault in ("done", "truncation", "reset_info", "terminal", "missing_rng"):
        import copy

        bad = copy.deepcopy(rows)
        terminal = bad[14]["response"]["tuple"]
        if fault == "done":
            terminal[2] = False
        elif fault == "truncation":
            next(
                pair for pair in terminal[3]["dict"] if pair[0] == "TimeLimit.truncated"
            )[1] = True
        elif fault == "reset_info":
            terminal[4] = {"dict": []}
        elif fault == "terminal":
            terminal[3]["dict"] = [
                p for p in terminal[3]["dict"] if p[0] != "terminal_observation"
            ]
        else:
            bad[0]["before"]["dict"] = [
                p for p in bad[0]["before"]["dict"] if p[0] != "rng"
            ]
        with pytest.raises(ValueError):
            check_semantics(bad)


def saved_pairs(tmp_path):
    folders = {}
    for name, profile, seed in probe.SLOTS:
        path = tmp_path / name
        path.mkdir()
        folders[name] = str(path)
        semantic = [{"fake": [0.0, False]} for _ in range(26)]
        result = {
            "id": name,
            "profile": profile,
            "seed": seed,
            "semantic_sha256": probe.digest(semantic),
            "closed_state_sha256": "closed",
            "worker_rss_before": 1000 if profile == "sb3" else 400,
            "worker_rss_after": 1000 if profile == "sb3" else 450,
            "physics": 0,
            "actors": 0,
            "updates": 0,
            "inference": 0,
        }
        (path / "semantic.json").write_text(json.dumps(semantic))
        (path / "probe-result.json").write_text(json.dumps(result))
    return folders


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "reward",
        "rng",
        "signed_zero",
        "bool_type",
        "missing",
        "short",
        "work",
        "memory",
        "state",
    ],
)
def test_conclusion_rejects_false_pair_or_denominator(tmp_path, fault):
    folders = saved_pairs(tmp_path)
    path = tmp_path / "leanB"
    s = json.loads((path / "semantic.json").read_text())
    r = json.loads((path / "probe-result.json").read_text())
    if fault in ("reward", "rng"):
        s[0]["fake"][0] = 1e-15 if fault == "reward" else "changed_rng"
    elif fault == "signed_zero":
        s[0]["fake"][0] = -0.0
    elif fault == "bool_type":
        s[0]["fake"][1] = 0
    elif fault == "short":
        for name in ("leanB", "sb3B"):
            p = tmp_path / name
            v = json.loads((p / "semantic.json").read_text())[:-1]
            (p / "semantic.json").write_text(json.dumps(v))
            x = json.loads((p / "probe-result.json").read_text())
            x["semantic_sha256"] = probe.digest(v)
            (p / "probe-result.json").write_text(json.dumps(x))
        s = s[:-1]
    elif fault == "work":
        r["physics"] = 1
    elif fault == "memory":
        r["worker_rss_after"] = 751
    elif fault == "state":
        r["closed_state_sha256"] = "changed"
    (path / "semantic.json").write_text(json.dumps(s))
    r["semantic_sha256"] = probe.digest(s)
    (path / "probe-result.json").write_text(json.dumps(r))
    if fault == "missing":
        (path / "probe-result.json").unlink()
    if fault == "none":
        assert probe.conclude(folders)["qualified_fake_ipc_only"]
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            probe.conclude(folders)


def test_wrong_packet_rejected_before_readiness(monkeypatch):
    monkeypatch.setattr(probe, "wait_ready", lambda *a: pytest.fail("must reject"))
    with pytest.raises(ValueError):
        probe.run(None, None, {"schema": "old"}, None)


@pytest.mark.parametrize(
    "fault", ["none", "method", "threads", "parent", "torch", "mujoco"]
)
def test_witness_rejects_context_or_import_drift(tmp_path, monkeypatch, fault):
    bound = {"pid": 19, "start": "2"}
    monkeypatch.setattr(probe, "process_identity", lambda: bound)
    monkeypatch.setattr(
        probe.mp,
        "get_start_method",
        lambda: "spawn" if fault == "method" else "forkserver",
    )
    monkeypatch.setattr(
        probe,
        "imports",
        lambda: {
            "torch": fault == "torch",
            "stable_baselines3": False,
            "mujoco": fault == "mujoco",
        },
    )

    def facts(*args, evidence_dir):
        evidence_dir.mkdir()
        (evidence_dir / "status.txt").write_text("VmRSS:\t100 kB\n")
        return {
            "parent_pid": 99 if fault == "parent" else 10,
            "threads": 2 if fault == "threads" else 1,
            "after_threads": 1,
            "heavy_mappings": ["libtorch"] if fault == "torch" else [],
        }

    monkeypatch.setattr(probe, "facts", facts)
    if fault == "none":
        v = probe.witness(tmp_path, {"pid": 10, "start": "1"}, "lean", "observed")
        assert v["rss_bytes"] == 102400
        assert (
            probe.digest(json.loads(Path(v["modules_path"]).read_text()))
            == v["modules_sha256"]
        )
    else:
        with pytest.raises(ValueError):
            probe.witness(tmp_path, {"pid": 10, "start": "1"}, "lean", "observed")


def test_sampling_profile_is_scoped_and_original_default_unchanged(monkeypatch):
    import runpy

    import trial_rss_watchdog as original

    old = original.monitor
    called = []

    def main():
        assert original.monitor.func is old
        called.append(original.monitor.keywords)

    monkeypatch.setattr(original, "main", main)
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/lean_probe_watchdog.py"),
        run_name="__main__",
    )
    assert called == [{"poll_s": 0.001}] and original.monitor is old


@pytest.mark.parametrize(
    "fault", ["none", "schema", "workers", "physics", "cap", "poll", "slots"]
)
def test_entry_requires_explicit_zero_native_profile(fault):
    import qualify_lean_worker_probe as entry

    p = {
        "schema": probe.SCHEMA,
        "whole_wall_s": 300,
        "per_slot_wall_s": 30,
        "control_wall_s": 30,
        "physics_limit": 0,
        "fixture_bytes": 16 * 1024**2,
        "receipt_bytes": 1024**2,
        "receipt_reserve_bytes": 768 * 1024,
        "simulation_workers": 1,
        "threads_per_process": 1,
        "process_ceiling": 12,
        "planned_stop_rss_bytes": 7 * 1024**3,
        "owned_rss_bytes": 8 * 1024**3,
        "max_monitor_gap_s": 0.1,
        "close_deadline_s": 2,
        "ipc_bytes": 1024**2,
        "rss_reader": "terminal-mm-reserve-v3",
        "watchdog_output_backend": "private-tmpfs-v1",
        "watchdog_sampling_profile": "lean-probe-1ms-v1",
        "slots": [{"id": n, "profile": m, "seed": s} for n, m, s in probe.SLOTS],
    }
    if fault == "schema":
        p["schema"] = "old"
    elif fault == "workers":
        p["simulation_workers"] = 2
    elif fault == "physics":
        p["physics_limit"] = 1
    elif fault == "cap":
        p["max_monitor_gap_s"] = 0.11
    elif fault == "poll":
        p["watchdog_sampling_profile"] = "old"
    elif fault == "slots":
        p["slots"].pop()
    if fault == "none":
        entry.validate(p)
    else:
        with pytest.raises(ValueError):
            entry.validate(p)


@pytest.mark.parametrize("fault", ["none", "budget", "projection", "scope"])
def test_receipt_first_limit_blocks_before_probe(tmp_path, monkeypatch, fault):
    import qualify_lean_worker_probe as entry

    p = {
        "receipt_projection_file": "sizing.json",
        "receipt_reserve_bytes": 768 * 1024,
        "receipt_bytes": 1024**2,
    }
    v = {
        "scope": "synthetic stored-evidence sizing; zero native work",
        "projected_peak_receipt_bytes": 500000,
    }
    usage = {"receipt_bytes": 20000}
    if fault == "budget":
        usage["receipt_bytes"] = 300000
    elif fault == "projection":
        v["projected_peak_receipt_bytes"] = 800000
    elif fault == "scope":
        v["scope"] = "unregistered"
    (tmp_path / "sizing.json").write_text(json.dumps(v))
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    if fault == "none":
        assert entry.receipt_headroom(p, usage)["reserved"] == 768 * 1024
    else:
        with pytest.raises(ValueError):
            entry.receipt_headroom(p, usage)
