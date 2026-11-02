"""Import-only process backend diagnostic. Parent and server stay stdlib-only."""

import argparse
import atexit
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from pathlib import Path

from .deferred_qualification import wait_ready
from .physics_credit import process_identity
from .serial_retirement import read_identity, serial_vec_class

SLOTS = [
    ("spawn1097", "spawn", 1097),
    ("forkserver1097", "forkserver", 1097),
    ("forkserver1098", "forkserver", 1098),
    ("spawn1098", "spawn", 1098),
    ("spawn1099", "spawn", 1099),
    ("forkserver1099", "forkserver", 1099),
]
HEAVY = ("torch", "numpy", "mujoco", "stable_baselines3")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def emit(folder, phase, **values):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker-events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "phase": phase,
                    "identity": process_identity(),
                    "monotonic_s": time.monotonic(),
                    **values,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def facts(bound, proc_root=Path("/proc"), *, evidence_dir=None):
    folder = proc_root / str(bound["pid"])
    before_raw = (folder / "stat").read_text()
    before = before_raw.rsplit(")", 1)[1].split()
    if (
        before[19] != str(bound["start"])
        or before[0] in ("Z", "X")
        or int(before[6]) & 4
    ):
        raise RuntimeError("Probe identity/live-state drift")
    status_raw = (folder / "status").read_text()
    status = {
        k: v.strip()
        for line in status_raw.splitlines()
        if ":" in line
        for k, v in [line.split(":", 1)]
    }
    maps_raw = (folder / "maps").read_text()
    maps = maps_raw.lower()
    after_raw = (folder / "stat").read_text()
    after = after_raw.rsplit(")", 1)[1].split()
    if after[19] != str(bound["start"]) or after[0] in ("Z", "X") or int(after[6]) & 4:
        raise RuntimeError("Probe identity changed during proof")
    evidence = {}
    if evidence_dir is not None:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=False)
        for name, raw in {
            "stat-before.txt": before_raw,
            "stat-after.txt": after_raw,
            "status.txt": status_raw,
            "maps.txt": maps_raw,
        }.items():
            path = evidence_dir / name
            path.write_text(raw)
            evidence[name] = {
                "path": str(path),
                "bytes": len(raw.encode()),
                "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            }
    return {
        "identity": bound,
        "parent_pid": int(after[1]),
        "threads": int(status["Threads"]),
        "after_threads": int(after[17]),
        "before_state": before[0],
        "after_state": after[0],
        "before_flags": int(before[6]),
        "after_flags": int(after[6]),
        "heavy_mappings": [
            s for s in ("libtorch", "numpy", "mujoco", "openblas", "mkl") if s in maps
        ],
        "maps_sha256": hashlib.sha256(maps_raw.encode()).hexdigest(),
        "proc_evidence": evidence,
    }


def validate_server(server, worker, parent_pid):
    if (
        server["threads"] != 1
        or server["after_threads"] != 1
        or server["heavy_mappings"]
        or server["parent_pid"] != parent_pid
        or worker["parent_pid"] != server["identity"]["pid"]
    ):
        raise RuntimeError("Unqualified forkserver/worker ancestry or preload")


def worker(conn, folder, method, seed):
    if any(name in sys.modules for name in HEAVY):
        raise RuntimeError("Heavy import before worker release")
    if mp.get_start_method() != method:
        raise RuntimeError("Mixed multiprocessing contexts")
    emit(folder, "waiting_release", method=method, seed=seed)
    conn.send({"identity": process_identity(), "method": mp.get_start_method()})
    if not conn.poll(5) or conn.recv() != "release":
        raise TimeoutError("Worker readiness release")
    emit(folder, "import_begin")
    from multiprocessing.util import Finalize

    import mujoco
    import numpy as np
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3.sac.policies import Actor

    def denied(*args, **kwargs):
        emit(folder, "forbidden_operation")
        raise RuntimeError("Import-only probe attempted native/learning work")

    mujoco.mj_step = mujoco.MjData = mujoco.MjModel = denied
    Actor.__init__ = denied
    for name in ("forward", "_predict", "predict", "action_log_prob"):
        setattr(Actor, name, denied)
    sb3.SAC.learn = denied
    for name in dir(torch.optim):
        cls = getattr(torch.optim, name)
        if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
            cls.step = denied
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    array = np.arange(8, dtype=np.float64) + np.random.random(8)
    tensor = torch.arange(8, dtype=torch.float64) + torch.rand(8, dtype=torch.float64)
    rng = np.random.get_state()
    payload = {
        "seed": seed,
        "array_dtype": str(array.dtype),
        "tensor_dtype": str(tensor.dtype),
        "shape": [8],
        "array_hex": array.tobytes().hex(),
        "tensor_hex": tensor.numpy().tobytes().hex(),
        "requires_grad": tensor.requires_grad,
        "python_rng_sha256": hashlib.sha256(
            repr(random.getstate()).encode()
        ).hexdigest(),
        "numpy_rng_sha256": hashlib.sha256(
            rng[1].tobytes() + str(rng[2:]).encode()
        ).hexdigest(),
        "torch_rng_sha256": hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "sb3": sb3.__version__,
        },
        "physics": 0,
        "actors": 0,
        "updates": 0,
        "inference": 0,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    Path(folder, "payload.json").write_text(raw + "\n")
    Finalize(None, emit, args=(folder, "managed_finalizer"), exitpriority=0)
    atexit.register(emit, folder, "python_atexit")
    emit(
        folder,
        "ready",
        payload_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        live=facts(process_identity()),
    )
    conn.send({"payload_sha256": hashlib.sha256(raw.encode()).hexdigest()})
    if not conn.poll(4) or conn.recv() != ("close", None):
        raise TimeoutError("Worker close command")
    emit(folder, "close_received")
    conn.close()
    emit(folder, "before_return", physics=0, updates=0)


def recv(conn, seconds):
    if not conn.poll(seconds):
        raise TimeoutError("Probe response deadline")
    return conn.recv()


def validate_phases(method, events, bound):
    phases = [e["phase"] for e in events]
    if (
        phases
        != [
            "waiting_release",
            "import_begin",
            "ready",
            "close_received",
            "before_return",
            "managed_finalizer",
        ]
        + (["python_atexit"] if method == "spawn" else [])
        or any(e["identity"] != bound for e in events)
        or any(
            a["monotonic_s"] > b["monotonic_s"] for a, b in itertools.pairwise(events)
        )
    ):
        raise ValueError("Incomplete/foreign worker cleanup or exit chronology")
    return phases


def run(folder, slot, packet, ready):
    wait_ready(ready, 15)
    from multiprocessing import forkserver, resource_tracker

    registered = next(s for s in packet["slots"] if s["id"] == slot)
    method, seed = registered["method"], registered["seed"]
    if (slot, method, seed) not in SLOTS or any(n in sys.modules for n in HEAVY):
        raise ValueError("Invalid probe root contract/imports")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if method == "forkserver":
        mp.set_forkserver_preload([])
    ctx = mp.get_context(method)
    conn, child = ctx.Pipe()
    proc = ctx.Process(
        target=worker, args=(child, str(folder), method, seed), daemon=True
    )
    proc.start()
    child.close()
    bound = {"pid": proc.pid, "start": read_identity(proc.pid)["start"]}
    try:
        hello = recv(conn, 4)
        if hello != {"identity": bound, "method": method}:
            raise ValueError("Worker handshake identity/context mismatch")
        worker_before = facts(bound, evidence_dir=folder / "proc-worker-before")
        proof = {
            "worker_before": worker_before,
            "root": facts(process_identity(), evidence_dir=folder / "proc-root-before"),
            "context": method,
            "ipc_context": ctx.get_start_method(),
            "auxiliaries": [],
        }
        if any(
            proof[k]["threads"] != 1
            or proof[k]["after_threads"] != 1
            or proof[k]["heavy_mappings"]
            for k in ("worker_before", "root")
        ):
            raise RuntimeError("Heavy/threaded worker or parent before release")
        if method == "forkserver":
            pid = forkserver._forkserver._forkserver_pid
            server_bound = {"pid": pid, "start": read_identity(pid)["start"]}
            proof["server_before"] = facts(
                server_bound, evidence_dir=folder / "proc-server-before"
            )
            validate_server(proof["server_before"], worker_before, os.getpid())
            proof["auxiliaries"].append(server_bound)
        elif worker_before["parent_pid"] != os.getpid():
            raise RuntimeError("Spawn worker is not owned by root")
        tracker_pid = resource_tracker._resource_tracker._pid
        if tracker_pid:
            proof["auxiliaries"].append(
                {"pid": tracker_pid, "start": read_identity(tracker_pid)["start"]}
            )
        write_json(folder / "before-release.json", proof)
        conn.send("release")
        response = recv(conn, 7)
        payload = json.loads((folder / "payload.json").read_text())
        if payload["versions"] != packet["runtime_versions"] or payload["seed"] != seed:
            raise ValueError("Probe runtime/seed drift")
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if response != {"payload_sha256": hashlib.sha256(raw.encode()).hexdigest()}:
            raise ValueError("Probe payload transport mismatch")
        proof["worker_after"] = facts(bound, evidence_dir=folder / "proc-worker-after")
        if proof["worker_after"]["threads"] != 1:
            raise ValueError("Probe worker threads")
        if method == "forkserver":
            proof["server_after"] = facts(
                server_bound, evidence_dir=folder / "proc-server-after"
            )
            validate_server(proof["server_after"], proof["worker_after"], os.getpid())
        write_json(folder / "after-import.json", proof)

        class Handles:
            def __init__(self):
                self.processes, self.remotes = [proc], [conn]
                self.closed = self.waiting = False

        vec = serial_vec_class(
            Handles, folder / "serial-close.jsonl", packet["close_deadline_s"]
        )()
        vec.close()
        events = [
            json.loads(s)
            for s in (folder / "worker-events.jsonl").read_text().splitlines()
        ]
        phases = validate_phases(method, events, bound)
        close = [
            json.loads(s)
            for s in (folder / "serial-close.jsonl").read_text().splitlines()
        ]
        result = {
            "schema": "joxing-process-exit-probe-v1",
            "method": method,
            "seed": seed,
            "worker": bound,
            "auxiliaries": proof["auxiliaries"],
            "close_to_join_s": close[2]["monotonic_s"] - close[1]["monotonic_s"],
            "payload_sha256": response["payload_sha256"],
            "phases": phases,
            "python_atexit_observed": "python_atexit" in phases,
            "physics": 0,
            "actors": 0,
            "updates": 0,
            "inference": 0,
        }
        write_json(folder / "probe-result.json", result)
        return result
    except BaseException as error:
        write_json(
            folder / "probe-fault.json",
            {"error": type(error).__name__ + ": " + str(error), "worker": bound},
        )
        raise
    finally:
        current = read_identity(proc.pid)
        if (
            proc.is_alive()
            and current is not None
            and current["start"] == bound["start"]
        ):
            proc.terminate()
            proc.join(timeout=0.5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.5)
        conn.close()


def conclude(folders):
    rows = [
        json.loads(Path(folders[name], "probe-result.json").read_text())
        for name, _, _ in SLOTS
    ]
    if [(r["method"], r["seed"]) for r in rows] != [(m, s) for _, m, s in SLOTS]:
        raise ValueError("Incomplete probe denominator")
    if any(
        not math.isfinite(r["close_to_join_s"])
        or not 0 < r["close_to_join_s"] <= 2
        or any(r[k] != 0 for k in ("physics", "actors", "updates", "inference"))
        for r in rows
    ):
        raise ValueError("Invalid probe resource/work facts")
    paired = []
    for seed in (1097, 1098, 1099):
        pair = {r["method"]: r for r in rows if r["seed"] == seed}
        if pair["spawn"]["payload_sha256"] != pair["forkserver"]["payload_sha256"]:
            raise ValueError("Exact paired payload mismatch")
        paired.append(
            {
                "seed": seed,
                "spawn_s": pair["spawn"]["close_to_join_s"],
                "forkserver_s": pair["forkserver"]["close_to_join_s"],
            }
        )
    a, b = (
        statistics.median(r["spawn_s"] for r in paired),
        statistics.median(r["forkserver_s"] for r in paired),
    )
    supported = b <= 0.5 * a and all(r["forkserver_s"] <= r["spawn_s"] for r in paired)
    return {
        "scope": "Import-only exit latency, not native/game/training throughput",
        "pairs": paired,
        "spawn_median_s": a,
        "forkserver_median_s": b,
        "scientific_verdict": "supported_import_only"
        if supported
        else "not_supported_import_only",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k in ("folder", "slot", "packet", "ready", "ledger", "binding"):
        parser.add_argument("--" + k, required=True)
    args = parser.parse_args()
    run(args.folder, args.slot, json.loads(Path(args.packet).read_text()), args.ready)
