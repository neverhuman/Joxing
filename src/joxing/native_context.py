"""Stdlib-only, explicitly scoped native forkserver proof; no default changes."""

import json
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from pathlib import Path

from .physics_credit import process_identity
from .process_exit_probe import HEAVY, facts, validate_server, write_json
from .serial_retirement import read_identity


def bound_process(pid):
    value = read_identity(pid)
    if value is None or value["state"] in ("Z", "X"):
        raise RuntimeError("Native context process not live")
    return {"pid": pid, "start": value["start"]}


def worker_witness(folder, server):
    """SB3 has imported libraries; this witness precedes native env creation."""
    folder = Path(folder)
    bound = process_identity()
    witness = {
        "method": mp.get_start_method(),
        "worker": bound,
        "server": server,
        "phase": "before_native_constructor_not_before_library_imports",
        "monotonic_s": time.monotonic(),
        "proc": facts(bound, evidence_dir=folder / "proc-context-worker-before"),
    }
    write_json(folder / "context-worker-before.json", witness)
    if (
        witness["method"] != "forkserver"
        or witness["proc"]["parent_pid"] != server["pid"]
    ):
        raise RuntimeError("Native worker context/ancestry mismatch")
    if witness["proc"]["threads"] != 1 or witness["proc"]["after_threads"] != 1:
        raise RuntimeError("Native worker context threads")
    return witness


def finalizer_witness(folder, bound):
    with (Path(folder) / "context-finalizer.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "phase": "managed_finalizer",
                    "identity": bound,
                    "monotonic_s": time.monotonic(),
                }
            )
            + "\n"
        )


def constructor_worker(*args, folder, server, **kwargs):
    from multiprocessing.util import Finalize

    worker_witness(folder, server)
    Finalize(None, finalizer_witness, args=(folder, process_identity()), exitpriority=5)
    from .release_native_teardown import constructor_worker as release_worker

    return release_worker(*args, folder=folder, **kwargs)


class NativeContextProof:
    method = "forkserver"

    def __init__(self, folder):
        self.folder = Path(folder)
        self.server = None
        self.proof = None

    def prepare(self):
        if any(name in sys.modules for name in HEAVY):
            raise RuntimeError("Native context root must start stdlib-only")
        from multiprocessing import forkserver

        if forkserver._forkserver._forkserver_pid is not None:
            raise RuntimeError("Native context requires fresh server")
        self.folder.mkdir(parents=True, exist_ok=False)
        root = process_identity()
        root_before = facts(root, evidence_dir=self.folder / "proc-root-before")
        if (
            root_before["threads"] != 1
            or root_before["after_threads"] != 1
            or root_before["heavy_mappings"]
        ):
            raise RuntimeError("Native context dirty root")
        forkserver.set_forkserver_preload([])
        if forkserver._forkserver._preload_modules != []:
            raise RuntimeError("Native context preload changed")
        forkserver.ensure_running()
        self.server = bound_process(forkserver._forkserver._forkserver_pid)
        server_before = facts(
            self.server, evidence_dir=self.folder / "proc-server-before"
        )
        # A not-yet-launched child placeholder checks only server/root properties.
        validate_server(server_before, {"parent_pid": self.server["pid"]}, os.getpid())
        self.proof = {
            "schema": "joxing-native-context-proof-v1",
            "method": self.method,
            "root": root,
            "root_before": root_before,
            "server": self.server,
            "server_before": server_before,
            "preload": [],
        }
        write_json(self.folder / "before.json", self.proof)

    def factory(self):
        if self.proof is None:
            raise RuntimeError("Native context not prepared")
        return partial(constructor_worker, server=self.server)

    def after_construct(self, vec):
        if self.proof is None or len(vec.processes) != 1:
            raise RuntimeError("Native context worker denominator")
        worker = bound_process(vec.processes[0].pid)
        witness_path = (
            self.folder.parent
            / self.folder.name.removesuffix("-context")
            / "context-worker-before.json"
        )
        witness = json.loads(witness_path.read_text())
        if (
            witness["worker"] != worker
            or witness["server"] != self.server
            or witness["method"] != self.method
        ):
            raise RuntimeError("Native context witness identity drift")
        after = facts(worker, evidence_dir=self.folder / "proc-worker-after")
        server_after = facts(
            self.server, evidence_dir=self.folder / "proc-server-after"
        )
        validate_server(server_after, after, self.proof["root"]["pid"])
        if after["threads"] != 1 or after["after_threads"] != 1:
            raise RuntimeError("Native context worker after threads")
        self.proof.update(
            worker=worker,
            worker_before=witness,
            worker_after=after,
            server_after=server_after,
        )
        write_json(self.folder / "after.json", self.proof)

    def qualify(self, result, native_folder):
        if self.proof is None or self.proof.get("worker") != result["worker"]:
            raise ValueError("Native context result worker mismatch")
        rows = [
            json.loads(s)
            for s in (Path(native_folder) / "context-finalizer.jsonl")
            .read_text()
            .splitlines()
        ]
        if (
            len(rows) != 1
            or rows[0]["phase"] != "managed_finalizer"
            or rows[0]["identity"] != result["worker"]
        ):
            raise ValueError("Native context managed finalizer missing")
        if rows[0]["monotonic_s"] < result["release_phases"][-1]["monotonic_s"]:
            raise ValueError("Native context finalizer before release")
        return self.proof | {"finalizer": rows[0]}
