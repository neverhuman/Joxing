"""Strict source/protocol/ownership prerequisite checks before D2 physics."""

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from .motor_pilot import SAC_PARAMETERS, TRAINING_SEEDS
from .provenance import ROOT, digest


def pid_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    p = Path("/proc") / str(pid)
    result = {"pid": pid}
    if p.exists():
        stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
        result.update(
            start_ticks=int(stat[19]),
            parent_pid=int(stat[1]),
            rss_bytes=int((p / "statm").read_text().split()[1])
            * os.sysconf("SC_PAGE_SIZE"),
            cwd=str((p / "cwd").resolve()),
        )
    else:
        result["start_ticks"] = None
    return result


def unknown_owned_processes():
    if platform.system() != "Linux":
        raise RuntimeError("Formal D2 requires the qualified Linux source host")
    ancestors = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = pid_identity(pid)["parent_pid"]
        except (OSError, KeyError):
            break
    owned = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        try:
            cwd = str((entry / "cwd").resolve())
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            if "/joxing/" in cwd and ("python" in command or "/uv " in command):
                owned.append(pid_identity(int(entry.name)))
        except OSError:
            pass
    return owned


def validate(p, expected_sha, *, source=True):
    if digest(p) != expected_sha or p["schema"] != "joxing-motor-pilot-D2-0.0.10":
        raise ValueError("Unregistered pilot protocol digest/schema")
    if p["training_seeds"] != list(TRAINING_SEEDS) or digest(p["sac"]) != digest(
        SAC_PARAMETERS
    ):
        raise ValueError("Unregistered learner configuration")
    if p["transitions_per_seed"] != 8192 or p["evaluation_seeds"] != list(
        range(83000, 83008)
    ):
        raise ValueError("Unregistered attempt schedule")
    if (
        p["caps"]["physics_steps"] != 550000
        or p["caps"]["wall_s"] != 1800
        or p["caps"]["raw_bytes"] != 1024**3
    ):
        raise ValueError("Whole-version caps changed")
    if p["audit_caps"] != {"physics_steps": 50000, "wall_s": 300}:
        raise ValueError("Audit reserve changed")
    expected = {
        "parent_bytes": 192 * 1024**2,
        "worker_bytes": 32 * 1024**2,
        "evaluation_bytes": 12 * 1024**2,
        "replay_bytes": 12 * 1024**2,
    }
    if p["raw_credits"] != expected:
        raise ValueError("Raw credit allocation changed")
    charge = (
        expected["parent_bytes"]
        + 8 * expected["worker_bytes"]
        + 32 * expected["evaluation_bytes"]
        + 2 * expected["replay_bytes"]
        + p["prior"]["raw_charge_bytes"]
    )
    if charge > p["caps"]["raw_bytes"] or p["prior"]["physics_steps"] != 10394:
        raise ValueError("Unsafe raw/prior allocation")
    if source:
        committed = json.loads(
            subprocess.check_output(
                ["git", "show", "HEAD:experiments/v0.0.10/protocol.json"], cwd=ROOT
            )
        )
        if digest(committed) != expected_sha:
            raise ValueError("Protocol is not source-frozen at HEAD")
        for name, sha in p["source_files"].items():
            data = (ROOT / name).read_bytes()
            frozen = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != sha or data != frozen:
                raise ValueError("Frozen source drift: " + name)
        for name, sha in p["upstream_files"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                raise ValueError("Pinned upstream drift: " + name)
        d1 = json.loads((ROOT / "artifacts/v0.0.10/D1/raw/result.json").read_text())
        audit = json.loads((ROOT / "artifacts/v0.0.10/D1/audit.json").read_text())
        if (
            not audit["passed"]
            or d1["status"] != "completed"
            or d1["bundle_hash"] != p["bundle_hash"]
        ):
            raise ValueError("Missing D1 qualification")
    return {"protocol_sha256": expected_sha, "maximum_raw_credit_bytes": charge}


def optimizer_hashes(model):
    import torch

    def encode(value):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise FloatingPointError("Nonfinite optimizer state")
            return {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value

    return {
        name: digest(encode(optimizer.state_dict()))
        for name, optimizer in [
            ("actor", model.actor.optimizer),
            ("critic", model.critic.optimizer),
        ]
    }
