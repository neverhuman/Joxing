"""Bounded, zero-native pipe/import probe. Stdlib only before readiness."""

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from multiprocessing.util import Finalize
from pathlib import Path
from typing import ClassVar

from .deferred_qualification import wait_ready
from .lean_rollout_worker import LeanCloudpickleWrapper, lean_pipe_worker
from .native_context import NativeContextProof, finalizer_witness
from .physics_credit import process_identity
from .process_exit_probe import facts, recv, validate_server, write_json
from .serial_retirement import read_identity, serial_vec_class

SCHEMA = "joxing-lean-rollout-worker-probe-v1"
SLOTS = [
    ("sb3A", "sb3", 1097),
    ("leanA", "lean", 1097),
    ("sb3B", "sb3", 1098),
    ("leanB", "lean", 1098),
]
FORBIDDEN = ("torch", "stable_baselines3", "mujoco")


def portable(value):
    """Lossless typed semantics including array dtype/shape and signed zero."""
    import gymnasium as gym
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError("Unsupported/nonfinite array")
        return {
            "array": {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "hex": np.ascontiguousarray(value).tobytes().hex(),
            }
        }
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, gym.spaces.Box):
        return {
            "Box": {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "low": portable(value.low),
                "high": portable(value.high),
            }
        }
    if isinstance(value, gym.spaces.Dict):
        return {"DictSpace": [[k, portable(v)] for k, v in value.spaces.items()]}
    if isinstance(value, type):
        return {"type": value.__module__ + "." + value.__qualname__}
    if isinstance(value, dict):
        return {"dict": [[portable(k), portable(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [portable(v) for v in value]}
    if type(value) is float:
        import math

        if not math.isfinite(value):
            raise ValueError("Nonfinite response")
        return {"float": value.hex()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError("Unsupported pipe semantic type " + type(value).__name__)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def imports():
    return {
        name: any(k == name or k.startswith(name + ".") for k in sys.modules)
        for name in FORBIDDEN
    }


def witness(folder, server, profile, phase):
    bound = process_identity()
    proof = facts(bound, evidence_dir=Path(folder) / ("proc-" + phase))
    status = Path(folder, "proc-" + phase, "status.txt").read_text()
    memory = (
        int(
            next(
                line.split()[1]
                for line in status.splitlines()
                if line.startswith("VmRSS:")
            )
        )
        * 1024
    )
    modules = sorted(sys.modules)
    module_path = Path(folder) / ("modules-" + phase + ".json")
    write_json(module_path, modules)
    value = {
        "method": mp.get_start_method(),
        "worker": bound,
        "server": server,
        "profile": profile,
        "phase": phase,
        "proc": proof,
        "rss_bytes": memory,
        "imports": imports(),
        "modules_sha256": digest(modules),
        "modules_path": str(module_path),
        "monotonic_s": time.monotonic(),
    }
    if (
        value["method"] != "forkserver"
        or proof["parent_pid"] != server["pid"]
        or proof["threads"] != 1
        or proof["after_threads"] != 1
    ):
        raise ValueError("Fake worker context/thread drift")
    if profile == "lean" and (
        any(value["imports"].values())
        or "libtorch" in proof["heavy_mappings"]
        or "mujoco" in proof["heavy_mappings"]
    ):
        raise ValueError("Heavy import in lean worker")
    if profile == "sb3" and (
        not value["imports"]["torch"]
        or not value["imports"]["stable_baselines3"]
        or value["imports"]["mujoco"]
    ):
        raise ValueError("Baseline import contract")
    write_json(Path(folder) / (phase + ".json"), value)
    return value


def make_fake(seed, *, folder=None, server=None, profile=None):
    import gymnasium as gym
    import numpy as np

    class Fake(gym.Env):
        metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}
        render_mode = "rgb_array"

        def __init__(self):
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Box(-100, 100, (2,), np.float32),
                    "draw": gym.spaces.Box(0, 1, (1,), np.float32),
                }
            )
            self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
            self.rng = np.random.default_rng(seed)
            self.tick = self.resets = 0
            self.mode = "none"
            self.value = 3
            self.last_draw = 0.0

        def obs(self):
            return {
                "state": np.array([self.tick, -0.0], dtype=np.float32),
                "draw": np.array([self.last_draw], dtype=np.float32),
            }

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.tick = 0
            self.resets += 1
            self.mode = (options or {}).get("mode", "none")
            self.last_draw = float(self.rng.random())
            return self.obs(), {
                "reset": self.resets,
                "mode": self.mode,
                "options": copy.deepcopy(options),
            }

        def step(self, action):
            self.tick += 1
            self.last_draw = float(self.rng.random())
            done = self.tick >= 2
            return (
                self.obs(),
                float(action[0]) + 0.125 * self.tick + self.last_draw,
                done and self.mode in ("terminated", "both"),
                done and self.mode in ("truncated", "both"),
                {"nested": {"tick": self.tick}, "mode": self.mode},
            )

        def snapshot(self):
            return {
                "tick": self.tick,
                "resets": self.resets,
                "mode": self.mode,
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                "value": self.value,
                "observation": self.obs(),
            }

        def echo(self, x, *, add):
            return {
                "array": np.array([x + add, -0.0], dtype=np.float32),
                "nested": {"value": self.value},
            }

        def diagnostic(self):
            return witness(folder, server, profile, "after_commands")

        def import_snapshot(self):
            return imports()

        def render(self):
            return np.full((2, 2, 3), self.tick, dtype=np.uint8)

        def close(self):
            if folder is not None:
                write_json(
                    Path(folder) / "closed-state.json", portable(self.snapshot())
                )

    env = gym.Wrapper(gym.Wrapper(Fake()))
    if folder is not None:
        witness(folder, server, profile, "fake_initialized")
        Finalize(
            None, finalizer_witness, args=(folder, process_identity()), exitpriority=5
        )
    return env


def commands(seed):
    import gymnasium as gym
    import numpy as np

    action = np.array([0.25, -0.5], dtype=np.float32)
    result = [
        ("get_spaces", None),
        ("reset", (seed, None)),
        ("has_attr", "value"),
        ("has_attr", "missing"),
        ("get_attr", "value"),
        ("set_attr", ("value", 17)),
        ("get_attr", "value"),
        ("env_method", ("echo", (4,), {"add": 2})),
        ("is_wrapped", gym.Wrapper),
        ("is_wrapped", gym.ObservationWrapper),
        ("render", None),
        ("step", action),
    ]
    for mode in ("terminated", "truncated", "both"):
        result += [
            ("reset", (seed, {"mode": mode})),
            ("step", action),
            ("step", action),
            ("step", action),
        ]
    result += [("reset", (seed, {})), ("render", None)]
    return result


def transact(conn, command):
    conn.send(command)
    return recv(conn, 3)


def exercise(conn, seed):
    result = []
    for cmd in commands(seed):
        before = transact(conn, ("env_method", ("snapshot", (), {})))
        answer = transact(conn, cmd)
        # Mutating a received value must not alias the remote environment.
        encoded = portable(answer)
        if cmd[0] == "env_method":
            answer["nested"]["value"] = -999
        after = transact(conn, ("env_method", ("snapshot", (), {})))
        result.append(
            {
                "command": portable(cmd),
                "response": encoded,
                "before": portable(before),
                "after": portable(after),
            }
        )
    return result


def run(folder, slot, packet, ready):
    if (
        packet["schema"] != SCHEMA
        or packet["physics_limit"] != 0
        or packet["simulation_workers"] != 1
        or [(s["id"], s["profile"], s["seed"]) for s in packet["slots"]] != SLOTS
    ):
        raise ValueError("Explicit zero-native four-row probe required")
    wait_ready(ready, 15)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    proof = NativeContextProof(Path(str(folder) + "-context"))
    proof.prepare()
    registered = next(s for s in packet["slots"] if s["id"] == slot)
    profile, seed = registered["profile"], registered["seed"]
    # Same heavy parent in both variants; only actual child footprint compared.
    from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
    from stable_baselines3.common.vec_env.subproc_vec_env import _worker

    worker = lean_pipe_worker if profile == "lean" else _worker
    wrapper = LeanCloudpickleWrapper if profile == "lean" else CloudpickleWrapper
    ctx = mp.get_context("forkserver")
    conn, child = ctx.Pipe()
    proc = ctx.Process(
        target=worker,
        args=(
            child,
            conn,
            wrapper(
                partial(
                    make_fake,
                    seed,
                    folder=str(folder),
                    server=proof.server,
                    profile=profile,
                )
            ),
        ),
        daemon=True,
    )
    proc.start()
    child.close()
    bound = {"pid": proc.pid, "start": read_identity(proc.pid)["start"]}
    try:
        # Initial get_spaces is part of the transcript; wait for factory first.
        end = time.monotonic() + 8
        while not (folder / "fake_initialized.json").exists():
            if not proc.is_alive() or time.monotonic() > end:
                raise TimeoutError("Fake initialization")
            time.sleep(0.005)
        initial = json.loads((folder / "fake_initialized.json").read_text())
        if initial["worker"] != bound:
            raise ValueError("Fake worker identity")
        semantic = exercise(conn, seed)
        after = transact(conn, ("env_method", ("diagnostic", (), {})))
        server_after = facts(
            proof.server, evidence_dir=proof.folder / "proc-server-after"
        )
        validate_server(server_after, after["proc"], os.getpid())
        proof.proof.update(
            worker=bound,
            worker_before=initial,
            worker_after=after,
            server_after=server_after,
        )
        write_json(proof.folder / "after.json", proof.proof)
        write_json(folder / "semantic.json", semantic)

        class Handles:
            def __init__(self):
                self.processes, self.remotes, self.closed, self.waiting = (
                    [proc],
                    [conn],
                    False,
                    False,
                )

        vec = serial_vec_class(
            Handles, folder / "serial-close.jsonl", packet["close_deadline_s"]
        )()
        vec.close()
        if (
            json.loads((folder / "closed-state.json").read_text())
            != semantic[-1]["after"]
        ):
            raise ValueError("Fake close changed final state/RNG")
        close = [
            json.loads(s)
            for s in (folder / "serial-close.jsonl").read_text().splitlines()
        ]
        final = [
            json.loads(s)
            for s in (folder / "context-finalizer.jsonl").read_text().splitlines()
        ]
        if (
            len(final) != 1
            or final[0]["identity"] != bound
            or not close[1]["monotonic_s"]
            <= final[0]["monotonic_s"]
            <= close[2]["monotonic_s"]
            or proc.exitcode != 0
        ):
            raise ValueError("Fake finalizer/close proof")
        result = {
            "schema": SCHEMA,
            "id": slot,
            "profile": profile,
            "seed": seed,
            "worker": bound,
            "context": proof.proof,
            "semantic_sha256": digest(semantic),
            "commands": len(semantic),
            "closed_state_sha256": digest(
                json.loads((folder / "closed-state.json").read_text())
            ),
            "worker_rss_before": initial["rss_bytes"],
            "worker_rss_after": after["rss_bytes"],
            "close_to_join_s": close[2]["monotonic_s"] - close[1]["monotonic_s"],
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
        if proc.is_alive():
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
    if [(r["id"], r["profile"], r["seed"]) for r in rows] != SLOTS:
        raise ValueError("Four-row denominator")
    pairs = []
    for label in ("A", "B"):
        a, b = (
            next(r for r in rows if r["id"] == "sb3" + label),
            next(r for r in rows if r["id"] == "lean" + label),
        )
        left = json.loads(Path(folders[a["id"]], "semantic.json").read_text())
        right = json.loads(Path(folders[b["id"]], "semantic.json").read_text())
        if (
            left != right
            or digest(left) != digest(right)
            or digest(left) != a["semantic_sha256"]
            or digest(right) != b["semantic_sha256"]
            or a["closed_state_sha256"] != b["closed_state_sha256"]
        ):
            raise ValueError("Exact fake command/state/RNG pair mismatch")
        if len(left) != 26 or any(
            any(r[k] != 0 for k in ("physics", "actors", "updates", "inference"))
            for r in (a, b)
        ):
            raise ValueError("Fake command/work denominator")
        ratio = max(b["worker_rss_before"], b["worker_rss_after"]) / min(
            a["worker_rss_before"], a["worker_rss_after"]
        )
        if ratio > 0.75:
            raise ValueError("Paired worker RSS footprint gate")
        pairs.append(
            {
                "pair": label,
                "rss_ratio_upper_bound": ratio,
                "commands": len(left),
                "exact_semantics": True,
            }
        )
    return {
        "qualified_fake_ipc_only": True,
        "pairs": pairs,
        "native_or_learner_parity": False,
        "scope": "Zero-native fake command/import/worker RSS, no training/game throughput",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("folder", "slot", "packet", "ready", "ledger", "binding"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    from .lean_worker_probe import run as execute

    execute(
        args.folder, args.slot, json.loads(Path(args.packet).read_text()), args.ready
    )
