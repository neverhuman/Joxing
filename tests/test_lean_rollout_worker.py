import ast
import hashlib
import inspect
import json
import pickle
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import pytest

from joxing import lean_rollout_worker as lean
from joxing.lean_worker_probe import make_fake, portable

ROOT = Path(__file__).resolve().parents[1]


def test_cold_import_does_not_import_heavy_libraries():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;import joxing.lean_rollout_worker;import joxing.lean_worker_probe;assert not {'numpy','gymnasium','torch','mujoco','stable_baselines3'}&sys.modules.keys()",
        ],
        check=True,
        timeout=5,
    )


def normalize_body(nodes):
    for n in nodes:
        for item in ast.walk(n):
            if isinstance(item, ast.AnnAssign):
                item.annotation = ast.Constant(value="TYPE_ONLY")
    return "\n".join(ast.dump(n, include_attributes=False) for n in nodes)


def test_pipe_operations_match_pinned_271_body_exactly():
    target = ast.parse(inspect.getsource(lean.lean_pipe_worker)).body[0]
    actual = hashlib.sha256(normalize_body(target.body).encode()).hexdigest()
    expected = json.loads(
        (
            ROOT / "artifacts/v0.0.11/development/lean-worker-pinned-source-01.json"
        ).read_text()
    )["worker_body_without_helper_import_annotation_normalized_sha256"]
    assert actual == expected


def test_installed_baseline_is_pinned_on_linux_and_operation_compatible_locally():
    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env.subproc_vec_env import _worker

    if sys.platform == "linux":
        assert sb3.__version__ == "2.7.1"
    original = ast.parse(inspect.getsource(_worker)).body[0]
    candidate = ast.parse(inspect.getsource(lean.lean_pipe_worker)).body[0]
    assert normalize_body(original.body[1:]) == normalize_body(candidate.body)


def test_pickle_wrapper_roundtrip_keeps_closure_without_sb3_reference():
    wrapped = lean.LeanCloudpickleWrapper(lambda: 19)
    assert pickle.loads(pickle.dumps(wrapped)).var() == 19
    code = "import pickle,sys;from joxing.lean_rollout_worker import LeanCloudpickleWrapper;v=pickle.loads(bytes.fromhex(sys.argv[1]));assert v.var()==19;assert not {'torch','stable_baselines3','mujoco'}&sys.modules.keys()"
    subprocess.run(
        [sys.executable, "-c", code, pickle.dumps(wrapped).hex()], check=True, timeout=5
    )


@pytest.mark.parametrize("value", [None, object(), 3])
def test_explicit_gymnasium_only_profile_rejects_unsupported_environment(value):
    with pytest.raises(TypeError, match="Gymnasium"):
        lean._patch_env(value)


def test_nested_wrapper_search_keeps_original_semantics():
    env = make_fake(1097)
    assert lean._patch_env(env) is env
    assert lean.unwrap_wrapper(env, gym.Wrapper) is env
    assert lean.is_wrapped(env, gym.Wrapper)
    assert not lean.is_wrapped(env, gym.ObservationWrapper)
    assert not lean.is_wrapped(env.unwrapped, gym.Wrapper)


class Parent:
    def close(self):
        pass


class Pipe:
    def __init__(self, cmds):
        self.cmds = iter(cmds)
        self.sent = []
        self.closed = False

    def recv(self):
        return next(self.cmds)

    def send(self, item):
        self.sent.append(portable(item))

    def close(self):
        self.closed = True


@pytest.mark.parametrize("fault", ["unknown", "eof", "interrupt", "step_error"])
def test_error_eof_and_interrupt_match_original_behavior(fault):
    from stable_baselines3.common.vec_env.subproc_vec_env import _worker

    outcomes = []
    for target in (_worker, lean.lean_pipe_worker):
        pipe = Pipe(
            [("not_a_command", None)] if fault == "unknown" else [("step", None)]
        )
        if fault in ("eof", "interrupt"):

            def failure():
                raise EOFError() if fault == "eof" else KeyboardInterrupt()

            pipe.recv = failure
        try:
            target(pipe, Parent(), lean.LeanCloudpickleWrapper(lambda: make_fake(1097)))
        except (TypeError, NotImplementedError) as error:
            outcomes.append((type(error).__name__, str(error), pipe.sent, pipe.closed))
        else:
            outcomes.append(("returned", None, pipe.sent, pipe.closed))
    assert outcomes[0] == outcomes[1]
