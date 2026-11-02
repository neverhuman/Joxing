import gc
import importlib.util
import json
import subprocess
import sys
import weakref
from pathlib import Path

import pytest

from joxing.live_resource_release import live_rss, release_wrapper_class


def test_module_stdlib_until_ready():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;import joxing.release_native_teardown;"
                "assert not {'torch','numpy','mujoco'}&sys.modules.keys()"
            ),
        ],
        check=True,
        timeout=5,
    )


@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "aliases",
        "retained",
        "close_fault",
        "changed",
        "unsupported",
        "rss_fault",
    ],
)
def test_release_lifetime_and_faults(tmp_path, case):
    calls, hidden = [], []

    class Model:
        nu = 700

    class Data:
        pass

    class Core:
        def __init__(self):
            self.model, self.data = Model(), Data()
            self.value = 17
            if case == "aliases":
                self.aliases = [self.model, {"native": self.data}, self]
            if case == "retained":
                hidden.append(self.data)

        @property
        def unwrapped(self):
            return self

        def snapshot(self):
            return self.value

        def close(self):
            calls.append("close")
            if case == "close_fault":
                raise ValueError("Inner close original fault")
            if case == "changed":
                self.value = 18

        def step(self, action):
            return ("step", action)

        def reset(self, **kwargs):
            return kwargs

    class Nested:
        def __init__(self):
            self.env = Core()

        @property
        def unwrapped(self):
            return self.env.unwrapped

        def close(self):
            return self.env.close()

        def step(self, action):
            return self.env.step(action)

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

    class Base:
        def __init__(self, env):
            self.env = env

    count = [0]

    def memory(bound):
        count[0] += 1
        if case == "rss_fault" and count[0] == 2:
            raise RuntimeError("Live RSS read fault")
        return {"rss_bytes": 200 if count[0] == 1 else 150}

    def ref(obj):
        if case == "unsupported":
            raise TypeError("Unsupported weakref")
        return weakref.ref(obj)

    path = tmp_path / "release.jsonl"
    cls = release_wrapper_class(
        Base,
        path,
        state_hash=str,
        bound={"pid": 17, "start": "10"},
        ref_factory=ref,
        read_memory=memory,
    )
    env = cls(Nested())
    assert env.snapshot() == 17 and env.step([0.1]) == ("step", [0.1])
    assert env.reset(seed=5) == {"seed": 5}
    if case in ("normal", "aliases"):
        env.close()
        assert env.env is None and env._release_state == "released"
        before = path.read_bytes()
        env.close()
        assert path.read_bytes() == before
        for f in [env.snapshot, env.reset, lambda: env.step([0.2])]:
            with pytest.raises(RuntimeError, match="Released world"):
                f()
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        assert rows[2]["weakrefs_dead"] == {"env": True, "model": True, "data": True}
        assert rows[2]["observed_rss_change_bytes"] == -50
    else:
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            env.close()
        assert env._release_state == "failed"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        assert rows[-1]["phase"] == "fault"
        if case == "retained":
            assert rows[-2]["weakrefs_dead"]["data"] is False
        if case == "close_fault":
            assert rows[-1]["error"] == "ValueError: Inner close original fault"
        with pytest.raises(RuntimeError):
            env.close()
    hidden.clear()


@pytest.mark.parametrize(
    "case", ["live", "reused", "exiting", "missing", "changed_after"]
)
def test_live_rss_requires_same_live_identity(tmp_path, monkeypatch, case):
    folder = tmp_path / "17"
    folder.mkdir()
    fields = (
        ["R", "2", "0", "0", "0", "0", "4" if case == "exiting" else "0"]
        + ["0"] * 12
        + ["11" if case == "reused" else "10"]
    )
    stat = folder / "stat"
    stat.write_text("17 (fixture) " + " ".join(fields))
    (folder / "status").write_text(
        "Name: fake\n" if case == "missing" else "VmRSS: 123 kB\n"
    )
    original = Path.read_text
    reads = [0]

    def read(p, *args, **kwargs):
        s = original(p, *args, **kwargs)
        if p == stat:
            reads[0] += 1
            if case == "changed_after" and reads[0] == 2:
                return s[:-2] + "11"
        return s

    monkeypatch.setattr(Path, "read_text", read)
    if case == "live":
        assert live_rss({"pid": 17, "start": "10"}, tmp_path)["rss_bytes"] == 123 * 1024
    else:
        with pytest.raises(RuntimeError):
            live_rss({"pid": 17, "start": "10"}, tmp_path)


def test_mujoco_binding_weakrefs_zero_physics(tmp_path):
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    refs = {"model": weakref.ref(model), "data": weakref.ref(data)}
    del data, model
    gc.collect()
    dead = {k: r() is None for k, r in refs.items()}
    assert dead == {"model": True, "data": True}
    (tmp_path / "binding.json").write_text(
        json.dumps(
            {
                "scope": "Toy binding lifetime only, zero physics",
                "mujoco": mujoco.__version__,
                "weakrefs_dead": dead,
                "physics": 0,
            }
        )
        + "\n"
    )


def test_release_supervisor_dispatch_is_explicit():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "qualify_deferred_trial", scripts / "qualify_deferred_trial.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.runner_module("joxing-release-native-teardown-v1")
        == "joxing.release_native_teardown"
    )
    assert (
        module.runner_module("joxing-serial-native-teardown-v1")
        == "joxing.serial_native_teardown"
    )


@pytest.mark.parametrize("fault", ["none", "close_fault", "changed_after_close"])
def test_before_close_receipt_survives_native_close_fault(tmp_path, monkeypatch, fault):
    from stable_baselines3.common import vec_env

    from joxing import release_native_teardown as nt
    from joxing.physics_credit import PhysicsCredit

    folder, credit = tmp_path / "native", tmp_path / "credit.json"
    PhysicsCredit.create(credit, "fixture", 200)
    spec = {
        "model_hash": "fixture-model",
        "action_hash": "fixture-action",
        "observation_hash": "fixture-obs",
    }
    own = {"pid": 17, "start": "10"}
    state = {"state": [1.0], "rng": {"fixture": 1}}

    class Vec:
        def __init__(self, factories, **kwargs):
            assert len(factories) == 1 and kwargs == {"start_method": "spawn"}
            self.processes = []
            (folder / "traces").mkdir()
            (folder / "traces/native-world-0.jsonl").write_text(
                json.dumps(
                    {"owner": own, "semantic": {"phase": "reset", "muscles": 700}}
                )
                + "\n"
            )
            ledger = PhysicsCredit(credit, "fixture")
            token = ledger.reserve(200, "native0", "preparation")
            ledger.commit(token, 200)

        def env_method(self, name):
            return [spec] if name == "specification" else [state]

        def get_attr(self, name):
            return [0 if name == "physics_total" else 200]

        def close(self):
            before = json.loads((folder / "before-close.json").read_text())
            assert before["initial_state_hash"] == nt.value_digest(state)
            if fault == "close_fault":
                raise RuntimeError("Original close fault")
            (folder / "traces/worker-0.jsonl").write_text(
                json.dumps(
                    {
                        "owner": own,
                        "semantic": {"phase": "close", "state": nt.value_digest(state)},
                    }
                )
                + "\n"
            )
            rows = [
                {"phase": p, "identity": own}
                for p in ("before_close", "inner_closed", "graph_checked", "released")
            ]
            rows[0]["state_hash"] = nt.value_digest(state)
            rows[1]["state_hash"] = (
                "wrong" if fault == "changed_after_close" else nt.value_digest(state)
            )
            rows[2]["weakrefs_dead"] = {"env": True, "model": True, "data": True}
            (folder / "resource-release.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )

    monkeypatch.setattr(nt, "wait_ready", lambda *a: None)
    monkeypatch.setattr(nt, "serial_vec_class", lambda base, path, deadline: base)
    monkeypatch.setattr(vec_env, "SubprocVecEnv", Vec)
    args = (
        folder,
        "native0",
        {"seed": 1097, "expected_specification": spec, "close_deadline_s": 2},
        credit,
        "fixture",
        tmp_path / "ready",
    )
    if fault == "none":
        r = nt.run(*args)
        assert (
            r["preparation_steps"] == 200
            and r["resource_release"] == "live-graph-release-v1"
        )
    else:
        with pytest.raises((RuntimeError, ValueError)):
            nt.run(*args)
        assert not (folder / "constructor-result.json").exists()
    assert (folder / "before-close.json").exists()
