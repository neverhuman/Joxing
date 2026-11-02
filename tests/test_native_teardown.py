import json
import subprocess
import sys
from functools import partial

import pytest


def test_constructor_module_stdlib_until_readiness():
    code = "import sys;import joxing.native_teardown;assert not {'numpy','torch','mujoco'}&sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5)


@pytest.mark.parametrize("wrong_spec", [False, True])
def test_one_worker_close_order_and_no_rollout(tmp_path, monkeypatch, wrong_spec):
    from stable_baselines3.common import vec_env

    from joxing import native_teardown as nt
    from joxing.physics_credit import PhysicsCredit

    calls = []
    state = {"state": [0.02], "rng": {"fixture": 1}}
    specification = {
        "model_hash": "fixture-model",
        "action_hash": "fixture-actions",
        "observation_hash": "fixture-observations",
    }
    ledger_path = tmp_path / "credit.json"
    PhysicsCredit.create(ledger_path, "fixed", 200)
    folder = tmp_path / "native"

    class Vec:
        def __init__(self, factories, **kwargs):
            assert len(factories) == 1 and kwargs == {"start_method": "spawn"}
            calls.append("construct")
            self.processes = []
            token = PhysicsCredit(ledger_path, "fixed").reserve(
                200, "native0", "preparation"
            )
            PhysicsCredit(ledger_path, "fixed").commit(token, 200)
            traces = folder / "traces"
            traces.mkdir()
            (traces / "native-world-0.jsonl").write_text(
                json.dumps(
                    {
                        "owner": {"pid": 17, "start": "10"},
                        "semantic": {"phase": "reset", "muscles": 700},
                    }
                )
                + "\n"
            )

        def env_method(self, name):
            return [specification] if name == "specification" else [state]

        def get_attr(self, name):
            return [0 if name == "physics_total" else 200]

        def close(self):
            calls.append("close")
            (folder / "traces/worker-0.jsonl").write_text(
                json.dumps(
                    {"semantic": {"phase": "close", "state": nt.value_digest(state)}}
                )
                + "\n"
            )

        def step(self, *args):
            pytest.fail("Unexpected rollout")

        def reset(self, *args):
            pytest.fail("Unexpected reset")

    monkeypatch.setattr(nt, "wait_ready", lambda *a: None)
    monkeypatch.setattr(vec_env, "SubprocVecEnv", Vec)
    packet = {
        "seed": 1097,
        "expected_specification": specification
        | ({"model_hash": "wrong"} if wrong_spec else {}),
    }
    call = partial(
        nt.run, folder, "native0", packet, ledger_path, "fixed", tmp_path / "ready"
    )
    if wrong_spec:
        with pytest.raises(ValueError, match="specification drift"):
            call()
    else:
        result = call()
        assert (
            result["preparation_steps"] == 200
            and result["actors_constructed"]
            == result["learner_updates"]
            == result["rollout_steps"]
            == 0
        )
    assert calls == ["construct", "close"]
