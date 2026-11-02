import json
from functools import partial

import pytest


@pytest.mark.parametrize(
    "fault", ["none", "spec", "close_state", "extra_reset", "guard"]
)
def test_four_worker_exact_denominators_and_bad_evidence_rejected(
    tmp_path, monkeypatch, fault
):
    from stable_baselines3.common import vec_env

    from joxing import serial_native_teardown as nt
    from joxing.physics_credit import PhysicsCredit

    folder, credit = tmp_path / "native", tmp_path / "credit.json"
    PhysicsCredit.create(credit, "fixture", 800)
    spec = {
        "model_hash": "fixture-model",
        "action_hash": "fixture-action",
        "observation_hash": "fixture-obs",
    }
    snapshots = [{"fixture": i, "rng": [10 + i]} for i in range(4)]
    calls = []

    class Vec:
        def __init__(self, factories, **kwargs):
            assert kwargs == {"start_method": "spawn"} and len(factories) == 4
            assert [f.args[1] for f in factories] == list(range(4))
            assert all(f.args[2:] == (1097, 1.0) for f in factories)
            self.processes = []
            calls.append("construct")
            (folder / "traces").mkdir()
            ledger = PhysicsCredit(credit, "fixture")
            for i in range(4):
                token = ledger.reserve(200, "native0", "preparation")
                ledger.commit(token, 200)
                row = {
                    "owner": {"pid": 100 + i, "start": str(10 + i)},
                    "semantic": {"phase": "reset", "muscles": 700},
                }
                (folder / f"traces/native-world-{i}.jsonl").write_text(
                    (json.dumps(row) + "\n")
                    * (2 if fault == "extra_reset" and i == 3 else 1)
                )

        def env_method(self, name):
            if name == "snapshot":
                return snapshots
            return [
                spec | ({"model_hash": "wrong"} if fault == "spec" and i == 2 else {})
                for i in range(4)
            ]

        def get_attr(self, name):
            return [0 if name == "physics_total" else 200] * 4

        def close(self):
            calls.append("close")
            for i, s in enumerate(snapshots):
                (folder / f"traces/worker-{i}.jsonl").write_text(
                    json.dumps(
                        {
                            "owner": {"pid": 100 + i, "start": str(10 + i)},
                            "semantic": {
                                "phase": "close",
                                "state": "wrong"
                                if fault == "close_state" and i == 3
                                else nt.value_digest(s),
                            },
                        }
                    )
                    + "\n"
                )
            if fault == "guard":
                (folder / "guard-events").mkdir()
                (folder / "guard-events/fault.jsonl").write_text("{}\n")

        def reset(self):
            pytest.fail("No extra reset allowed")

        def step(self, action):
            pytest.fail("No rollout allowed")

    monkeypatch.setattr(nt, "wait_ready", lambda *a: None)
    monkeypatch.setattr(nt, "serial_vec_class", lambda base, path, deadline: base)
    monkeypatch.setattr(vec_env, "SubprocVecEnv", Vec)
    call = partial(
        nt.run,
        folder,
        "native0",
        {"seed": 1097, "expected_specification": spec, "close_deadline_s": 2},
        credit,
        "fixture",
        tmp_path / "ready",
    )
    if fault == "none":
        r = call()
        assert r["preparation_steps"] == 800 and len(r["workers"]) == 4
        assert (
            r["actors_constructed"]
            == r["rollout_steps"]
            == r["learner_updates"]
            == r["policy_inferences"]
            == 0
        )
    else:
        with pytest.raises(ValueError):
            call()
    assert calls == ["construct", "close"]


def test_supervisor_dispatch_preserves_existing_schema_commands():
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "qualify_deferred_trial", scripts / "qualify_deferred_trial.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.runner_module("joxing-deferred-native-parity-v1")
        == "joxing.deferred_qualification"
    )
    assert module.runner_module("joxing-native-teardown-v1") == "joxing.native_teardown"
    assert (
        module.runner_module("joxing-serial-native-teardown-v1")
        == "joxing.serial_native_teardown"
    )
    with pytest.raises(KeyError):
        module.runner_module("future-unregistered")
