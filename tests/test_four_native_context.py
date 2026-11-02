import json
import subprocess
import sys
from pathlib import Path

import pytest

from joxing import four_native_context as four


def test_four_module_stdlib_until_readiness():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;import joxing.four_native_context;assert not {'numpy','torch','mujoco','stable_baselines3'}&sys.modules.keys()",
        ],
        check=True,
        timeout=5,
    )


def test_one_worker_packet_cannot_enter_four_variant(monkeypatch):
    monkeypatch.setattr(
        four, "wait_ready", lambda *a: pytest.fail("Do not start wrong packet")
    )
    with pytest.raises(ValueError):
        four.run(
            None,
            None,
            {"schema": "joxing-native-context-transfer-v1", "simulation_workers": 1},
            None,
            None,
            None,
        )


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def release_files(folder, i, own, state, *, weak=True, finalizer=True):
    world = folder / f"world-{i}"
    start = 100 + i * 0.1
    memory = {
        "rss_bytes": 123456,
        "before": own | {"state": "R", "flags": 0},
        "after": own | {"state": "S", "flags": 0},
    }
    phases = [
        {"phase": p, "identity": own, "monotonic_s": start + 0.01 * j}
        for j, p in enumerate(
            ["before_close", "inner_closed", "graph_checked", "released"]
        )
    ]
    phases[0].update(state_hash=state, muscles=700, memory=memory)
    phases[1]["state_hash"] = state
    phases[2].update(
        observed_rss_change_bytes=0,
        weakrefs_dead={"env": True, "model": True, "data": weak},
        memory=memory,
    )
    dump(world / "resource-release.jsonl", phases)
    dump(
        world / "context-finalizer.jsonl",
        [{"phase": "managed_finalizer", "identity": own, "monotonic_s": start + 0.04}]
        if finalizer
        else [],
    )
    dump(
        world / f"traces/worker-{i}.jsonl",
        [
            {
                "owner": own,
                "semantic": {"phase": "close", "state": state},
                "monotonic_s": start + 0.005,
            }
        ],
    )


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "state",
        "spec",
        "prep",
        "second_close",
        "deadline",
        "weakref",
        "finalizer",
        "thread",
        "context",
    ],
)
def test_four_parent_receipt_faults_and_shared_close(tmp_path, monkeypatch, fault):
    from stable_baselines3.common import vec_env

    from joxing.physics_credit import PhysicsCredit

    folder, ledger = tmp_path / "native0", tmp_path / "credit.json"
    PhysicsCredit.create(ledger, "fixed", 800)
    state, specification = (
        {"physical": [0.2]},
        {"model_hash": "model", "action_hash": "action", "observation_hash": "obs"},
    )
    identities = [{"pid": 21 + i, "start": str(3 + i)} for i in range(4)]

    class Proof:
        def __init__(self, path):
            self.folder, self.server = Path(path), {"pid": 17, "start": "2"}
            self.proof = {"root": {"pid": 11, "start": "1"}}

        def prepare(self):
            self.folder.mkdir()

    def capture(bound, **kwargs):
        return {
            "identity": bound,
            "threads": 2 if fault == "thread" else 1,
            "after_threads": 1,
            "heavy_mappings": [],
            "parent_pid": 11 if bound["pid"] == 17 else 17,
        }

    monkeypatch.setattr(four, "NativeContextProof", Proof)
    monkeypatch.setattr(four, "wait_ready", lambda *a: None)
    monkeypatch.setattr(four, "facts", capture)
    monkeypatch.setattr(four, "bound_process", lambda pid: identities[pid - 21])

    def adapter(base, path, deadline):
        assert deadline == 2
        return base  # Serial adapter's actual deadline/ownership tests run separately.

    monkeypatch.setattr(four, "serial_vec_class", adapter)

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def is_alive(self):
            return False

        def join(self, timeout):
            pass

    class Vec:
        def __init__(self, factories, *, start_method):
            assert start_method == "forkserver" and len(factories) == 4
            self.processes = [Process(o["pid"]) for o in identities]
            for i, factory in enumerate(factories):
                assert factory.args[1] == i and factory.keywords["folder"] == str(
                    folder / f"world-{i}"
                )
                world = folder / f"world-{i}"
                proc = capture(identities[i])
                witness = {
                    "method": "spawn"
                    if fault == "context" and i == 2
                    else "forkserver",
                    "worker": identities[i],
                    "server": {"pid": 17, "start": "2"},
                    "proc": proc,
                }
                dump(world / "context-worker-before.json", [witness])
                dump(
                    world / f"traces/native-world-{i}.jsonl",
                    [
                        {
                            "owner": identities[i],
                            "semantic": {"phase": "reset", "muscles": 700},
                        }
                    ],
                )
                credit = PhysicsCredit(ledger, "fixed")
                token = credit.reserve(200, "native0", "preparation")
                credit.commit(token, 200)

        def env_method(self, name):
            values = (
                [specification.copy() for _ in range(4)]
                if name == "specification"
                else [state.copy() for _ in range(4)]
            )
            if fault == ("spec" if name == "specification" else "state"):
                values[2] = {"wrong": True}
            return values

        def get_attr(self, name):
            return [
                199
                if fault == "prep" and i == 2 and name == "preparation_total"
                else (200 if name == "preparation_total" else 0)
                for i in range(4)
            ]

        def close(self):
            before = (
                json.loads((folder / "before-close.json").read_text())
                if (folder / "before-close.json").exists()
                else None
            )
            if fault not in ("thread", "context"):
                assert (
                    before
                    and len(before["workers"])
                    == len(before["initial_and_close_state_hashes"])
                    == 4
                )
            events = [{"phase": "bound", "monotonic_s": 99}]
            for i, own in enumerate(identities):
                if fault == "second_close" and i == 1:
                    raise TimeoutError("Original later-worker close fault")
                release_files(
                    folder,
                    i,
                    own,
                    four.value_digest(state),
                    weak=not (fault == "weakref" and i == 2),
                    finalizer=not (fault == "finalizer" and i == 2),
                )
                events.extend(
                    [
                        {
                            "phase": "close_send",
                            "worker": i,
                            "identity": own,
                            "monotonic_s": 99.99 + i * 0.1,
                        },
                        {
                            "phase": "joined",
                            "worker": i,
                            "identity": own,
                            "exitcode": 0,
                            "monotonic_s": (
                                102.5
                                if fault == "deadline" and i == 3
                                else 100.05 + i * 0.1
                            ),
                        },
                    ]
                )
            events.append({"phase": "closed", "monotonic_s": 103})
            dump(folder / "serial-close.jsonl", events)

    monkeypatch.setattr(vec_env, "SubprocVecEnv", Vec)
    p = {
        "schema": "joxing-four-native-context-v1",
        "simulation_workers": 4,
        "seed": 1097,
        "expected_specification": specification,
        "expected_state_hash": four.value_digest(state),
        "close_deadline_s": 2,
    }
    if fault == "none":
        result = four.run(folder, "native0", p, ledger, "fixed", "ready")
        assert result["preparation_steps"] == 800 and len(result["worlds"]) == 4
    else:
        with pytest.raises((ValueError, RuntimeError)):
            four.run(folder, "native0", p, ledger, "fixed", "ready")
        assert (folder / "constructor-fault.json").exists()
        assert not (folder / "constructor-result.json").exists()
    if fault not in ("thread", "context"):
        before = json.loads((folder / "before-close.json").read_text())
        assert before["per_worker_preparation_steps"] == (
            [200, 200, 199, 200] if fault == "prep" else [200] * 4
        )
    assert (
        PhysicsCredit(ledger, "fixed").snapshot()["used"] == 800
    )  # Mocked, no native physics.
