import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from joxing.boxing_evidence import controller_bundles
from joxing.motor_storage import write_json
from joxing.provenance import digest
from joxing.scene import build_scene
from joxing.sensor_protocol import sensor_cases, validate_packet


def packet():
    return {
        "schema": "joxing-sensor-proof-protocol-v1",
        "cases": sensor_cases(),
        "bundles": controller_bundles(),
        "planned_physics": 2748,
        "physics_cap": 10000,
        "raw_bytes": 96 * 1024**2,
        "delays_ticks": {"A": 1, "B": 2},
        "runner_wall_s": 180,
        "minimum_memory_bytes": 8 * 1024**3,
        "minimum_disk_bytes": 10 * 1024**3,
    }


def load_script(name, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        name, root / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_sensor_schedule_and_replay_denominators():
    p = packet()
    validate_packet(p)
    assert len({c["id"] for c in p["cases"]}) == 8
    assert sum(c["physics"] for c in p["cases"]) == 2748
    assert all(c["physics"] == c["end_tick"] - c["start_tick"] for c in p["cases"])
    assert [c["start_tick"] for c in p["cases"] if "parent" in c] == [203, 83, 83, 83]


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "reorder",
        "invalid",
        "incomplete",
        "row_physics",
        "confirmed",
        "global_physics",
        "learned_game",
        "case_seed",
    ],
)
def test_whole_sensor_proof_rejects_hidden_attempts_before_model_load(
    tmp_path, monkeypatch, kind
):
    module = load_script("audit_sensor_env", monkeypatch)
    p = packet()
    rows = [
        dict(
            c,
            status="completed",
            physics_attempted=c["physics"],
            engine_attempts_confirmed=c["physics"],
        )
        for c in sensor_cases()
    ]
    result = {
        "protocol_sha256": digest(p),
        "status": "completed",
        "rows": rows,
        "physics_attempted": 2748,
        "optimizer_updates": 0,
        "valid_boxing_games": 0,
    }
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif kind == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif kind in {"invalid", "incomplete"}:
        rows[0]["status"] = kind
    elif kind == "row_physics":
        rows[0]["physics_attempted"] -= 1
    elif kind == "confirmed":
        rows[0]["engine_attempts_confirmed"] -= 1
    elif kind == "global_physics":
        result["physics_attempted"] -= 1
    elif kind == "learned_game":
        result["valid_boxing_games"] = 1
    else:
        rows[0]["seed"] += 1
    write_json(tmp_path / "protocol.json", p)
    write_json(tmp_path / "result.json", result)
    monkeypatch.setattr(
        module,
        "load_scene",
        lambda *_: pytest.fail("model load before denominator audit"),
    )
    with pytest.raises(ValueError):
        module.audit_run(tmp_path)


@pytest.mark.parametrize("kind", ["cases", "bundles", "physics", "timing", "raw"])
def test_sensor_packet_rejects_changed_cases_or_hard_contract(kind):
    p = packet()
    if kind == "cases":
        p["cases"][0]["sensor_schedule"]["A"] = [1]
    elif kind == "bundles":
        p["bundles"]["A"] = "foreign"
    elif kind == "physics":
        p["planned_physics"] = 2000
    elif kind == "timing":
        p["delays_ticks"]["B"] = 1
    else:
        p["raw_bytes"] *= 2
    with pytest.raises(ValueError):
        validate_packet(p)


@pytest.mark.parametrize("kind", ["load", "engine", "memory"])
def test_runner_keeps_all_attempts_and_partial_fault_evidence(
    tmp_path, monkeypatch, kind
):
    module = load_script("qualify_sensor_env", monkeypatch)
    monkeypatch.chdir(tmp_path)
    p = packet()
    monkeypatch.setattr(module, "read_packet", lambda *_: p)
    real_check_output = module.subprocess.check_output

    def git_identity_only(command, *args, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return "test-source\n"
        return real_check_output(command, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "check_output", git_identity_only)
    monkeypatch.setattr(
        module, "identity", lambda pid: {"pid": pid, "start_ticks": 123, "state": "R"}
    )
    monkeypatch.setattr(
        "sys.argv", ["qualify_sensor_env.py", "--protocol-sha256", digest(p)]
    )
    monkeypatch.setattr(
        module, "available_memory", lambda: 0 if kind == "memory" else 64 * 1024**3
    )
    if kind == "load":

        def fail(*_):
            raise ValueError("injected frozen model mismatch")

        monkeypatch.setattr(module, "load_scene", fail)
    else:
        scene = build_scene()
        monkeypatch.setattr(module, "load_scene", lambda *_: scene)
        if kind == "engine":
            import joxing.env

            original = joxing.env.step_contacts
            calls = []

            def fail_step(model, data):
                calls.append(1)
                if len(calls) == 3:
                    data.qvel[0] = np.nan
                    raise FloatingPointError("injected engine fault")
                return original(model, data)

            monkeypatch.setattr(joxing.env, "step_contacts", fail_step)
    with pytest.raises(SystemExit):
        module.main()
    folder = tmp_path / "runs/v0.0.11/sensor-proof-01"
    r = json.loads((folder / "result.json").read_text())
    assert r["status"] == "halted" and len(r["rows"]) == 8
    assert r["valid_boxing_games"] == r["optimizer_updates"] == 0
    if kind == "load":
        assert all(row["status"] == "not_run" for row in r["rows"])
        assert r["physics_attempted"] == 0
    else:
        assert all(row["status"] == "not_run" for row in r["rows"][1:])
        assert r["physics_attempted"] == (3 if kind == "engine" else 0)
        assert r["rows"][0]["status"] == (
            "invalid" if kind == "engine" else "incomplete"
        )
        assert (folder / "fault-engine.npz").exists()
        if kind == "engine":
            assert not (folder / "zero-90101/sensor-final.json").exists()
            with np.load(folder / "last-valid-engine.npz") as data:
                assert np.isfinite(data["state"]).all()
    assert (folder / "raw-manifest.json").exists()
