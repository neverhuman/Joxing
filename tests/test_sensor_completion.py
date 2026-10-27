import copy
import json
from pathlib import Path

import numpy as np
import pytest
from test_sensor_proof import load_script

from joxing.boxing_evidence import controller_bundles
from joxing.evidence_storage import read_arrays, read_json
from joxing.provenance import digest
from joxing.scene import build_scene
from joxing.sensor_completion import attempt_rows, validate_packet, validate_result
from joxing.sensor_protocol import sensor_cases


def packet():
    root = Path(__file__).resolve().parents[1]
    setup = json.loads(
        (
            root / "artifacts/v0.0.11/development/sensor-proof-attempt-01-failure.json"
        ).read_text()
    )
    prior = json.loads((root / "artifacts/v0.0.11/D1/result.json").read_text())
    return {
        "schema": "joxing-sensor-storage-completion-v1",
        "cases": sensor_cases()[6:],
        "fixed_cases": sensor_cases()[:6],
        "bundles": controller_bundles(),
        "planned_physics": 317,
        "physics_cap": 1000,
        "runner_wall_s": 60,
        "post_gate_wall_s": 120,
        "raw_bytes": 24 * 1024**2,
        "delays_ticks": {"A": 1, "B": 2},
        "formal_simulation_workers": 1,
        "threads_per_process": 1,
        "prior_v11_physics": 5073,
        "whole_v11_caps": {"physics": 120000, "raw_bytes": 256 * 1024**2},
        "fixed_source_commit": "ef42c872d13bfebb16b347f842ba11f0b8b8e776",
        "minimum_memory_bytes": 8 * 1024**3,
        "minimum_disk_bytes": 10 * 1024**3,
        "prior_attempts": [
            {
                "id": "setup-01",
                "result": {"rows": setup["rows"], "physics_attempted": 0},
            },
            {"id": "partial-02", "result": prior},
        ],
    }


def result(p):
    return {
        "protocol_sha256": digest(p),
        "status": "completed",
        "rows": [
            dict(
                c,
                status="completed",
                physics_attempted=c["physics"],
                engine_attempts_confirmed=c["physics"],
            )
            for c in p["cases"]
        ],
        "physics_attempted": 317,
        "valid_boxing_games": 0,
        "optimizer_updates": 0,
    }


def test_eighteen_attempts_remain_visible_including_prior_failures():
    p = packet()
    r = result(p)
    validate_result(p, r)
    rows = attempt_rows(p, r)
    assert len(rows) == 18 and len({row["id"] for row in rows}) == 8
    assert [row["status"] for row in rows] == ["not_run"] * 8 + ["completed"] * 6 + [
        "incomplete",
        "not_run",
    ] + ["completed"] * 2
    assert sum(row.get("physics_attempted", 0) for row in rows) == 2748


@pytest.mark.parametrize(
    "fault",
    [
        "old_case",
        "old_selection",
        "seed",
        "schedule",
        "cap",
        "raw",
        "time",
        "controller",
        "threads",
    ],
)
def test_changed_case_selection_or_contract_rejected(fault):
    p = packet()
    if fault == "old_case":
        p["cases"][0] = copy.deepcopy(sensor_cases()[0])
    elif fault == "old_selection":
        p["fixed_cases"][0]["id"] = "drop-90203"
    elif fault == "seed":
        p["cases"][0]["seed"] += 1
    elif fault == "schedule":
        p["cases"][0]["sensor_schedule"]["A"] = [0]
    elif fault == "cap":
        p["physics_cap"] += 1
    elif fault == "raw":
        p["raw_bytes"] *= 2
    elif fault == "time":
        p["runner_wall_s"] *= 2
    elif fault == "controller":
        p["bundles"]["B"] = "invented"
    else:
        p["threads_per_process"] = 2
    with pytest.raises(ValueError):
        validate_packet(p)


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "partial", "confirmed", "count", "learning", "hash"],
)
def test_missing_corrupt_completion_rows_rejected_before_model(fault):
    p = packet()
    r = result(p)
    if fault == "missing":
        r["rows"].pop()
    elif fault == "duplicate":
        r["rows"][1] = copy.deepcopy(r["rows"][0])
    elif fault == "partial":
        r["rows"][0]["status"] = "incomplete"
    elif fault == "confirmed":
        r["rows"][1]["engine_attempts_confirmed"] -= 1
    elif fault == "count":
        r["physics_attempted"] += 1
    elif fault == "learning":
        r["optimizer_updates"] = 1
    else:
        r["protocol_sha256"] = "wrong"
    with pytest.raises(ValueError):
        validate_result(p, r)


@pytest.mark.parametrize("fault", ["load", "engine", "memory"])
def test_packed_completion_fault_keeps_both_rows_and_last_valid(
    tmp_path, monkeypatch, fault
):
    module = load_script("qualify_sensor_storage_completion", monkeypatch)
    p = packet()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "read_packet", lambda *_: p)
    original = module.subprocess.check_output
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command, *a, **kw: (
            "test-source\n"
            if command == ["git", "rev-parse", "HEAD"]
            else original(command, *a, **kw)
        ),
    )
    monkeypatch.setattr(
        module, "identity", lambda pid: {"pid": pid, "start_ticks": 123}
    )
    monkeypatch.setattr(
        "sys.argv",
        ["qualify_sensor_storage_completion.py", "--protocol-sha256", digest(p)],
    )
    monkeypatch.setattr(
        module, "available_memory", lambda: 0 if fault == "memory" else 64 * 1024**3
    )
    if fault == "load":

        def fail(*_):
            raise ValueError("injected model failure")

        monkeypatch.setattr(module, "load_scene", fail)
    else:
        scene = build_scene()
        monkeypatch.setattr(module, "load_scene", lambda *_: scene)
        if fault == "engine":
            import joxing.env

            original_step = joxing.env.step_contacts
            calls = []

            def fail_step(model, data):
                calls.append(1)
                if len(calls) == 3:
                    data.qvel[0] = np.nan
                    raise FloatingPointError("injected engine failure")
                return original_step(model, data)

            monkeypatch.setattr(joxing.env, "step_contacts", fail_step)
    with pytest.raises(SystemExit):
        module.main()
    out = tmp_path / "runs/v0.0.11/sensor-storage-completion-01"
    r = json.loads((out / "result.json").read_text())
    assert r["status"] == "halted" and len(r["rows"]) == 2
    assert r["rows"][1]["status"] == "not_run"
    assert r["physics_attempted"] == (3 if fault == "engine" else 0)
    if fault != "load":
        assert read_arrays(out / "fault-engine.npz")["state"].size > 0
        if fault == "engine":
            assert np.isfinite(
                read_arrays(out / "last-valid-engine.npz")["state"]
            ).all()
            assert not (out / "drop-90203/sensor-final.json.gz").exists()
        assert read_json(out / "fault.json")["reason"]
    assert (out / "raw-manifest.json").exists()


def test_two_row_runner_uses_actual_mid_hold_and_dropped_actor_never_calls(
    tmp_path, monkeypatch
):
    module = load_script("qualify_sensor_storage_completion", monkeypatch)
    p = packet()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "read_packet", lambda *_: p)
    original = module.subprocess.check_output
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command, *a, **kw: (
            "test-source\n"
            if command == ["git", "rev-parse", "HEAD"]
            else original(command, *a, **kw)
        ),
    )
    monkeypatch.setattr(
        module, "identity", lambda pid: {"pid": pid, "start_ticks": 123}
    )
    monkeypatch.setattr(module, "available_memory", lambda: 64 * 1024**3)
    monkeypatch.setattr(
        "sys.argv",
        ["qualify_sensor_storage_completion.py", "--protocol-sha256", digest(p)],
    )
    scene = build_scene()
    monkeypatch.setattr(module, "load_scene", lambda *_: scene)
    module.main()
    out = tmp_path / "runs/v0.0.11/sensor-storage-completion-01"
    r = json.loads((out / "result.json").read_text())
    validate_result(p, r)
    assert r["physics_attempted"] == 317
    assert r["rows"][0]["runtime_physical_audit"]["actor_calls"] == {"A": 0, "B": 9}
    assert r["rows"][1]["runtime_replay"]["sensor_exact"]
    actual = read_json(out / "drop-90203/mid.json")
    restored = read_json(out / "drop-mid-90203/sensor-initial.json")
    assert actual == restored
    assert actual["base"]["application"]["tick"] == 83
    assert actual["base"]["application"]["remaining_control"] == 17
    assert read_json(out / "drop-90203/sensor-final.json") == read_json(
        out / "drop-mid-90203/sensor-final.json"
    )
