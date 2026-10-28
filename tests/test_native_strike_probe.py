import copy

import numpy as np
import pytest

from joxing.frozen_group_controller import FrozenGroupController
from joxing.native_strike_probe import NativeStrikeTape, continue_mid
from joxing.provenance import ROOT


@pytest.fixture(scope="module")
def source():
    import hashlib

    from grokbot.scenarios import RangeBoxingEnv
    from joxing.scene import build_scene

    path = ROOT / "grokbot/redteam/data/cycle-0025-hit-stay-gate-summary.json"
    policy_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    def controllers():
        return {
            a: FrozenGroupController.from_summary(path, expected_sha256=policy_hash)
            for a in "AB"
        }

    actors = controllers()
    scene = build_scene()
    env = RangeBoxingEnv(
        scene,
        separation_m=0.5,
        horizon_ticks=60,
        bundles={a: c.bundle_sha256 for a, c in actors.items()},
    )
    tape = NativeStrikeTape(maximum_ticks=60)
    result = tape.run(env, actors, seed=99101, mid_tick=3)
    return scene, env, tape, result, controllers


def test_all_native_muscles_and_actual_mid_hold_continue_exactly(source):
    from grokbot.scenarios import RangeBoxingEnv

    scene, env, tape, result, make = source
    actors = make()
    replay_env = RangeBoxingEnv(
        scene,
        separation_m=0.5,
        horizon_ticks=60,
        bundles={a: c.bundle_sha256 for a, c in actors.items()},
    )
    entered = []
    replay = continue_mid(
        replay_env,
        actors,
        tape,
        step_receipt=lambda phase, tick: entered.append((phase, tick)),
    )
    assert result["source_physics"] == 60
    assert env.model.nu == env.model.na == 1400
    assert replay["new_physics"] == 57 and replay["held_ticks_without_actor"] == 17
    assert replay["decision_ticks"] == [20, 40]
    assert replay["final_sha256"] == result["source_final_sha256"]
    assert all(
        r["actual_sha256"] == r["expected_sha256"]
        for r in replay["actual_physical_tail"]
    )
    assert entered == [
        (phase, k) for k in range(3, 60) for phase in ("entered", "completed")
    ]
    assert not np.any(replay_env.data.xfrc_applied) and not np.any(
        replay_env.data.qfrc_applied
    )


def test_replay_rejects_corrupted_source_contacts_after_passing_baseline(source):
    from grokbot.scenarios import RangeBoxingEnv

    scene, _env, tape, _result, make = source
    actors = make()
    replay_env = RangeBoxingEnv(
        scene,
        separation_m=0.5,
        horizon_ticks=60,
        bundles={a: c.bundle_sha256 for a, c in actors.items()},
    )
    assert continue_mid(replay_env, actors, tape)["status"] == "passed"
    changed = copy.copy(tape)
    changed.intervals = copy.deepcopy(tape.intervals)
    changed.intervals[3]["contacts_sha256"] = "wrong"
    with pytest.raises(ValueError, match="continuation differs"):
        continue_mid(replay_env, make(), changed)


def test_saved_native_measurements_and_frozen_commands_audit_without_physics(
    source, tmp_path
):
    from grokbot.scenarios import RangeBoxingEnv
    from joxing.evidence_storage import read_json, write_json
    from joxing.native_strike_audit import audit_source

    scene, env, tape, result, make = source
    tape.save(env, tmp_path)
    actors = make()
    audit_env = RangeBoxingEnv(
        scene,
        separation_m=0.5,
        horizon_ticks=60,
        bundles={a: c.bundle_sha256 for a, c in actors.items()},
    )
    report = audit_source(tmp_path, audit_env, actors)
    assert report["status"] == "passed" and report["new_physics"] == 0
    assert report["source_final_sha256"] == result["source_final_sha256"]
    assert audit_env.physics_total == 0
    trace = read_json(tmp_path / "trace.json")
    trace["decisions"][0]["rng_before"]["A"]["state"]["state"] += 1
    write_json(tmp_path / "trace.json", trace)
    with pytest.raises(ValueError, match="controller/RNG"):
        audit_source(tmp_path, audit_env, make())


def test_overlay_never_displays_future_points():
    from joxing.native_strike_display import points_at_tick

    event = {
        "event_id": "game/A_glove_r/1",
        "points": 2,
        "finalized_tick": 49,
        "reason": "window",
    }
    assert points_at_tick([event], 49) == [0, 0]
    assert points_at_tick([event], 50) == [2, 0]
    bell = {
        **event,
        "event_id": "game/B_glove_l/1",
        "points": 1,
        "finalized_tick": 60,
        "reason": "bell",
    }
    assert points_at_tick([bell], 59) == [0, 0] and points_at_tick([bell], 60) == [0, 1]


def test_native_runner_retains_unstarted_replay_if_model_preparation_fails(
    tmp_path, monkeypatch
):
    from joxing import native_strike_qualification as q

    def broken(actors):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(q, "controllers", lambda *a: {})
    monkeypatch.setattr(q, "environment", broken)
    p = {
        "raw_bytes": 24 * 1024**2,
        "stage_wall_limits_s": {"proof": 25},
        "maximum_rss_bytes": 4 * 1024**3,
    }
    report = q.prove(tmp_path, p, "test-source", "test-protocol")
    assert report["status"] == "halted" and report["rows"][0]["status"] == "failed"
    assert report["rows"][1]["status"] == "not_run"
    assert (
        report["physics_entered"]
        == report["physics_completed"]
        == report["environment_engine_entries"]
        == 0
    )
    assert (tmp_path / q.RUN / "fault.json").exists()
