import json
import shutil

import numpy as np
import pytest

from joxing.boxing_audit import audit_case, compare_replay
from joxing.boxing_evidence import BoxingTape, controller_bundles, recording_cases
from joxing.env import BoxingEnv
from joxing.motor_storage import write_json
from joxing.scene import build_scene

DELAYS = {"A": 1, "B": 2}


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("match-audit")
    scene = build_scene()
    env = BoxingEnv(scene, horizon_ticks=40, bundles=controller_bundles())
    env.reset(seed=90101)
    tape = BoxingTape(env)
    checkpoints = {}
    while env.tick < 40:
        if env.tick == 20:
            checkpoints["boundary"] = env.snapshot()
        if not env.remaining_control:
            env.begin_control(tape.choose(env), delays=DELAYS)
        env.advance_substep()
        tape.capture(env)
        if env.tick == 23:
            checkpoints["mid"] = env.snapshot()
    write_json(root / "source/contract.json", env.contract)
    tape.save(env, root / "source")
    for name, snapshot in checkpoints.items():
        env.restore(snapshot)
        replay = BoxingTape(env)
        while env.tick < 40:
            if not env.remaining_control:
                env.begin_control(replay.choose(env), delays=DELAYS)
            env.advance_substep()
            replay.capture(env)
        write_json(root / name / "contract.json", env.contract)
        replay.save(env, root / name)
    env = BoxingEnv(
        scene,
        horizon_ticks=20,
        scenario="close-overlap-collision-probe-v2",
        bundles=controller_bundles(),
    )
    env.reset(seed=90104)
    tape = BoxingTape(env)
    env.begin_control(tape.choose(env), delays=DELAYS)
    env.advance_substep()
    tape.capture(env)
    mid = env.snapshot()
    while env.remaining_control:
        env.advance_substep()
        tape.capture(env)
    write_json(root / "contact/contract.json", env.contract)
    tape.save(env, root / "contact")
    env.restore(mid)
    tape = BoxingTape(env)
    while env.remaining_control:
        env.advance_substep()
        tape.capture(env)
    write_json(root / "contact-replay/contract.json", env.contract)
    tape.save(env, root / "contact-replay")
    return root, scene


def test_typed_tapes_boundary_mid_command_and_actual_contact_replay(evidence):
    root, scene = evidence
    for name, count in [
        ("source", 40),
        ("boundary", 20),
        ("mid", 17),
        ("contact", 20),
        ("contact-replay", 19),
    ]:
        result = audit_case(root / name, scene, expected_physics=count, delays=DELAYS)
        assert result["status"] == "passed" and result["audit_new_physics"] == 0
        if name.startswith("contact"):
            assert result["positive_glove_touches"] > 0
    for source, replay in [
        ("source", "boundary"),
        ("source", "mid"),
        ("contact", "contact-replay"),
    ]:
        assert compare_replay(root / source, root / replay)["status"] == "passed"


@pytest.mark.parametrize(
    "kind",
    [
        "applied",
        "chosen",
        "observation",
        "history",
        "future_history",
        "contact_sign",
        "touch_force",
        "missing_interval",
        "packet",
        "rng",
        "ledger",
    ],
)
def test_audit_rejects_semantic_tape_corruption(evidence, tmp_path, kind):
    root, scene = evidence
    destination = tmp_path / "bad"
    shutil.copytree(root / "contact", destination)
    trace = json.loads((destination / "trace.json").read_text())
    with np.load(destination / "tape.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if kind == "applied":
        arrays["applied"][0, 0, 0] += 0.1
    if kind == "chosen":
        arrays["chosen"][0, 0, 0] += 0.1
    if kind == "observation":
        arrays["observations"][0, 0, 0] += 0.1
    if kind == "history":
        arrays["histories"][0, 0, -1, 0] += 0.1
    if kind == "future_history":
        arrays["history_ticks"][0, 0, -1] += 1
    if kind == "contact_sign":
        arrays["contacts"][0, 6:9] *= -1
    if kind == "touch_force":
        next(g for g in trace["intervals"][0]["gloves"].values() if g["touches"])[
            "touches"
        ][0]["normal_force_N"] += 1
    if kind == "missing_interval":
        trace["intervals"].pop()
    if kind == "packet":
        arrays["packets"][0, 0, 0] += 10
    if kind == "rng":
        trace["rngs_before"][0]["A"]["state"]["state"] += 1
    if kind == "ledger":
        trace["intervals"][0]["events"].append({"event_id": "invented", "points": 2})
    np.savez_compressed(destination / "tape.npz", **arrays)
    write_json(destination / "trace.json", trace)
    with pytest.raises(ValueError):
        audit_case(destination, scene, expected_physics=20, delays=DELAYS)


def test_replay_checks_contact_engine_hashes_and_complete_tail(evidence, tmp_path):
    root, _ = evidence
    destination = tmp_path / "bad"
    shutil.copytree(root / "contact-replay", destination)
    trace = json.loads((destination / "trace.json").read_text())
    trace["state_sha256"][0] = "0" * 64
    write_json(destination / "trace.json", trace)
    with pytest.raises(ValueError, match="tail mismatch"):
        compare_replay(root / "contact", destination)


def test_preregistered_case_denominator_and_no_missing_replay():
    cases = recording_cases()
    assert len(cases) == len({c["id"] for c in cases}) == 7
    assert sum(c["physics"] for c in cases) == 2642
    assert all(c["physics"] == c["end_tick"] - c["start_tick"] for c in cases)
    assert [c["start_tick"] for c in cases if "parent" in c] == [200, 203, 1]
    assert [c["checkpoint"] for c in cases if "parent" in c] == [
        "boundary",
        "mid",
        "contact-mid",
    ]


@pytest.mark.parametrize("kind", ["missing_row", "invalid_row"])
def test_whole_run_audit_rejects_hidden_attempts_before_model_load(tmp_path, kind):
    import importlib.util

    from joxing.provenance import digest

    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/audit_boxing_env.py"
    )
    spec = importlib.util.spec_from_file_location("boxing_whole_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = {"cases": recording_cases()}
    rows = [dict(c, status="completed") for c in recording_cases()]
    if kind == "missing_row":
        rows.pop()
    else:
        rows[0]["status"] = "invalid"
    write_json(tmp_path / "protocol.json", protocol)
    write_json(
        tmp_path / "result.json",
        {"protocol_sha256": digest(protocol), "rows": rows, "status": "completed"},
    )
    with pytest.raises(ValueError):
        module.audit_run(tmp_path)


def test_quarantine_saves_finite_partial_tape_without_inventing_valid_final(
    evidence, tmp_path, monkeypatch
):
    root, scene = evidence
    initial = json.loads((root / "contact/initial.json").read_text())
    env = BoxingEnv(
        scene,
        horizon_ticks=20,
        scenario="close-overlap-collision-probe-v2",
        bundles=controller_bundles(),
    )
    env.restore(initial)
    tape = BoxingTape(env)
    env.begin_control(tape.choose(env), delays=DELAYS)
    env.advance_substep()
    tape.capture(env)

    def fault(model, data):
        data.qvel[0] = np.nan
        raise FloatingPointError("saved-partial fixture")

    monkeypatch.setattr("joxing.env.step_contacts", fault)
    with pytest.raises(FloatingPointError):
        env.advance_substep()
    result = tape.save(env, tmp_path, quarantined=True)
    assert result["physics"] == 1 and result["final_sha256"] is None
    assert env.physics_total == 2 and not (tmp_path / "final.json").exists()
    assert json.loads((tmp_path / "trace.json").read_text())["transition"][
        "invalid_reason"
    ]
    with np.load(tmp_path / "tape.npz", allow_pickle=False) as data:
        assert (
            data["applied"].shape == (1, 2, 700) and np.isfinite(data["contacts"]).all()
        )
