import json

import numpy as np
import pytest

from joxing.shared_motor_inputs import FrozenPairTeacher, MotorPolicyInput


def samples():
    return {
        a: MotorPolicyInput(
            a,
            "game",
            200,
            np.arange(3601, dtype=np.float32) * (1 if a == "A" else -1),
            "a" * 64,
        )
        for a in ("A", "B")
    }


def test_actual_saved_native_cache_roundtrips_every_float32_byte():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    saved = json.loads(
        (root / "artifacts/v0.0.6/native/replay-scheduled_restored.json").read_text()
    )
    for key, tick in [("snapshot", 200), ("final_snapshot", 3520)]:
        raw = np.asarray(saved[key]["last_raw_observation"], dtype=np.float32)
        sample = MotorPolicyInput("B", "native-cache-oracle", tick, raw, "a" * 64)
        restored = MotorPolicyInput.restore(json.loads(json.dumps(sample.snapshot())))
        assert restored.values.tobytes() == raw.tobytes()
        assert not restored.values.flags.writeable
        with pytest.raises(ValueError):
            restored.values.flags.writeable = True


@pytest.mark.parametrize(
    "fault", ["hash", "precision", "nan", "shape", "dtype", "schema", "time"]
)
def test_cache_corruption_rejected(fault):
    value = samples()["A"].snapshot()
    if fault == "hash":
        value["values"][0] = 1.0
    elif fault == "precision":
        value["values"][0] = 1.000000001
    elif fault == "nan":
        value["values"][0] = float("nan")
    elif fault == "shape":
        value["values"].pop()
    elif fault == "dtype":
        value["dtype"] = "<f8"
    elif fault == "schema":
        value["private_opponent_activation"] = []
    else:
        value["sampled_tick"] = 203
    with pytest.raises(ValueError):
        MotorPolicyInput.restore(value)


class Normalizer:
    def __init__(self):
        self.checked = 0

    def check(self):
        self.checked += 1

    def apply(self, value):
        return value.copy()


class Actor:
    def __init__(self):
        self.observed = []

    def predict(self, value, *, deterministic):
        assert deterministic
        self.observed.append(value.copy())
        return np.tanh(value[:700] * 0.001).astype(np.float32), None


def test_both_corners_use_frozen_raw_inputs_with_scalar_inference_and_no_sigmoid():
    actor, normalizer = Actor(), Normalizer()
    pair = samples()
    original = {a: v.values.copy() for a, v in pair.items()}
    result = FrozenPairTeacher(actor, normalizer).infer_pair(
        pair, game_id="game", tick=200
    )
    assert normalizer.checked == 2 and len(actor.observed) == 2
    assert actor.observed[0].shape == actor.observed[1].shape == (3601,)
    for a, received in zip(("A", "B"), actor.observed, strict=True):
        assert np.array_equal(received, original[a])
        assert np.array_equal(
            result[a], np.tanh(original[a][:700] * 0.001).astype(np.float32)
        )
        assert not result[a].flags.writeable
    assert result["B"][1] < 0  # normalized action, not excitation


@pytest.mark.parametrize("fault", ["missing", "owner", "game", "time"])
def test_mixed_pair_rejected_before_actor_call(fault):
    pair = samples()
    actor = Actor()
    if fault == "missing":
        pair.pop("B")
    elif fault == "owner":
        pair["B"] = pair["A"]
    elif fault == "game":
        pair["B"] = MotorPolicyInput("B", "other", 200, pair["B"].values, "a" * 64)
    else:
        pair["B"] = MotorPolicyInput("B", "game", 220, pair["B"].values, "a" * 64)
    with pytest.raises(ValueError):
        FrozenPairTeacher(actor, Normalizer()).infer_pair(
            pair, game_id="game", tick=200
        )
    assert not actor.observed


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "range", "hidden"])
def test_invalid_teacher_output_rejected(fault):
    class BadActor:
        def predict(self, value, *, deterministic):
            action = np.zeros(699 if fault == "shape" else 700, dtype=np.float32)
            if fault == "nonfinite":
                action[0] = np.nan
            if fault == "range":
                action[0] = 1.1
            return action, np.zeros(1) if fault == "hidden" else None

    with pytest.raises(ValueError):
        FrozenPairTeacher(BadActor(), Normalizer()).infer_pair(
            samples(), game_id="game", tick=200
        )


@pytest.mark.parametrize("fault", ["none", "missing_reference", "command"])
def test_zero_step_qualification_retains_all_four_rows(tmp_path, monkeypatch, fault):
    from pathlib import Path

    from test_sensor_proof import load_script

    module = load_script("qualify_shared_motor_inputs", monkeypatch)
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts/v0.0.6/native/replay-scheduled_restored.json"
    reference = tmp_path / "reference.json"
    reference.write_bytes(source.read_bytes())
    p = {
        "source_files": {},
        "upstream_files": {},
        "reference_files": {"reference.json": module.sha_file(reference)},
        "reference": "reference.json",
        "inputs_root": str(tmp_path),
        "raw_bytes": 4 * 1024**2,
        "wall_seconds": 30,
    }
    saved = json.loads(reference.read_text())
    expected = np.asarray(saved["continuation_actions"][0], dtype=np.float32)

    class FixedActor:
        def predict(self, value, *, deterministic):
            a = expected.copy()
            if fault == "command":
                a[0] += np.float32(0.001)
            return a, None

    n = Normalizer()
    n.fingerprint = "fixture"
    monkeypatch.setattr(
        module.FrozenPairTeacher,
        "from_inputs",
        lambda *_: FrozenPairTeacher(FixedActor(), n),
    )
    original = module.subprocess.check_output
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda cmd, *a, **kw: (
            "fixture-source"
            if cmd == ["git", "rev-parse", "HEAD"]
            else original(cmd, *a, **kw)
        ),
    )
    if fault == "missing_reference":
        reference.unlink()
    r = module.qualify(p, tmp_path, tmp_path / "out")
    assert r["new_physics"] == 0 and len(r["rows"]) == 4
    assert r["status"] == ("passed" if fault == "none" else "failed")
    if fault == "command":
        assert [x["status"] for x in r["rows"]] == [
            "passed",
            "passed",
            "failed",
            "failed",
        ]
    if fault == "missing_reference":
        assert all(x["status"] == "not_run" for x in r["rows"])
