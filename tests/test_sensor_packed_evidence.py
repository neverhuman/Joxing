"""Same collected native inputs saved twice; packing adds zero physics."""

import copy
import json
from pathlib import Path

from test_sensor_audit import DELAYS, collect
from test_sensor_audit import evidence as evidence  # noqa: PLC0414 # pytest's real mid-held fixture

from joxing.evidence_storage import array_descriptor, read_arrays, read_json, write_json
from joxing.sensor_audit import compare_sensor_replay
from joxing.sensor_evidence import SensorTape
from joxing.sensor_packed_evidence import PackedSensorTape
from joxing.sensor_physics_audit import audit_sensor_physics


def repack(source, target):
    from joxing.evidence_storage import save_npz

    for path in sorted(Path(source).iterdir()):
        if path.suffix == ".json":
            write_json(target / path.name, read_json(path))
        elif path.suffix == ".npz":
            save_npz(target / path.name, **read_arrays(path))


def test_original_and_packed_pending_history_replay_have_identical_audit(
    evidence, tmp_path
):
    scene, source, mid = evidence
    packed_source, packed_mid = tmp_path / "source", tmp_path / "mid"
    repack(source, packed_source)
    repack(mid, packed_mid)
    for old, new, steps in [(source, packed_source, 200), (mid, packed_mid, 117)]:
        assert audit_sensor_physics(
            old, scene, expected_physics=steps, delays=DELAYS
        ) == audit_sensor_physics(new, scene, expected_physics=steps, delays=DELAYS)
    assert compare_sensor_replay(
        packed_source,
        packed_mid,
        source_checkpoint=packed_source / "checkpoint-83.json",
    ) == compare_sensor_replay(
        source, mid, source_checkpoint=source / "checkpoint-83.json"
    )


def test_save_only_subclass_matches_frozen_writer_and_quarantine(evidence, tmp_path):
    from joxing.sensed_env import SensedBoxingEnv

    scene, source, _ = evidence
    env = SensedBoxingEnv(
        scene, horizon_ticks=200, sensor_schedule={"A": [40, 0], "B": [None, 20, 0]}
    )
    env.restore(read_json(source / "checkpoint-83.json"))
    tape = PackedSensorTape(env, maximum_ticks=117)
    collect(env, tape)
    assert PackedSensorTape.choose is SensorTape.choose
    assert PackedSensorTape.capture is SensorTape.capture
    for quarantined in (False, True):
        old, new = (
            tmp_path / str(quarantined) / "old",
            tmp_path / str(quarantined) / "new",
        )
        # Same tape, final engine and RNG; writing must never step or choose.
        before = copy.deepcopy(env.snapshot())
        assert SensorTape.save(tape, env, old, quarantined=quarantined) == tape.save(
            env, new, quarantined=quarantined
        )
        assert env.snapshot() == before
        logical = {
            p.name.removesuffix(".gz")
            for p in new.iterdir()
            if p.name != "storage-manifest.json"
        }
        assert logical == {p.name for p in old.iterdir()}
        for p in old.iterdir():
            if p.suffix == ".json":
                assert json.loads(p.read_text()) == read_json(new / p.name)
            else:
                a, b = read_arrays(p), read_arrays(new / p.name)
                assert {k: array_descriptor(v) for k, v in a.items()} == {
                    k: array_descriptor(v) for k, v in b.items()
                }
        assert (new / "sensor-final.json.gz").exists() != quarantined
        assert (new / "final.json.gz").exists() != quarantined
