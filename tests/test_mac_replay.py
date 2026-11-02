import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.audit_mac_replay import verify
from scripts.qualify_mac_replay import exact_decode


@pytest.mark.parametrize("difference", [0.0, 1e-12, -0.0])
def test_decode_is_exact_and_preserves_signed_zero(difference):
    a = np.zeros(700, np.float64)
    b = a.copy()
    b[0] = difference
    row = exact_decode(a, b)
    assert row["exact"] == (a.tobytes() == b.tobytes())


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan", "inf"])
def test_bad_native_controls_reject(kind):
    a = np.zeros(700, np.float64)
    b = a.copy()
    if kind == "dtype":
        b = b.astype(np.float32)
    elif kind == "shape":
        b = b[:699]
    else:
        b[0] = float(kind)
    with pytest.raises(ValueError):
        exact_decode(a, b)


def make_fixture(tmp_path):
    inputs, out = tmp_path / "inputs", tmp_path / "out"
    inputs.mkdir()
    out.mkdir()
    command = np.zeros(700, np.float32)
    decoded = (1 / (1 + np.exp(-5 * (command - 0.5)))).astype(np.float64)
    state = np.zeros(3311, np.float64)
    state[1911:2611] = decoded
    state[2611:3311] = decoded
    state[1911] += 1e-6
    snapshot = {"engine": state.tolist(), "application": {"commands": {a: command.tolist() for a in ("A", "B")}}}
    path = inputs / "boundary.json"
    path.write_text(json.dumps(snapshot))
    cases = [{"id": x, "snapshot": "boundary.json"} for x in ("boundary", "mid", "contact")]
    native = [{"id": str(i)} for i in range(6)]
    packet = {"preflight_cases": cases, "future_native_cases": native,
              "versions": {"numpy": np.__version__}, "control_prefix": 1911,
              "input_files": {"boundary.json": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
    arrays, comparisons = {}, {}
    for a, begin in (("A", 1911), ("B", 2611)):
        expected = state[begin:begin + 700].copy()
        arrays.update({a + "_stored": expected, a + "_decoded": decoded.copy(), a + "_command": command.copy()})
        comparisons[a] = exact_decode(expected, decoded)
    np.savez_compressed(out / "boundary-controls.npz", **arrays)
    result = {"protocol_sha256": hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "versions": packet["versions"], "physics_calls": [], "physics_steps": 0,
              "control_prefix": 1911, "native_replay_rows": [{**c, "status": "not_run"} for c in native],
              "rows": [{**cases[0], "status": "failed", "phase": "control_decode", "corners": comparisons},
                       *[{**c, "status": "not_run"} for c in cases[1:]]],
              "status": "failed_prerequisite"}
    (out / "result.json").write_text(json.dumps(result))
    return packet, inputs, out, result


def test_independent_audit_confirms_failed_decode_without_physics(tmp_path):
    packet, inputs, out, _ = make_fixture(tmp_path)
    audit = verify(packet, out, inputs)
    assert audit["new_physics"] == 0 and audit["native_replay_not_run"] == 6
    assert audit["cases"][0]["corners"]["A"]["different"] == 1


@pytest.mark.parametrize("corruption", ["false_pass", "physics", "call", "missing_row", "duplicate_row", "native_ran", "changed_native", "wrong_phase", "false_error", "false_exact", "later_attempt", "wrong_prefix", "changed_input"])
def test_audit_rejects_forged_results(tmp_path, corruption):
    packet, inputs, out, original = make_fixture(tmp_path)
    result = copy.deepcopy(original)
    if corruption == "false_pass":
        result["rows"][0]["status"] = "passed"
        result["status"] = "passed_prerequisite_only"
    elif corruption == "physics":
        result["physics_steps"] = 1
    elif corruption == "call":
        result["physics_calls"] = ["mj_step"]
    elif corruption == "missing_row":
        result["rows"].pop()
    elif corruption == "duplicate_row":
        result["rows"][1] = result["rows"][0]
    elif corruption == "native_ran":
        result["native_replay_rows"][0]["status"] = "passed"
    elif corruption == "changed_native":
        result["native_replay_rows"][0]["id"] = "other"
    elif corruption == "wrong_phase":
        result["rows"][0]["phase"] = "atomic_restore"
    elif corruption == "false_error":
        result["rows"][0]["corners"]["A"]["max_absolute_error"] = 0.0
    elif corruption == "false_exact":
        result["rows"][0]["corners"]["A"]["exact"] = True
    elif corruption == "later_attempt":
        result["rows"][1]["status"] = "passed"
    elif corruption == "wrong_prefix":
        result["control_prefix"] = 0
    else:
        (inputs / "boundary.json").write_text("{}")
    (out / "result.json").write_text(json.dumps(result))
    with pytest.raises((ValueError, KeyError)):
        verify(packet, out, inputs)
