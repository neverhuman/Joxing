"""Independent zero-physics H25 prerequisite audit; does not import MuJoCo."""

import hashlib
import json
from pathlib import Path

import numpy as np


def verify(packet, folder, inputs):
    folder, inputs = Path(folder), Path(inputs)
    result = json.loads((folder / "result.json").read_text())
    if result["protocol_sha256"] != hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise ValueError("Protocol mismatch")
    if result["versions"] != packet["versions"] or result["physics_calls"] or result["physics_steps"]:
        raise ValueError("Runtime or forbidden physics")
    if len(result["rows"]) != 3 or len(result["native_replay_rows"]) != 6:
        raise ValueError("Row denominator")
    if [r["id"] for r in result["rows"]] != [r["id"] for r in packet["preflight_cases"]]:
        raise ValueError("Case order")
    if any(r["status"] != "not_run" for r in result["native_replay_rows"]):
        raise ValueError("Unimplemented native row ran")
    if any(any(row[k] != v for k, v in original.items()) for row, original in zip(result["native_replay_rows"], packet["future_native_cases"], strict=True)):
        raise ValueError("Changed native row identity")
    if packet["control_prefix"] != 1911 or result["control_prefix"] != 1911:
        raise ValueError("Native control-state offset")
    for name, expected in packet["input_files"].items():
        if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("Changed input: " + name)
    audited = []
    stopped = False
    for registered, row in zip(packet["preflight_cases"], result["rows"], strict=True):
        if any(row[k] != v for k, v in registered.items()):
            raise ValueError("Mutated registered case")
        if stopped or row["status"] == "not_run":
            if row["status"] != "not_run":
                raise ValueError("Rows after first failure")
            continue
        snapshot = json.loads((inputs / row["snapshot"]).read_text())
        state = np.asarray(snapshot["engine"], np.float64)
        with np.load(folder / (row["id"] + "-controls.npz"), allow_pickle=False) as data:
            if set(data.files) != {a + s for a in ("A", "B") for s in ("_stored", "_decoded", "_command")}:
                raise ValueError("Missing/extra arrays")
            corners = {}
            for actor, start in (("A", 0), ("B", 700)):
                offset = packet["control_prefix"] + start
                native = state[offset:offset + 700]
                command = np.asarray(snapshot["application"]["commands"][actor], np.float32)
                calculated = (1 / (1 + np.exp(-5 * (command - 0.5)))).astype(np.float64)
                for name, expected in (("_stored", native), ("_decoded", calculated), ("_command", command)):
                    actual = data[actor + name]
                    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
                        raise ValueError("Saved source/decoder array mismatch")
                different = int(np.count_nonzero(native.view(np.uint64) != calculated.view(np.uint64)))
                error = float(np.max(np.abs(native - calculated)))
                claimed = row["corners"][actor]
                if claimed["differing_values"] != different or claimed["max_absolute_error"] != error or claimed["exact"] != (different == 0):
                    raise ValueError("False decoder comparison")
                corners[actor] = {"different": different, "max_absolute_error": error}
        mismatch = any(v["different"] for v in corners.values())
        if mismatch and (row["status"] != "failed" or row["phase"] != "control_decode"):
            raise ValueError("Decode mismatch cannot pass")
        if row["status"] == "failed":
            stopped = True
        elif row["status"] != "passed":
            raise ValueError("Incomplete attempt")
        audited.append({"id": row["id"], "status": row["status"], "phase": row["phase"], "corners": corners})
    if not audited or (stopped != (result["status"] == "failed_prerequisite")):
        raise ValueError("False overall status")
    return {"status": "verified_prerequisite_outcome", "cases": audited,
            "native_replay_not_run": 6, "new_physics": 0,
            "scope": "reuses declared NumPy decoder; archived engine-control slice checked independently without model/physics"}
