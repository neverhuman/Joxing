"""H25 zero-step prerequisite: exact archived control decoding and restore.

This is deliberately a prerequisite, not the six-row native replay runner.
"""

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(raw + b"\n")
    temp.replace(path)


def exact_decode(held, restored):
    """No tolerance: signed zero/dtype/shape/bytes all participate."""
    import numpy as np

    held, restored = np.asarray(held), np.asarray(restored)
    if held.dtype != np.float64 or restored.dtype != np.float64:
        raise ValueError("Expected native float64 actuator controls")
    if held.shape != (700,) or restored.shape != held.shape:
        raise ValueError("Expected all700 controls per corner")
    if not np.isfinite(held).all() or not np.isfinite(restored).all():
        raise ValueError("Nonfinite controls")
    return {
        "exact": held.tobytes() == restored.tobytes(),
        "differing_values": int(np.count_nonzero(held.view(np.uint64) != restored.view(np.uint64))),
        "max_absolute_error": float(np.max(np.abs(held - restored))),
        "expected_sha256": hashlib.sha256(held.tobytes()).hexdigest(),
        "actual_sha256": hashlib.sha256(restored.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "oracle", "inputs", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    root = Path(__file__).resolve().parents[1]
    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == args.source_sha
    for name, expected in packet["wrapper_files"].items():
        assert sha(root / name) == expected, name
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.oracle, text=True).strip() == packet["oracle_commit"]
    for name, expected in packet["oracle_files"].items():
        assert sha(args.oracle / name) == expected, name
    for name, expected in packet["input_files"].items():
        assert sha(args.inputs / name) == expected["sha256"], name
    args.out.mkdir(parents=True, exist_ok=False)
    # Authentic original package, with its own provenance ROOT and native assets.
    sys.path.insert(0, str(args.oracle / "src"))
    import mujoco
    import numpy as np

    from joxing.boxing_evidence import controller_bundles
    from joxing.env import INTEGRATION, BoxingEnv, excitation
    from joxing.scene import build_scene

    for module in ("joxing.env", "joxing.replay", "joxing.scene"):
        __import__(module)
        assert Path(sys.modules[module].__file__).resolve().is_relative_to(args.oracle.resolve())
    calls = []

    def forbidden(*unused_args, **unused_kwargs):
        calls.append("forbidden_physics_call")
        raise AssertionError("H25preflight forbids any physics step")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, forbidden)
    result = {
        "schema": "joxing-mac-replay-prerequisite-v1", "source_commit": args.source_sha,
        "protocol_sha256": hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "oracle_commit": packet["oracle_commit"], "status": "running",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__},
        "rows": [{**c, "status": "not_run"} for c in packet["preflight_cases"]],
        "native_replay_rows": [{**c, "status": "not_run"} for c in packet["future_native_cases"]],
        "physics_calls": calls, "physics_steps": 0, "optimizer_updates": 0,
        "scope": "zero-step prerequisite; six-row native replay body remains unimplemented/not_run",
    }
    began = time.monotonic()
    active = None
    save(args.out / "result.json", result)
    try:
        assert result["versions"] == packet["versions"], result["versions"]
        scene = build_scene()
        assert scene.model.nu == scene.model.na == 1400
        assert all(len(scene.actuator_ids[a]) == 700 for a in ("A", "B"))
        prefix = mujoco.mj_stateSize(scene.model, int(INTEGRATION) & (int(mujoco.mjtState.mjSTATE_CTRL) - 1))
        assert prefix == packet["control_prefix"] == 1911
        for actor, begin in (("A", 0), ("B", 700)):
            assert np.array_equal(scene.actuator_ids[actor], np.arange(begin, begin + 700))
        result["control_prefix"] = prefix
        result["scene_hash"] = scene.report["scene_hash"]
        for active in result["rows"]:
            active["status"] = "attempted"
            active["phase"] = "control_decode"
            save(args.out / "result.json", result)
            snapshot = json.loads((args.inputs / active["snapshot"]).read_text())
            assert snapshot["application"]["tick"] == active["tick"]
            env = BoxingEnv(scene, horizon_ticks=active["horizon_ticks"], scenario=active["scenario"], bundles=controller_bundles())
            assert env.contract == json.loads((args.inputs / active["contract"]).read_text())
            data = mujoco.MjData(scene.model)
            mujoco.mj_setState(scene.model, data, np.asarray(snapshot["engine"], np.float64), INTEGRATION)
            arrays = {}
            active["corners"] = {}
            for actor in ("A", "B"):
                expected = data.ctrl[scene.actuator_ids[actor]].copy()
                command = np.asarray(snapshot["application"]["commands"][actor], np.float32)
                actual = excitation(command).astype(np.float64)
                active["corners"][actor] = exact_decode(expected, actual)
                arrays[actor + "_stored"] = expected
                arrays[actor + "_decoded"] = actual
                arrays[actor + "_command"] = command
            np.savez_compressed(args.out / (active["id"] + "-controls.npz"), **arrays)
            if not all(x["exact"] for x in active["corners"].values()):
                raise ValueError("Strict archived held-control decoding differs on Mac")
            active["phase"] = "atomic_restore"
            save(args.out / "result.json", result)
            env.restore(snapshot)
            assert env.snapshot() == snapshot
            active.update(status="passed", physics_total=env.physics_total)
            if time.monotonic() - began > packet["preflight_s"]:
                raise TimeoutError("Preflight wall cap")
            save(args.out / "result.json", result)
        result["status"] = "passed_prerequisite_only"
    except Exception as error:  # noqa: BLE001 -- every target fault retained
        result["status"] = "failed_prerequisite"
        result["error"] = type(error).__name__ + ": " + str(error)
        if active:
            active.update(status="failed", error=result["error"])
        with gzip.open(args.out / "fault.txt.gz", "wt") as stream:
            stream.write(traceback.format_exc())
    result["physics_calls"] = calls
    result["wall_s"] = time.monotonic() - began
    save(args.out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "error", "wall_s", "physics_steps") if k in result}))
    if result["status"] != "passed_prerequisite_only":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
