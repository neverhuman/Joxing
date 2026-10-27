"""Bounded zero-physics archived-input/normalizer/scalar-actor oracle.

This is not the shared-model geometry/reset/closed-loop transfer proof. Two
archived cache rows preserve bytes; two scalar predictions reproduce the actual
next saved native command at its original source host. Cross-host differences
must remain failures, not a reason to relax the exact assertion.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from joxing.evidence_storage import sha_file
from joxing.motor_storage import RawCredit, save_npz, write_json
from joxing.provenance import digest
from joxing.shared_motor_inputs import FrozenPairTeacher, MotorPolicyInput

PACKET = "experiments/v0.0.11/shared-motor-input-protocol.json"


def qualify(p, root, out):
    if out.exists():
        raise FileExistsError("Fresh oracle output required")
    out.mkdir(parents=True)
    credit = RawCredit(out, p["raw_bytes"])
    result = {
        "schema": "joxing-shared-motor-input-oracle-v1",
        "status": "running",
        "new_physics": 0,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "protocol_sha256": digest(p),
        "rows": [
            {"id": n, "status": "not_run"}
            for n in ["cache-mid", "cache-final", "actor-mid-A", "actor-mid-B"]
        ],
    }
    write_json(out / "result.json", result)
    started = time.monotonic()
    active = None
    try:
        for name, sha in {
            **p["source_files"],
            **p["upstream_files"],
            **p["reference_files"],
        }.items():
            if sha_file(root / name) != sha:
                raise ValueError("Frozen source/reference drift: " + name)
        saved = json.loads((root / p["reference"]).read_text())
        for row, key, tick in zip(
            result["rows"][:2], ["snapshot", "final_snapshot"], [200, 3520], strict=True
        ):
            active = row
            row["status"] = "attempted"
            write_json(out / "result.json", result)
            raw = np.asarray(saved[key]["last_raw_observation"], dtype=np.float32)
            sample = MotorPolicyInput(
                "A",
                "archived-native-input",
                tick,
                raw,
                digest(saved[key]["application"]),
            )
            restored = MotorPolicyInput.restore(
                json.loads(json.dumps(sample.snapshot()))
            )
            if restored.values.tobytes() != raw.tobytes():
                raise ValueError("Actual native cached input byte mismatch")
            row.update(
                status="passed",
                input_sha256=sample.snapshot()["values_sha256"],
                sampled_tick=tick,
            )
            write_json(out / "result.json", result)
        if time.monotonic() - started >= p["wall_seconds"]:
            raise TimeoutError("Oracle deadline")
        teacher = FrozenPairTeacher.from_inputs(Path(p["inputs_root"]))
        mid = saved["snapshot"]
        raw = np.asarray(mid["last_raw_observation"], dtype=np.float32)
        pair = {
            a: MotorPolicyInput(
                a, "archived-native-input", 200, raw, digest(mid["application"])
            )
            for a in ["A", "B"]
        }
        expected = np.asarray(saved["continuation_actions"][0], dtype=np.float32)
        for row in result["rows"][2:]:
            row["status"] = "attempted"
        write_json(out / "result.json", result)
        actions = teacher.infer_pair(pair, game_id="archived-native-input", tick=200)
        normalized = teacher.normalizer.apply(raw)
        save_npz(
            out / "oracle.npz",
            raw=raw,
            normalized=normalized,
            expected=expected,
            action_A=actions["A"],
            action_B=actions["B"],
        )
        failures = []
        for row, a in zip(result["rows"][2:], ["A", "B"], strict=True):
            active = row
            error = float(
                np.max(abs(actions[a].astype(np.float64) - expected.astype(np.float64)))
            )
            row.update(
                status="passed" if np.array_equal(actions[a], expected) else "failed",
                max_command_error=error,
            )
            if row["status"] != "passed":
                failures.append(a)
        if failures:
            raise ValueError(
                "Saved native command differs on this host: " + ",".join(failures)
            )
        teacher.normalizer.check()
        if time.monotonic() - started >= p["wall_seconds"]:
            raise TimeoutError("Oracle deadline")
        result.update(
            status="passed", normalizer_fingerprint=teacher.normalizer.fingerprint
        )
    except Exception as error:  # noqa: BLE001 -- preserve exact failures and all rows
        result.update(status="failed", reason=type(error).__name__ + ": " + str(error))
        if active is not None and active["status"] == "attempted":
            active.update(status="failed", reason=str(error))
    finally:
        result.update(
            wall_s=time.monotonic() - started,
            scope="Archived input/scalar policy compatibility; no shared physical motor trajectory, engine-bound input oracle or balance claim",
        )
        with credit.emergency():
            write_json(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / PACKET).read_text())
    if (
        digest(p) != args.protocol_sha256
        or p["schema"] != "joxing-shared-motor-input-protocol-v1"
        or p["new_physics"] != 0
        or p["wall_seconds"] != 30
        or p["raw_bytes"] != 4 * 1024**2
    ):
        raise ValueError("Oracle protocol/caps")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != args.source_sha
    ):
        raise ValueError("Oracle source HEAD")
    if (
        json.loads(
            subprocess.check_output(
                ["git", "show", "HEAD:" + PACKET], cwd=root, text=True
            )
        )
        != p
    ):
        raise ValueError("Uncommitted oracle protocol")
    result = qualify(p, root, args.out)
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
