"""Independently recalculate retained trace errors; no model execution."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from joxing.provenance import ROOT, digest, write_json
from joxing.stance_train import file_hash


def audit(folder):
    report = json.loads((folder / "result.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    assert report["protocol_sha256"] == digest(protocol)
    assert protocol == json.loads(
        (ROOT / "experiments/v0.0.9/numerics-protocol.json").read_text()
    )
    source = subprocess.check_output(
        [
            "git",
            "show",
            report["code_commit"] + ":scripts/diagnose_inference_numerics.py",
        ],
        cwd=ROOT,
    )
    assert hashlib.sha256(source).hexdigest() == report["script_sha256"]
    fault_path = ROOT / "artifacts/v0.0.9/H0/parity-fault.npz"
    assert file_hash(fault_path) == protocol["fault_sha256"]
    assert (
        report["status"] == "completed"
        and report["physics_steps"] == report["boxing_games"] == 0
    )
    assert [r["variant"] for r in report["cases"]] == protocol["variants"]
    previous_normalized = None
    findings = []
    with np.load(fault_path, allow_pickle=False) as fault:
        for row in report["cases"]:
            assert row["status"] == "completed"
            path = folder / (row["variant"] + ".npz")
            assert file_hash(path) == row["arrays_sha256"]
            with np.load(path, allow_pickle=False) as data:
                normalized = data["normalized"]
                assert normalized.shape == (2, 3601) and np.isfinite(normalized).all()
                if previous_normalized is not None:
                    assert np.array_equal(normalized, previous_normalized)
                previous_normalized = normalized.copy()
                first = None
                for i, module in enumerate(row["modules"]):
                    a, b = data[f"scalar_{i}"], data[f"batched_{i}"]
                    assert (
                        list(a.shape) == module["shape"] == list(b.shape)
                        and a.shape[0] == 2
                    )
                    assert np.isfinite(a).all() and np.isfinite(b).all()
                    delta = abs(a.astype(np.float64) - b.astype(np.float64))
                    assert float(delta.max()) == module["max_absolute_error"]
                    assert (
                        np.count_nonzero(delta, axis=1).tolist()
                        == module["changed_per_row"]
                    )
                    assert (
                        np.count_nonzero(delta > 1e-6, axis=1).tolist()
                        == module["channels_over_1e_6_per_row"]
                    )
                    assert bool(np.array_equal(a, b)) == module["exact"]
                    if not module["exact"] and first is None:
                        first = module["module_call"]
                scalar, batch = data["scalar_action"], data["batched_action"]
                assert scalar.shape == batch.shape == (2, 700)
                error = float(
                    abs(scalar.astype(np.float64) - batch.astype(np.float64)).max()
                )
                assert error == row["action_error"]["max_absolute_error"]
                assert first == row["first_divergent_module"]
                if row["variant"] == "default_float32":
                    assert np.array_equal(
                        scalar, fault["reference"]
                    ) and np.array_equal(batch, fault["actual"])
                assert row["old_h0_gate_met"] == (
                    error <= protocol["original_h0_action_max_error"]
                )
                assert (
                    row["hooks_output_preserving"] and row["unchanged_actor_normalizer"]
                )
                findings.append(
                    {
                        "variant": row["variant"],
                        "first_divergent_module": first,
                        "max_action_error": error,
                        "old_h0_gate_met": row["old_h0_gate_met"],
                    }
                )
    exited = json.loads((folder / "process-exit.json").read_text())
    assert (
        not exited["recorded_pids_present"]
        and not exited["owned_jobs"]
        and not exited["source_diff"]
        and exited["stages_all_zero"]
    )
    return {
        "passed": True,
        "scope": "Independent saved-array/hash/accounting audit; not another actor run or backend qualification.",
        "source_commit": report["code_commit"],
        "protocol_sha256": digest(protocol),
        "completed": len(findings),
        "invalid": 0,
        "incomplete": 0,
        "not_run": 0,
        "physics_steps": 0,
        "backend_qualified": False,
        "findings": findings,
        "result_sha256": file_hash(folder / "result.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    write_json(args.folder / "audit.json", result)
    print(json.dumps(result, indent=2))
