"""The independent saved-evidence auditor must reject plausible corruptions."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "residual_audit", ROOT / "scripts/verify_motor_residual.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("corruption", ["reward", "contact"])
def test_saved_evidence_corruption_rejected(tmp_path, corruption):
    receipts = ROOT / "artifacts/v0.0.10/D1"
    folder = tmp_path / "raw"
    shutil.copytree(receipts / "raw", folder)
    if corruption == "reward":
        p = folder / "boundary-recorded_actions-trace.json"
        value = json.loads(p.read_text())
        value[-1]["motor_reward"] += 0.001
        p.write_text(json.dumps(value))
    else:
        p = folder / "boundary-recorded_actions-substeps.npz"
        with np.load(p, allow_pickle=False) as a:
            values = {k: a[k].copy() for k in a.files}
        values["contact_force_N"][0, 2] *= -1
        assert values["contact_force_N"][0, 2] != 0
        np.savez_compressed(p, **values)
    with pytest.raises(AssertionError):
        audit.verify(folder, receipts)
