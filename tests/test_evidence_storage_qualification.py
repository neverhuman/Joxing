import importlib.util
from pathlib import Path

import numpy as np
import pytest

from joxing.evidence_storage import sha_file

spec = importlib.util.spec_from_file_location(
    "qualify_storage",
    Path(__file__).resolve().parents[1] / "scripts/qualify_evidence_storage.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def packet(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.json").write_text(
        '{"sampled_ticks":{"203":[0.0,-0.0]},"rng":99999999999999999999}'
    )
    np.savez_compressed(source / "b.npz", a=np.arange(100, dtype=np.float32))
    (source / "events.jsonl").write_bytes(b'{"status":"incomplete"}\n')
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha_file(p)}
        for p in source.iterdir()
    }
    return {
        "caps": {"raw_bytes": 4 * 1024**2, "wall_seconds": 30},
        "source_files": {},
        "corpora": [{"id": "fixture", "root": str(source), "files": files}],
    }


def test_corpus_retains_every_logical_file_and_fault_log(tmp_path, monkeypatch):
    p = packet(tmp_path)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "test-head")
    r = module.qualify(p, tmp_path, tmp_path / "out")
    assert r["status"] == "passed" and r["new_physics"] == 0 and r["passed_files"] == 3
    assert len(r["rows"]) == 3 and all(row["status"] == "passed" for row in r["rows"])
    assert (tmp_path / "out/packed/fixture/events.jsonl").read_bytes() == (
        tmp_path / "source/events.jsonl"
    ).read_bytes()


@pytest.mark.parametrize("fault", ["changed", "extra", "missing", "time"])
def test_failed_corpus_preserves_whole_denominator_without_retry(
    tmp_path, monkeypatch, fault
):
    p = packet(tmp_path)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "test-head")
    if fault == "changed":
        (tmp_path / "source/a.json").write_text("{}")
    elif fault == "extra":
        (tmp_path / "source/unknown.json").write_text("{}")
    elif fault == "missing":
        (tmp_path / "source/b.npz").unlink()
    else:
        p["caps"]["wall_seconds"] = -1
    r = module.qualify(p, tmp_path, tmp_path / "out")
    assert r["status"] == "failed" and r["new_physics"] == 0 and len(r["rows"]) == 3
    assert all(row["status"] == "not_run" for row in r["rows"])
    with pytest.raises(ValueError, match="Fresh"):
        module.qualify(p, tmp_path, tmp_path / "out")
