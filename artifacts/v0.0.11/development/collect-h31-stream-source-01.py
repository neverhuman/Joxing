"""Read-only source receipt collector; never launches native simulation."""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path.cwd()
dest = root / "artifacts/v0.0.11/native-pair-streaming-source-01"
dest.mkdir(exist_ok=True)
assert not (dest / "summary.json").exists()
names = ["src/joxing/native_pair_motor_evidence.py", "src/joxing/native_pair_motor_vec.py",
         "src/joxing/native_pair_motor_train.py", "tests/test_native_pair_motor_evidence.py",
         "tests/test_native_pair_motor_rl.py"]
files = {n: {"bytes": (root/n).stat().st_size, "sha256": hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in names}
checks = []
for p in sorted(dest.glob("*.xml")):
    e = ET.parse(p).getroot()
    cases = e.findall(".//testcase")
    checks.append({"file": p.name, "cases": len(cases),
                   "failed": sum(c.find("failure") is not None for c in cases),
                   "errors": sum(c.find("error") is not None for c in cases),
                   "skipped": sum(c.find("skipped") is not None for c in cases),
                   "time_s": sum(float(s.attrib["time"]) for s in e.findall(".//testsuite")),
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
base = root / "runs/v0.0.11/h31-stream-fixtures-03/test_native_streaming_evidence0"
manifest = {str(p.relative_to(base)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(base.rglob("*")) if p.is_file() and not p.is_symlink()}
prefix = "rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -m pytest "
summary = {
    "schema": "joxing-native-streaming-source-development-v1",
    "scope": "Development source controls only: real native engine, synthetic teacher; not formal released-controller training or a boxing result",
    "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_files": files, "checks": checks,
    "native_fixture": json.loads((dest/"native-fixture-comparison.json").read_text()),
    "raw_fixture_root": str(base), "raw_native_fixture_manifest": manifest,
    "storage_forecasts": ["../development/h31-stream-storage-forecast-01.json", "../development/h31-stream-storage-forecast-02.json"],
    "new_formal_native_attempts": 0, "qualified_boxing_games": 0,
    "source_physics": {"dev01": 0, "dev02": 883, "dev03": 883, "regression01": 0, "total": 1766,
                       "kind": "Static count: original native fixture loops563 plus new16joint intervals320 per dev02/dev03; not formal physics credit"},
    "retained_faults": [
        "dev01:28/29pass; RAW rejection test allocated300000B, legitimately enough for one row. Corrected fixture to200000B; no production cap changed.",
        "Optional .venv/bin/ruff path missing; cached uvx --offline ruff used, no runtime installation.",
        "Ruff literal/iterator style findings corrected; final check passes.",
        "dev02 lacked explicit thread-limit environment; no resource/throughput qualification claimed. dev03 and regression01 explicitly use five thread environment limits=1.",
        "Initial metadata collector raised relative-vs-absolute Path.relative_to mismatch after copying five small receipts. No source/native writes; corrected collector preserves those copies and adds summary.",
    ],
    "commands": [
        "rtk proxy .venv/bin/python -m pytest tests/test_native_pair_motor_evidence.py -q --junitxml=runs/v0.0.11/h31-stream-dev-01.xml",
        "rtk proxy .venv/bin/python -m pytest tests/test_native_pair_motor_evidence.py tests/test_native_pair_motor_rl.py tests/test_native_pair_motor_vec.py -q --junitxml=runs/v0.0.11/h31-stream-dev-02.xml --basetemp=runs/v0.0.11/h31-stream-fixtures-02",
        prefix + "tests/test_native_pair_motor_evidence.py tests/test_native_pair_motor_rl.py tests/test_native_pair_motor_vec.py -q --junitxml=runs/v0.0.11/h31-stream-dev-03.xml --basetemp=runs/v0.0.11/h31-stream-fixtures-03",
        prefix + "tests/test_history.py tests/test_rules.py tests/test_match_referee.py tests/test_native_tape_storage.py -q --junitxml=runs/v0.0.11/h31-stream-regression-01.xml --basetemp=runs/v0.0.11/h31-stream-regression-fixtures-01",
        "rtk proxy uvx --offline ruff check " + " ".join(names),
    ],
    "sampled_dev03_process": {"pid": 42646, "start": "Mon Nov  2 23:18:50 2026", "rss_bytes": 2745632*1024,
                              "cpu_percent": 95.2, "scope": "One live sample; not peak/full resource qualification"},
    "fresh_capacity": "../development/four-host-capacity-20261103-13.json",
    "limits": ["Synthetic teacher source fixture; no new released-model improvement outcome",
               "Exact reconstruction against original recorder, not new independent physics replay",
               "Checkpoint/evaluation/source/media guarded writers, owner and independent pilot audit still required",
               "Old dedicated prototype source/protocol untouched; wholev11inprogress/untagged"],
    "next": "Guard full checkpoint/continuation cadence and run_native_pilot for2x1024joint steps/3s fixed evals; publish protocol/source before launch.2x2048worst-reset forecast exceeds1GiB."
}
(dest/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps({"checks": checks, "native_comparison_fraction": summary["native_fixture"]["fraction"],
                  "formal_native_attempts": 0, "summary": str(dest/"summary.json")}))
