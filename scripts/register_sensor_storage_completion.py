"""Register ONLY the two unexecuted cases with immutable prior dependencies."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from joxing.boxing_evidence import controller_bundles
from joxing.provenance import digest
from joxing.sensor_completion import PACKET, validate_packet
from joxing.sensor_protocol import sensor_cases


def descriptor(path):
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tests", required=True, type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / PACKET
    if target.exists():
        raise FileExistsError("Never overwrite a frozen packet")
    old = json.loads(
        (root / "experiments/v0.0.11/sensor-proof-repair-protocol.json").read_text()
    )
    setup_name = "artifacts/v0.0.11/development/sensor-proof-attempt-01-failure.json"
    partial_name = "artifacts/v0.0.11/D1/result.json"
    setup = json.loads((root / setup_name).read_text())
    partial = json.loads((root / partial_name).read_text())
    qualification = json.loads(
        (
            root / "artifacts/v0.0.11/development/storage-independent-audit-01.json"
        ).read_text()
    )
    if (
        qualification["status"] != "passed"
        or qualification["live_recorded"]
        or qualification["unknown_owned"]
    ):
        raise ValueError("Storage prerequisite/ownership")
    source_paths = [
        *root.glob("src/joxing/**/*.py"),
        *root.glob("scripts/*.py"),
        *root.glob("tests/*.py"),
        root / "pyproject.toml",
        root / "experiments/v0.0.11/runtime/pyproject.toml",
        root / "experiments/v0.0.11/runtime/uv.lock",
    ]
    fixed_source = "ef42c872d13bfebb16b347f842ba11f0b8b8e776"
    old_files = {}
    for name, sha in old["source_files"].items():
        raw = subprocess.check_output(
            ["git", "show", fixed_source + ":" + name], cwd=root
        )
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Prior committed source hash")
        old_files[name] = {"bytes": len(raw), "sha256": sha}
    storage_protocol = json.loads(
        (root / "experiments/v0.0.11/storage-qualification-protocol.json").read_text()
    )
    d1 = next(c for c in storage_protocol["corpora"] if c["id"] == "D1")
    core = json.loads(
        (root / "artifacts/v0.0.11/development/storage-implementation.json").read_text()
    )["unchanged_nonstorage_modules"]
    receipt_names = [
        setup_name,
        partial_name,
        "artifacts/v0.0.11/development/storage-independent-audit-01.json",
        "artifacts/v0.0.11/development/storage-implementation.json",
        "artifacts/v0.0.11/D1/partial-audit.json",
        "artifacts/v0.0.11/D1/exit-audit.json",
    ]
    p = copy.deepcopy(old)
    p.update(
        schema="joxing-sensor-storage-completion-v1",
        scope="Storage-only unfinished200+117steps; eight fixed conditions with18attempt rows. No learning, motor or native boxing claim.",
        cases=sensor_cases()[6:],
        fixed_cases=sensor_cases()[:6],
        planned_physics=317,
        physics_cap=1000,
        runner_wall_s=60,
        post_gate_wall_s=120,
        raw_bytes=24 * 1024**2,
        expected_tests=args.expected_tests,
        stage_wall_limits_s={"sync": 180, "tests": 480, "D1": 60, "audit": 60},
        source_files={
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(source_paths)
        },
        bundles=controller_bundles(),
        unchanged_core=core,
        fixed_run=d1["root"],
        fixed_files=d1["files"],
        fixed_source_commit=fixed_source,
        fixed_source_root="/home/ubuntu/joxing/v011-sensor-proof-repair-20261027",
        fixed_source_files=old_files,
        fixed_protocol_sha256=digest(old),
        receipt_files={n: descriptor(root / n) for n in receipt_names},
        prior_v11_physics=5073,
        prior_v11_raw_bytes=55053851 + 97341018 + qualification["retained_bytes"],
        prior_attempts=[
            {
                "id": "setup-01",
                "result": {
                    "source_commit": setup["supervision"]["source_commit"],
                    "protocol_sha256": setup["supervision"]["protocol_sha256"],
                    "status": "halted-before-tests",
                    "physics_attempted": 0,
                    "rows": setup["rows"],
                },
            },
            {"id": "partial-02", "result": partial},
        ],
        storage_schema="joxing-lossless-sensor-evidence-v1",
        prior_storage_qualification_source=qualification["source_sha"],
    )
    # All whole-version arithmetic is conservative and explicit, not a new cap.
    if p["prior_v11_raw_bytes"] + p["raw_bytes"] > p["whole_v11_caps"]["raw_bytes"]:
        raise ValueError("Insufficient whole-version raw credit")
    # Include bounded supplementary read-audits and video repair conservatively;
    # these are upper reserves, not invented measured runtimes.
    storage_wall = json.loads(
        (root / "artifacts/v0.0.11/storage/supervision.json").read_text()
    )["post_gate_wall_s"]
    partial_wall = json.loads(
        (root / "artifacts/v0.0.11/D1/supervision.json").read_text()
    )["post_gate_wall_s"]
    p["prior_post_gate_accounting"] = {
        "D0_measured_s": 40.476303,
        "partial_D1_measured_s": partial_wall,
        "storage_measured_s": storage_wall,
        "supplementary_upper_reserves_s": {
            "video_repair": 90,
            "partial_read_audit": 90,
            "independent_storage_read_audit": 90,
        },
    }
    p["prior_v11_post_gate_upper_s"] = 40.476303 + partial_wall + storage_wall + 270
    if (
        p["prior_v11_post_gate_upper_s"] + p["post_gate_wall_s"]
        > p["whole_v11_caps"]["post_gate_wall_s"]
    ):
        raise ValueError("Insufficient whole-version wall credit")
    validate_packet(p)
    target.write_text(json.dumps(p, indent=2) + "\n")
    print(digest(p))


if __name__ == "__main__":
    main()
