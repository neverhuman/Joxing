"""Zero-physics, hash-bound corpus conversion; never steps or rebuilds a model."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

from joxing.evidence_storage import (
    SCHEMA,
    array_descriptor,
    read_arrays,
    read_json,
    save_npz,
    sha_file,
    write_json,
)
from joxing.motor_storage import RawCredit, append_json
from joxing.motor_storage import write_json as plain_json


def verify(root, files):
    for name, expected in files.items():
        path = root / name
        if (
            path.stat().st_size != expected["bytes"]
            or sha_file(path) != expected["sha256"]
        ):
            raise ValueError("Input drift: " + str(path))


def qualify(packet, root, output):
    started = time.monotonic()
    if output.exists():
        raise ValueError("Fresh output root required")
    output.mkdir(parents=True)
    owner = RawCredit(output, packet["caps"]["raw_bytes"])
    rows = [
        {"corpus": c["id"], "path": p, "status": "not_run"}
        for c in packet["corpora"]
        for p in sorted(c["files"])
    ]
    result = {
        "schema": SCHEMA,
        "status": "running",
        "new_physics": 0,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "pid": os.getpid(),
        "rows": rows,
    }
    before_cpu = time.process_time()
    try:
        verify(root, packet["source_files"])
        # Reject omissions, additions and changed source files before conversion.
        for corpus in packet["corpora"]:
            folder = Path(corpus["root"])
            actual = {
                str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()
            }
            if actual != set(corpus["files"]):
                raise ValueError("Corpus file denominator changed: " + corpus["id"])
            verify(folder, corpus["files"])
        plain_json(output / "result.json", result)
        for corpus in packet["corpora"]:
            folder = Path(corpus["root"])
            for row in (r for r in rows if r["corpus"] == corpus["id"]):
                if time.monotonic() - started > packet["caps"]["wall_seconds"]:
                    raise TimeoutError("Storage qualification wall limit")
                row["status"] = "attempted"
                plain_json(output / "result.json", result)
                source = folder / row["path"]
                target = output / "packed" / corpus["id"] / row["path"]
                tick = time.monotonic()
                if source.suffix == ".json":
                    value = read_json(source)
                    write_json(target, value)
                    if read_json(target) != value:
                        raise ValueError("JSON parsed-value mismatch")
                    row["kind"] = "json"
                elif source.suffix == ".npz":
                    arrays = read_arrays(source)
                    expected = {k: array_descriptor(v) for k, v in arrays.items()}
                    save_npz(target, **arrays)
                    if {
                        k: array_descriptor(v) for k, v in read_arrays(target).items()
                    } != expected:
                        raise ValueError("Array dtype/shape/byte mismatch")
                    row.update(kind="npz", arrays=expected)
                else:
                    # Logs/binary evidence stay identical, not silently omitted.
                    from joxing.motor_storage import write_bytes

                    write_bytes(target, source.read_bytes())
                    if sha_file(target) != corpus["files"][row["path"]]["sha256"]:
                        raise ValueError("Opaque evidence mismatch")
                    row["kind"] = "identity"
                row.update(
                    status="passed",
                    seconds=time.monotonic() - tick,
                    source=corpus["files"][row["path"]],
                )
                append_json(
                    output / "events.jsonl",
                    {k: v for k, v in row.items() if k != "arrays"},
                )
                plain_json(output / "result.json", result)
        for corpus in packet["corpora"]:
            verify(Path(corpus["root"]), corpus["files"])
        verify(root, packet["source_files"])
        result.update(status="passed", passed_files=len(rows), failed_files=0)
    except BaseException as exc:  # noqa: BLE001 # durable evidence includes interrupts
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        for row in rows:
            if row["status"] == "attempted":
                row.update(status="failed", error=str(exc))
    finally:
        result.update(
            wall_seconds=time.monotonic() - started,
            cpu_seconds=time.process_time() - before_cpu,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            input_bytes=sum(
                f["bytes"] for c in packet["corpora"] for f in c["files"].values()
            ),
            manifests={
                str(p.relative_to(output)): sha_file(p)
                for p in output.rglob("storage-manifest.json")
            },
            limitations=[
                "Container correctness only; no new physics, game, motor or learning outcome",
                "Manifest hashes require external receipt/source anchoring",
            ],
        )
        with owner.emergency():
            plain_json(output / "result.json", result)
        result["bytes_before_final_receipt"] = owner.used()
        with owner.emergency():
            plain_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if hashlib.sha256(args.protocol.read_bytes()).hexdigest() != args.protocol_sha256:
        raise ValueError("Protocol file hash")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != args.source_sha
    ):
        raise ValueError("Frozen source HEAD")
    packet = json.loads(args.protocol.read_text())
    if packet["schema"] != "joxing-storage-qualification-protocol-v1":
        raise ValueError("Protocol schema")
    result = qualify(packet, root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
