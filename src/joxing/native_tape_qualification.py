"""Offline writer qualification only: fixed saved corpora, no model or simulation."""

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import resource
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from . import native_tape_storage as storage
from .motor_storage import RawCredit, append_json, write_json

ROOT = Path(__file__).resolve().parents[2]
PACKET = "experiments/v0.0.11/native-tape-storage-protocol.json"
RUN = "runs/v0.0.11/native-tape-storage-01"


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )


def packet(protocol_sha, source_sha):
    p = json.loads((ROOT / PACKET).read_text())
    if canonical_sha(p) != protocol_sha:
        raise ValueError("Stored qualification protocol identity")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != source_sha
        or json.loads(
            subprocess.check_output(["git", "show", "HEAD:" + PACKET], cwd=ROOT)
        )
        != p
    ):
        raise ValueError("Stored qualification source/protocol not frozen")
    for name, d in {**p["source_files"], **p["input_files"]}.items():
        f = Path(name) if name.startswith("/") else ROOT / name
        if f.stat().st_size != d["bytes"] or storage.sha_file(f) != d["sha256"]:
            raise ValueError("Stored qualification file identity: " + name)
    if (
        np.__version__ != p["numpy_version"]
        or sys.version.split()[0] != p["python_version"]
    ):
        raise ValueError("Stored qualification NumPy runtime")
    return p


def load_corpus(p, identifier):
    source = p["corpora"][identifier]
    if source["kind"] == "incomplete-native-prefix":
        file = ROOT / "experiments/v0.0.11/native_source_partial_display_01.py"
        spec = importlib.util.spec_from_file_location("qualified_partial_reader", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        arrays, prefix = module.read_complete_prefix(Path(source["path"]))
        if prefix["first_incomplete_member"] != "contacts.npy" or set(arrays) != {
            "commands",
            "observations",
            "decision_engines",
            "contact_offsets",
            "contact_geoms",
        }:
            raise ValueError("Actual incomplete prefix changed")
    elif source["kind"] == "complete-legacy-npz":
        with np.load(
            source["path"], allow_pickle=False, max_header_size=4096
        ) as archive:
            arrays = {k: archive[k] for k in archive.files}
    else:
        raise ValueError("Unknown stored qualification corpus")
    if (
        len(arrays) != source["members"]
        or sum(a.nbytes for a in arrays.values()) != source["payload_bytes"]
    ):
        raise ValueError("Stored qualification original corpus extent")
    return arrays


def check_numpy(path_or_stream, expected):
    with np.load(path_or_stream, allow_pickle=False, max_header_size=4096) as archive:
        if sorted(archive.files) != sorted(expected):
            raise ValueError("Independent NumPy member identity")
        for key, descriptor in expected.items():
            a = archive[key]
            observed = {
                "dtype": a.dtype.str,
                "shape": list(a.shape),
                "bytes": a.nbytes,
                "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest(),
            }
            if observed != descriptor:
                raise ValueError("Independent NumPy logical bytes differ")


class BoundedMemory(io.BytesIO):
    def __init__(self, maximum):
        super().__init__()
        self.maximum = maximum

    def write(self, value):
        if self.tell() + len(value) > self.maximum:
            raise ValueError("LZMA comparison memory output cap")
        return super().write(value)


def lzma_baseline(arrays, maximum):
    stream = BoundedMemory(maximum)
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_LZMA) as archive:
        for key in sorted(arrays):
            with archive.open(key + ".npy", "w") as member:
                np.lib.format.write_array(member, arrays[key], allow_pickle=False)
    return stream


def native_projection(p, arrays):
    """Size arithmetic only; missing physical values are never created or saved."""
    count = int(arrays["contact_offsets"][-1])
    width = arrays["decision_engines"].shape[1]
    layout = {k: (a.dtype.str, a.shape) for k, a in arrays.items()}
    layout.update(
        contacts=("<f8", (count, 18)),
        video_ticks=("<i8", (76,)),
        video_states=("<f8", (76, width)),
    )
    payload = 0
    for dtype, shape in layout.values():
        payload += int(np.prod(shape)) * np.dtype(dtype).itemsize
    metadata_bound = p["native_preflight"]["archive_and_receipt_metadata_bound_bytes"]
    manifest = json.loads((ROOT / p["native_preflight"]["raw_manifest"]).read_text())
    prefix = "runs/v0.0.11/native-strike-proof-01/"
    fixed = sum(
        d["bytes"]
        for n, d in manifest.items()
        if n.startswith(prefix) and not n.endswith("tape.npz.tmp")
    )
    replay_journal = manifest[prefix + "physics.jsonl"]["bytes"]
    allocations = p["native_preflight"]["additional_allocations_bytes"]
    before_video = (
        fixed + payload + metadata_bound + replay_journal + sum(allocations.values())
    )
    variants = {
        key: {
            "video_cap_bytes": n,
            "whole_raw_upper_bytes": before_video + n,
            "fits": before_video + n <= p["native_preflight"]["whole_raw_cap_bytes"],
        }
        for key, n in p["native_preflight"]["video_variants"].items()
    }
    return {
        "scope": "Future complete tape size arithmetic only, not recovered missing data or a native result",
        "projected_array_payload_bytes": payload,
        "fixed_source_bytes": fixed,
        "archive_and_receipt_metadata_bound_bytes": metadata_bound,
        "extra_replay_journal_upper_bytes": replay_journal,
        "additional_allocations_bytes": allocations,
        "variants": variants,
        "gate_variant": p["native_preflight"]["gate_variant"],
        "gate_passed": variants[p["native_preflight"]["gate_variant"]]["fits"],
        "missing_signed_contacts_remain_missing": True,
    }


def prove(p, source_sha, protocol_sha):
    out = ROOT / RUN
    out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(out, p["raw_bytes"] - p.get("supervision_reserve_bytes", 0))
    began = time.monotonic()
    result = {
        "schema": "joxing-native-stored-qualification-v1",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "status": "running",
        "new_physics": 0,
        "rows": [{"id": key, "status": "not_run"} for key in p["corpora"]],
        "native_boxing_proof_passed": False,
        "runtime": {"python": sys.version, "numpy": np.__version__},
    }
    current = None

    def event(stage, phase, row):
        append_json(
            out / "events.jsonl",
            {
                "row": row,
                "stage": stage,
                "phase": phase,
                "elapsed_s": time.monotonic() - began,
            },
        )
        if (
            time.monotonic() - began > p["stage_wall_limits_s"]["proof"]
            or rss_bytes() > p["maximum_rss_bytes"]
        ):
            raise TimeoutError("Offline writer proof wall/RSS cap")

    write_json(out / "result.json", result)
    try:
        for current in result["rows"]:
            key = current["id"]
            current.update(status="running")
            write_json(out / "result.json", result)
            event("input", "entered", key)
            arrays = load_corpus(p, key)
            event("input", "completed", key)
            event("stored-write", "entered", key)
            started = time.monotonic()
            receipt = storage.store(out / (key + ".npz"), arrays)
            stored_s = time.monotonic() - started
            write_json(out / (key + ".receipt.json"), receipt)
            event("stored-write", "completed", key)
            event("candidate-read", "entered", key)
            started = time.monotonic()
            actual = storage.load(out / (key + ".npz"), receipt)
            if storage.plan(actual)["arrays"] != receipt["arrays"]:
                raise ValueError("Stored candidate decoded payload mismatch")
            candidate_read_s = time.monotonic() - started
            del actual
            event("candidate-read", "completed", key)
            event("numpy-read", "entered", key)
            started = time.monotonic()
            check_numpy(out / (key + ".npz"), receipt["arrays"])
            numpy_read_s = time.monotonic() - started
            event("numpy-read", "completed", key)
            event("lzma-memory-write", "entered", key)
            started = time.monotonic()
            baseline = lzma_baseline(arrays, p["maximum_ephemeral_lzma_bytes"])
            lzma_s = time.monotonic() - started
            lzma_bytes = baseline.getbuffer().nbytes
            lzma_sha = hashlib.sha256(baseline.getbuffer()).hexdigest()
            event("lzma-memory-write", "completed", key)
            baseline.seek(0)
            check_numpy(baseline, receipt["arrays"])
            baseline.close()
            speedup = lzma_s / stored_s
            speed_required = p["corpora"][key]["minimum_encode_speedup"]
            current.update(
                status="passed" if speedup >= speed_required else "failed",
                integrity_passed=True,
                arrays=receipt["arrays"],
                stored_archive_bytes=receipt["archive_bytes"],
                stored_sha256=receipt["sha256"],
                stored_encode_s=stored_s,
                candidate_read_s=candidate_read_s,
                numpy_read_s=numpy_read_s,
                lzma_memory_encode_s=lzma_s,
                lzma_memory_bytes=lzma_bytes,
                lzma_memory_sha256=lzma_sha,
                encode_speedup=speedup,
                minimum_encode_speedup=speed_required,
                baseline_scope="LZMA in bounded memory; candidate includes disk fsync, planning and hashing (conservative speed comparison)",
            )
            if key == "native-prefix":
                result["native_size_preflight"] = native_projection(p, arrays)
            write_json(out / "result.json", result)
            del arrays
            gc.collect()
        result["status"] = (
            "passed"
            if all(r["status"] == "passed" for r in result["rows"])
            and result["native_size_preflight"]["gate_passed"]
            else "failed"
        )
    except Exception as error:  # noqa: BLE001 -- retain failed and not_run evidence
        import traceback

        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        if current is not None:
            current.update(status="failed", reason=result["error"])
        with credit.emergency():
            write_json(
                out / "fault.json",
                {"error": result["error"], "traceback": traceback.format_exc()},
            )
    finally:
        result.update(wall_s=time.monotonic() - began, max_rss_bytes=rss_bytes())
        with credit.emergency():
            write_json(out / "result.json", result)
    return result


def audit(p, source_sha, protocol_sha):
    began = time.monotonic()
    out = ROOT / RUN
    result = json.loads((out / "result.json").read_text())
    if (
        result["source_commit"] != source_sha
        or result["protocol_sha256"] != protocol_sha
    ):
        raise ValueError("Independent stored audit run identity")
    audited = []
    for row in result["rows"]:
        if not row.get("integrity_passed"):
            continue
        original = load_corpus(p, row["id"])
        expected = storage.plan(original)["arrays"]
        receipt = json.loads((out / (row["id"] + ".receipt.json")).read_text())
        archive = out / (row["id"] + ".npz")
        if (
            expected != row["arrays"]
            or expected != receipt["arrays"]
            or storage.sha_file(archive) != row["stored_sha256"]
            or archive.stat().st_size != row["stored_archive_bytes"]
        ):
            raise ValueError("Independent original/encoded identity")
        check_numpy(archive, expected)  # Candidate reader deliberately not used.
        audited.append(
            {
                "id": row["id"],
                "members": len(expected),
                "payload_bytes": sum(d["bytes"] for d in expected.values()),
            }
        )
        if (
            row["id"] == "native-prefix"
            and native_projection(p, original) != result["native_size_preflight"]
        ):
            raise ValueError("Independent whole-size arithmetic")
        del original
        gc.collect()
    if (
        time.monotonic() - began > p["stage_wall_limits_s"]["audit"]
        or rss_bytes() > p["maximum_rss_bytes"]
    ):
        raise TimeoutError("Independent stored audit wall/RSS cap")
    report = {
        "schema": "joxing-native-stored-saved-audit-v1",
        "status": "passed",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "audited_rows": audited,
        "all_planned_rows_audited": len(audited) == len(p["corpora"]),
        "qualification_status": result["status"],
        "candidate_reader_used": False,
        "wall_s": time.monotonic() - began,
        "max_rss_bytes": rss_bytes(),
        "new_physics": 0,
    }
    RawCredit(out, p["raw_bytes"] - p.get("supervision_reserve_bytes", 0))
    write_json(out / "independent-audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("proof", "audit"), required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise TimeoutError("Owned offline qualification interrupted: " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    p = packet(args.protocol_sha256, args.source_sha)
    result = (prove if args.stage == "proof" else audit)(
        p, args.source_sha, args.protocol_sha256
    )
    print(json.dumps(result))
    # A failed criterion is a failure even if complete receipts were written.
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
