"""Saved-corpus codec qualification, with no new physical or actor transitions."""

import json
import os
import resource
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import mujoco
import numpy as np

from .evidence_storage import array_descriptor, read_arrays, read_json
from .motor_codec_independent import independent_decode
from .motor_evidence_codec import (
    blob_sha,
    context_from_trace,
    decode,
    encode,
    write_blob,
)
from .motor_storage import RawCredit, append_json, write_json
from .provenance import digest
from .scene import SOURCE, build_scene

PACKET = "experiments/v0.0.11/motor-codec-qualification-protocol.json"
RUN = "runs/v0.0.11/motor-codec-qualification-01"
ROW_IDS = [
    "source-71000-full-memory-roundtrip",
    "source-71001-incomplete-recognition",
    "source-71000-prefix60-disk-roundtrip",
]


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def read_packet(root, protocol_sha, source_sha):
    p = json.loads((root / PACKET).read_text())
    require(
        digest(p) == protocol_sha
        and p["schema"] == "joxing-motor-codec-qualification-v1",
        "Codec protocol hash/schema",
    )
    require(
        p["rows"] == ROW_IDS
        and p["new_physics"] == 0
        and p["raw_bytes"] == 4 * 1024**2
        and p["post_gate_wall_s"] == 90
        and p["maximum_rss_bytes"] == 512 * 1024**2,
        "Codec fixed corpus and caps",
    )
    require(
        p["prior_v11_physics"] == 12430
        and p["threads_per_process"] == 1
        and p["formal_simulation_workers"] == 0,
        "Codec physics/thread scope",
    )
    require(
        p["prior_v11_raw_upper_bytes"] + p["raw_bytes"] <= 256 * 1024**2
        and p["prior_v11_postgate_upper_s"] + p["post_gate_wall_s"] <= 900,
        "Codec whole-version credit",
    )
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        == source_sha,
        "Codec frozen source HEAD",
    )
    require(
        json.loads(
            subprocess.check_output(
                ["git", "show", "HEAD:" + PACKET], cwd=root, text=True
            )
        )
        == p,
        "Codec committed protocol",
    )
    verify_files(p, root)
    return p


def verify_files(p, root):
    for group in ("source_files", "upstream_files", "reference_files"):
        for n, h in p[group].items():
            require(sha_file(root / n) == h, "Codec frozen source file: " + n)
    require(
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=root, check=False
        ).returncode
        == 0,
        "Codec source diff",
    )
    for n, d in p["input_files"].items():
        f = Path(p["proof_root"]) / n
        require(
            f.stat().st_size == d["bytes"] and sha_file(f) == d["sha256"],
            "Codec immutable corpus: " + n,
        )


def originals(p):
    run = Path(p["proof_root"]) / "runs/v0.0.11/shared-motor-proof-01"
    trace = read_json(run / "source-71000/trace.json")
    arrays = read_arrays(run / "source-71000/tape.npz")
    scene = build_scene(spawn_variant="parallel-walk-v1")
    require(
        scene.report["scene_hash"] == trace["contract"]["scene_hash"],
        "Codec exact original model",
    )
    canonical = mujoco.MjModel.from_xml_path(str(SOURCE))
    c, refs = context_from_trace(scene.model, canonical, trace)
    return trace, arrays, c, refs


def prefix60(trace, arrays):
    n = int(np.searchsorted(arrays["decision_ticks"], 60))
    contacts = int(arrays["contact_offsets"][60])
    v = int(np.searchsorted(arrays["video_ticks"], 60, side="right"))
    partial = {}
    for k, a in arrays.items():
        if k.startswith("witness_") or k in (
            "decision_ticks",
            "decision_engines",
            "raw",
            "normalized",
            "commands",
            "excitation",
        ):
            partial[k] = a[:n].copy()
        elif k in ("contacts", "contact_geoms"):
            partial[k] = a[:contacts].copy()
        elif k == "contact_offsets":
            partial[k] = a[:61].copy()
        elif k.startswith("video_"):
            partial[k] = a[:v].copy()
        else:
            partial[k] = a[:60].copy()
    metadata = {
        "contract": trace["contract"],
        "normalizer": trace["normalizer"],
        "decisions": trace["decisions"][:n],
        "source_trace_sha256": digest(trace),
        "scope": "Exact first60interval codec view; not a new physical trial",
    }
    return metadata, partial


def qualify(p, root, out, source_sha, protocol_sha):
    out.mkdir(parents=True, exist_ok=False)
    credit = RawCredit(out, p["raw_bytes"] - 256 * 1024)
    started = time.monotonic()
    rows = [{"id": n, "status": "not_run", "reason": "pending"} for n in ROW_IDS]
    result = {
        "schema": "joxing-motor-codec-qualification-result-v1",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "status": "running",
        "rows": rows,
        "new_physics": 0,
        "optimizer_updates": 0,
        "whole_motor_proof_passed": False,
        "process": {
            "pid": os.getpid(),
            "start_ticks": (
                int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
                if Path("/proc/self/stat").exists()
                else None
            ),
        },
    }
    write_json(out / "result.json", result)
    write_json(out / "protocol.json", p)
    current = None

    def budget():
        rss = peak_rss()
        if time.monotonic() - started >= 60 or rss > p["maximum_rss_bytes"]:
            raise TimeoutError("Codec first wall/RSS cap")
        write_json(
            out / "progress.json",
            {
                "wall_s": time.monotonic() - started,
                "rss_peak_bytes": rss,
                "row": current,
                "new_physics": 0,
            },
        )

    def finish(row):
        row["status"] = "passed"
        append_json(
            out / "events.jsonl",
            {"row": row["id"], "status": "passed", "new_physics": 0},
        )
        write_json(out / "result.json", result)
        budget()

    try:
        current = ROW_IDS[0]
        rows[0]["status"] = "running"
        budget()
        trace, arrays, c, refs = originals(p)
        budget()
        expected = {k: array_descriptor(a) for k, a in arrays.items()}
        t = time.monotonic()
        blob = encode(arrays, c, refs)
        rows[0].update(
            encoded_bytes=len(blob),
            encoded_sha256=blob_sha(blob),
            encode_s=time.monotonic() - t,
            context_sha256=digest(c),
            logical=expected,
            scope="Full encoded corpus is a bounded in-memory intermediate; immutable original retained",
        )
        budget()
        for name in ("production", "independent"):
            t = time.monotonic()
            restored = (
                decode(blob, expected_context_sha256=digest(c))
                if name == "production"
                else independent_decode(blob, c, refs, expected)
            )
            require(
                {k: array_descriptor(a) for k, a in restored.items()} == expected,
                "Codec complete corpus byte parity",
            )
            rows[0][name + "_decode_s"] = time.monotonic() - t
            del restored
            budget()
        del blob
        finish(rows[0])
        current = ROW_IDS[1]
        rows[1]["status"] = "running"
        path = (
            Path(p["proof_root"])
            / "runs/v0.0.11/shared-motor-proof-01/source-71001/tape.npz.tmp"
        )
        with zipfile.ZipFile(path) as z:
            names = [i.filename for i in z.infolist()]
        require(
            names
            == [
                n + ".npy"
                for n in [
                    "decision_ticks",
                    "decision_engines",
                    "raw",
                    "normalized",
                    "commands",
                    "excitation",
                    "ticks",
                ]
            ],
            "Codec original seven-member partial prefix",
        )
        rows[1].update(
            complete=False,
            original_members=names,
            padded_arrays=0,
            scope="Only incomplete membership recognized; no physical arrays recovered or prefix values qualified",
        )
        finish(rows[1])
        current = ROW_IDS[2]
        rows[2]["status"] = "running"
        partial_trace, shard = prefix60(trace, arrays)
        del arrays, refs
        scene = build_scene(spawn_variant="parallel-walk-v1")
        canonical = mujoco.MjModel.from_xml_path(str(SOURCE))
        sc, sr = context_from_trace(scene.model, canonical, partial_trace)
        blob = encode(shard, sc, sr)
        write_blob(out / "prefix60.jmc", blob)
        expected = {k: array_descriptor(a) for k, a in shard.items()}
        restored = independent_decode(
            (out / "prefix60.jmc").read_bytes(), sc, sr, expected
        )
        require(
            {k: array_descriptor(a) for k, a in restored.items()} == expected,
            "Codec actual persisted shard",
        )
        rows[2].update(
            encoded_bytes=len(blob),
            encoded_sha256=blob_sha(blob),
            context_sha256=digest(sc),
            logical=expected,
            scope="Stored exact60interval view of original; no new simulation",
        )
        finish(rows[2])
        verify_files(p, root)
        result["status"] = "passed"
    except Exception as error:  # noqa: BLE001 -- all corpus/fault rows retained
        import traceback

        result.update(status="halted", error=type(error).__name__ + ": " + str(error))
        for row in rows:
            if row["status"] == "running":
                row.update(status="failed", reason=result["error"])
        with credit.emergency():
            write_json(
                out / "fault.json",
                {
                    "error": result["error"],
                    "traceback": traceback.format_exc(),
                    "new_physics": 0,
                },
            )
    finally:
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "stopped after earlier fault"
        result.update(
            wall_s=time.monotonic() - started,
            max_rss_bytes=peak_rss(),
        )
        with credit.emergency():
            write_json(out / "result.json", result)
    return result


def audit_qualification(p, root, out, source_sha, protocol_sha):
    result = json.loads((out / "result.json").read_text())
    require(
        result["source_commit"] == source_sha
        and result["protocol_sha256"] == protocol_sha
        and result["status"] == "passed"
        and [r["id"] for r in result["rows"]] == ROW_IDS
        and all(r["status"] == "passed" for r in result["rows"])
        and result["new_physics"] == 0,
        "Codec qualification rows",
    )
    trace, arrays, c, refs = originals(p)
    blob = encode(arrays, c, refs)
    row = result["rows"][0]
    require(
        blob_sha(blob) == row["encoded_sha256"] and len(blob) == row["encoded_bytes"],
        "Codec deterministic ephemeral corpus",
    )
    decoded = independent_decode(blob, c, refs, row["logical"])
    require(
        all(
            array_descriptor(decoded[k]) == array_descriptor(a)
            for k, a in arrays.items()
        ),
        "Codec independent original corpus",
    )
    del blob, decoded
    pt, _shard = prefix60(trace, arrays)
    scene = build_scene(spawn_variant="parallel-walk-v1")
    canonical = mujoco.MjModel.from_xml_path(str(SOURCE))
    sc, sr = context_from_trace(scene.model, canonical, pt)
    data = (out / "prefix60.jmc").read_bytes()
    require(
        blob_sha(data) == result["rows"][2]["encoded_sha256"],
        "Codec persisted shard checksum",
    )
    independent_decode(data, sc, sr, result["rows"][2]["logical"])
    require(
        peak_rss() <= p["maximum_rss_bytes"],
        "Codec independent RSS",
    )
    verify_files(p, root)
    return {
        "schema": "joxing-motor-codec-independent-audit-v1",
        "status": "passed",
        "source_commit": source_sha,
        "protocol_sha256": protocol_sha,
        "complete_logical_arrays": len(arrays),
        "new_physics": 0,
        "full_encoded_bytes": row["encoded_bytes"],
        "full_encoded_sha256": row["encoded_sha256"],
        "persisted_shard_sha256": blob_sha(data),
        "whole_motor_proof_passed": False,
        "max_rss_bytes": peak_rss(),
        "scope": "Independent second-process regeneration/standard LZMA+NumPy decode; no tail references or physical continuation qualified",
    }
