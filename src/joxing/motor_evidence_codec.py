"""Lossless motor measurement recipes plus temporal XOR and byte planes.

Never steps or calls forward. Actual solver caches stay literal float64 values.
Derived fields are omitted ONLY after every original logical byte agrees. A
decoder needs an externally pinned context hash; self-described hashes alone
are not authentication. This first format has no external/tail dependencies.
"""

import hashlib
import lzma
import math
import struct
from pathlib import Path

import mujoco
import numpy as np

from .env import INTEGRATION
from .evidence_storage import _atomic, _parse, array_descriptor, canonical
from .provenance import digest
from .shared_motor_audit import SemanticInputAudit

MAGIC = b"JOXMC001"
SCHEMA = "joxing-motor-measurement-codec-v1"
MAX_STORED = 32 * 1024**2
MAX_DECODED = 128 * 1024**2
MAX_HEADER = 512 * 1024
DERIVED = {
    "raw",
    "normalized",
    "excitation",
    "witness_qpos",
    "witness_qvel",
    "witness_act",
}
KEYS = {
    "decision_ticks",
    "decision_engines",
    "raw",
    "normalized",
    "commands",
    "excitation",
    "ticks",
    "end_qpos",
    "posture",
    "loads",
    "contact_offsets",
    "contact_geoms",
    "contacts",
    "video_ticks",
    "video_states",
    "witness_qpos",
    "witness_qvel",
    "witness_act",
    "witness_qacc",
    "witness_actuator_force",
    "witness_actuator_length",
    "witness_actuator_velocity",
    "witness_key_xpos",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def context_from_trace(model, canonical_model, trace):
    """Compile semantic indices/state components once; no engine calculation."""
    mapping = SemanticInputAudit(model, canonical_model)
    fields = {}
    for name, flag in [
        ("qpos", mujoco.mjtState.mjSTATE_QPOS),
        ("qvel", mujoco.mjtState.mjSTATE_QVEL),
        ("act", mujoco.mjtState.mjSTATE_ACT),
    ]:
        start = mujoco.mj_stateSize(model, int(INTEGRATION) & (int(flag) - 1))
        fields[name] = [start, start + mujoco.mj_stateSize(model, flag)]
    refs = np.asarray(
        [
            [
                np.r_[
                    row["references"][p]["qpos"],
                    np.asarray(row["references"][p]["future_qpos"]).ravel(),
                    np.asarray(
                        row["references"][p]["keypoints_relative_pelvis"]
                    ).ravel(),
                ]
                for p in ("A", "B")
            ]
            for row in trace["decisions"]
        ],
        np.float64,
    ).reshape(-1, 2, 528)
    metadata = {
        "schema": "joxing-motor-codec-context-v1",
        "trace_sha256": digest(trace),
        "scene_hash": trace["contract"]["scene_hash"],
        "state_specification": int(INTEGRATION),
        "state_width": mujoco.mj_stateSize(model, INTEGRATION),
        "fields": fields,
        "normalizer": trace["normalizer"],
        "references": array_descriptor(refs),
        "maps": {
            p: {
                "qpos": mapping.joints[p][0].tolist(),
                "qvel": mapping.joints[p][1].tolist(),
                "act": mapping.activation[p].tolist(),
                "muscles": mapping.muscles[p].tolist(),
            }
            for p in ("A", "B")
        },
    }
    validate_context(metadata)
    return metadata, refs


def validate_context(c):
    require(
        set(c)
        == {
            "schema",
            "trace_sha256",
            "scene_hash",
            "state_specification",
            "state_width",
            "fields",
            "normalizer",
            "references",
            "maps",
        },
        "Codec context fields",
    )
    require(
        c["schema"] == "joxing-motor-codec-context-v1"
        and c["state_specification"] == int(INTEGRATION),
        "Codec context schema/state",
    )
    require(
        type(c["state_width"]) is int and 1741 <= c["state_width"] <= 20000,
        "Codec engine width",
    )
    require(
        set(c["fields"]) == {"qpos", "qvel", "act"} and set(c["maps"]) == {"A", "B"},
        "Codec semantic owners/fields",
    )
    ranges = []
    for k, n in [("qpos", 170), ("qvel", 170), ("act", 1400)]:
        pair = c["fields"][k]
        require(
            type(pair) is list
            and len(pair) == 2
            and all(type(v) is int for v in pair)
            and 1 <= pair[0] < pair[1] <= c["state_width"]
            and pair[1] - pair[0] == n,
            "Codec engine component",
        )
        ranges.extend(range(*pair))
    require(len(set(ranges)) == len(ranges), "Codec overlapping engine components")
    for key, size, maximum in [
        ("qpos", 85, 170),
        ("qvel", 85, 170),
        ("act", 700, 1400),
        ("muscles", 700, 1400),
    ]:
        owners = []
        for p in ("A", "B"):
            require(
                set(c["maps"][p]) == {"qpos", "qvel", "act", "muscles"},
                "Codec map fields",
            )
            ids = c["maps"][p][key]
            require(
                type(ids) is list
                and len(ids) == size
                and all(type(i) is int and 0 <= i < maximum for i in ids),
                "Codec semantic index",
            )
            owners.extend(ids)
        require(len(set(owners)) == 2 * size, "Codec duplicated/overlapping owners")
    n = c["normalizer"]
    require(
        n["training"] is False and n["norm_reward"] is False and n["norm_obs"] is True,
        "Codec frozen normalization",
    )
    require(
        np.asarray(n["mean"]).shape == np.asarray(n["var"]).shape == (3601,)
        and np.isfinite([*n["mean"], *n["var"], n["epsilon"], n["clip"]]).all()
        and min(n["var"]) >= 0
        and n["epsilon"] > 0
        and n["clip"] > 0,
        "Codec normalizer values",
    )


def _description(d):
    require(
        type(d) is dict and set(d) == {"dtype", "shape", "bytes", "sha256"},
        "Codec array descriptor",
    )
    require(
        type(d["dtype"]) is str and d["dtype"] in {"<f8", "<f4", "<i8", "|b1"},
        "Codec primitive precision",
    )
    dtype, shape = np.dtype(d["dtype"]), d["shape"]
    require(
        type(shape) is list
        and len(shape) <= 8
        and all(type(v) is int and 0 <= v <= 1000000 for v in shape),
        "Codec shape bound",
    )
    size = math.prod(shape) * dtype.itemsize
    require(
        type(d["bytes"]) is int and size == d["bytes"] and size <= MAX_DECODED,
        "Codec array byte bound",
    )
    require(
        type(d["sha256"]) is str
        and len(d["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in d["sha256"]),
        "Codec array digest",
    )
    return dtype, tuple(shape)


def validate_shapes(descriptions, context):
    """Check dependent shapes BEFORE recipe allocation or decompression."""
    require(set(descriptions) == KEYS, "Codec complete shape denominator")
    for d in descriptions.values():
        _description(d)
    n = descriptions["decision_ticks"]["shape"]
    t = descriptions["ticks"]["shape"]
    v = descriptions["video_ticks"]["shape"]
    contact = descriptions["contacts"]["shape"]
    require(
        len(n) == len(t) == len(v) == 1 and len(contact) == 2, "Codec clock dimensions"
    )
    n, t, v, contact = n[0], t[0], v[0], contact[0]
    require(n <= 1000 and t <= 20000 and v <= 1000, "Codec motor horizon bound")
    shapes = {
        "decision_ticks": (n,),
        "decision_engines": (n, context["state_width"]),
        "raw": (n, 2, 3601),
        "normalized": (n, 2, 3601),
        "commands": (n, 2, 700),
        "excitation": (n, 2, 700),
        "ticks": (t,),
        "end_qpos": (t, 170),
        "posture": (t, 2, 2),
        "loads": (t, 2, 3),
        "contact_offsets": (t + 1,),
        "contact_geoms": (contact, 2),
        "contacts": (contact, 18),
        "video_ticks": (v,),
        "video_states": (v, context["state_width"]),
        "witness_key_xpos": (n, 2, 7, 3),
    }
    for k, width in [
        ("qpos", 170),
        ("qvel", 170),
        ("act", 1400),
        ("qacc", 170),
        ("actuator_force", 1400),
        ("actuator_length", 1400),
        ("actuator_velocity", 1400),
    ]:
        shapes["witness_" + k] = (n, width)
    integers = {
        "ticks",
        "decision_ticks",
        "contact_offsets",
        "contact_geoms",
        "video_ticks",
    }
    singles = {"raw", "normalized", "commands"}
    for k, shape in shapes.items():
        require(
            descriptions[k]["shape"] == list(shape)
            and descriptions[k]["dtype"]
            == ("<i8" if k in integers else "<f4" if k in singles else "<f8"),
            "Codec dependent shape/type: " + k,
        )
    _description(context["references"])
    require(
        context["references"]["shape"] == [n, 2, 528]
        and context["references"]["dtype"] == "<f8",
        "Codec reference dimensions",
    )


def xor_planes(array):
    """Operate on bytes, preserving signed zero, endian and every mantissa bit."""
    a = np.asarray(array)
    if not a.size:
        return b""
    rows = a.shape[0] if a.ndim else 1
    original = np.frombuffer(a.tobytes(order="C"), np.uint8).reshape(rows, -1)
    delta = original.copy()
    delta[1:] ^= original[:-1]
    return delta.reshape(-1, a.dtype.itemsize).T.copy().tobytes()


def unxor_planes(raw, dtype, shape):
    count = math.prod(shape)
    if not count:
        return np.empty(shape, dtype)
    rows = shape[0] if shape else 1
    delta = (
        np.frombuffer(raw, np.uint8)
        .reshape(dtype.itemsize, count)
        .T.copy()
        .reshape(rows, -1)
    )
    restored = np.bitwise_xor.accumulate(delta, axis=0)
    return np.frombuffer(restored.tobytes(), dtype).reshape(shape).copy()


def derive(arrays, c, refs):
    """Exact recipes, using saved measurements only. No fresh dynamics readout."""
    validate_context(c)
    require(array_descriptor(refs) == c["references"], "Codec reference identity")
    n = len(arrays["decision_ticks"])
    require(
        refs.dtype == np.float64 and refs.shape == (n, 2, 528), "Codec reference shape"
    )
    engine = arrays["decision_engines"]
    require(
        engine.dtype == np.float64 and engine.shape == (n, c["state_width"]),
        "Codec measured engine shape",
    )
    result = {
        "witness_" + k: engine[:, s:e].copy() for k, (s, e) in c["fields"].items()
    }
    blocks = result | arrays
    raw = np.empty((n, 2, 3601), np.float32)
    for pi, p in enumerate(("A", "B")):
        m = c["maps"][p]
        xyz = blocks["witness_key_xpos"][:, pi]
        raw[:, pi] = np.concatenate(
            [
                blocks["witness_qpos"][:, m["qpos"]],
                blocks["witness_qvel"][:, m["qvel"]],
                blocks["witness_qacc"][:, m["qvel"]],
                blocks["witness_act"][:, m["act"]],
                np.clip(
                    blocks["witness_actuator_force"][:, m["muscles"]] / 1000, -100, 100
                ),
                blocks["witness_actuator_length"][:, m["muscles"]],
                np.clip(
                    blocks["witness_actuator_velocity"][:, m["muscles"]], -100, 100
                ),
                (xyz[:, 1:] - xyz[:, :1]).reshape(n, 18),
                refs[:, pi],
            ],
            axis=1,
        ).astype(np.float32)
    norm = c["normalizer"]
    normalized = np.clip(
        (raw - np.asarray(norm["mean"]))
        / np.sqrt(np.asarray(norm["var"]) + norm["epsilon"]),
        -norm["clip"],
        norm["clip"],
    ).astype(np.float32)
    command = arrays["commands"]
    require(
        command.dtype == np.float32
        and command.shape == (n, 2, 700)
        and np.all(abs(command) <= 1),
        "Codec normalized commands",
    )
    result.update(
        raw=raw,
        normalized=normalized,
        excitation=(1 / (1 + np.exp(-5 * (command - 0.5)))).astype(np.float64),
    )
    return result


def encode(arrays, context, references):
    require(set(arrays) == KEYS, "Codec complete 23-array denominator")
    descriptions = {k: array_descriptor(a) for k, a in arrays.items()}
    validate_shapes(descriptions, context)
    for d in descriptions.values():
        _description(d)
    require(
        sum(d["bytes"] for d in descriptions.values()) <= MAX_DECODED,
        "Codec logical aggregate bound",
    )
    require(
        all(np.isfinite(a).all() for a in arrays.values()),
        "Codec finite motor measurements",
    )
    # No lossy omission: all six recipes must reproduce the observed bytes.
    predicted = derive(
        {k: a for k, a in arrays.items() if k not in DERIVED}, context, references
    )
    require(
        all(array_descriptor(predicted[k]) == descriptions[k] for k in DERIVED),
        "Codec actual derived measurement mismatch",
    )
    stored = {k: a for k, a in arrays.items() if k not in DERIVED} | {
        "reference_targets": references
    }
    members = [
        {"name": k, "descriptor": array_descriptor(a)}
        for k, a in sorted(stored.items())
    ]
    total = sum(v["descriptor"]["bytes"] for v in members)
    require(total <= MAX_DECODED, "Codec encoded aggregate bound")
    header = {
        "schema": SCHEMA,
        "context": context,
        "context_sha256": digest(context),
        "logical": descriptions,
        "members": members,
        "decoded_bytes": total,
        "transform": "axis0-byte-XOR_then_item-byte-planes",
        "dependencies": [],
    }
    raw_header = canonical(header)
    require(len(raw_header) <= MAX_HEADER, "Codec header bound")
    body = b"".join(xor_planes(stored[v["name"]]) for v in members)
    compressed = lzma.compress(
        body,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 8 * 1024**2}],
    )
    blob = MAGIC + struct.pack("<I", len(raw_header)) + raw_header + compressed
    require(len(blob) <= MAX_STORED, "Codec stored byte bound")
    return blob


def unpack(blob, *, expected_context_sha256):
    require(
        type(blob) is bytes and 12 <= len(blob) <= MAX_STORED and blob[:8] == MAGIC,
        "Codec magic/size",
    )
    n = struct.unpack("<I", blob[8:12])[0]
    require(0 < n <= MAX_HEADER and 12 + n < len(blob), "Codec header byte bound")
    h = _parse(blob[12 : 12 + n])
    require(
        set(h)
        == {
            "schema",
            "context",
            "context_sha256",
            "logical",
            "members",
            "decoded_bytes",
            "transform",
            "dependencies",
        },
        "Codec header fields",
    )
    require(
        h["schema"] == SCHEMA
        and h["transform"] == "axis0-byte-XOR_then_item-byte-planes"
        and h["dependencies"] == [],
        "Codec schema/unsupported dependency",
    )
    require(
        digest(h["context"]) == h["context_sha256"] == expected_context_sha256,
        "Codec externally pinned context",
    )
    validate_context(h["context"])
    require(set(h["logical"]) == KEYS, "Codec logical denominator")
    validate_shapes(h["logical"], h["context"])
    for d in h["logical"].values():
        _description(d)
    require(
        sum(d["bytes"] for d in h["logical"].values()) <= MAX_DECODED,
        "Codec logical aggregate bound",
    )
    require(
        type(h["members"]) is list
        and len(h["members"]) == len(KEYS) - len(DERIVED) + 1,
        "Codec member count",
    )
    names = [v["name"] for v in h["members"]]
    require(
        len(set(names)) == len(names)
        and set(names) == (KEYS - DERIVED) | {"reference_targets"},
        "Codec member names/dependencies",
    )
    for v in h["members"]:
        require(set(v) == {"name", "descriptor"}, "Codec member fields")
        _description(v["descriptor"])
        expected = (
            h["context"]["references"]
            if v["name"] == "reference_targets"
            else h["logical"][v["name"]]
        )
        require(v["descriptor"] == expected, "Codec stored/logical descriptor")
    total = sum(v["descriptor"]["bytes"] for v in h["members"])
    require(
        type(h["decoded_bytes"]) is int and total == h["decoded_bytes"] <= MAX_DECODED,
        "Codec decompression byte bound",
    )
    decoder = lzma.LZMADecompressor(lzma.FORMAT_XZ, memlimit=16 * 1024**2)
    try:
        body = decoder.decompress(blob[12 + n :], max_length=total + 1)
    except lzma.LZMAError as e:
        raise ValueError("Codec bounded LZMA stream") from e
    require(
        len(body) == total and decoder.eof and not decoder.unused_data,
        "Codec decoded length/trailing stream",
    )
    return h, body


def decode(blob, *, expected_context_sha256):
    h, body = unpack(blob, expected_context_sha256=expected_context_sha256)
    arrays, cursor = {}, 0
    for member in h["members"]:
        d = member["descriptor"]
        dtype, shape = _description(d)
        arrays[member["name"]] = unxor_planes(
            body[cursor : cursor + d["bytes"]], dtype, shape
        )
        require(
            array_descriptor(arrays[member["name"]]) == d, "Codec stored value digest"
        )
        cursor += d["bytes"]
    refs = arrays.pop("reference_targets")
    arrays.update(derive(arrays, h["context"], refs))
    require(
        {k: array_descriptor(v) for k, v in arrays.items()} == h["logical"],
        "Codec logical value digest",
    )
    return arrays


def write_blob(path, blob):
    """Existing atomic credit sink counts compressed growth and keeps faults."""
    require(type(blob) is bytes and len(blob) <= MAX_STORED, "Codec write bound")
    path = Path(path)
    require(not path.exists(), "Never overwrite a codec proof")
    with _atomic(path) as sink:
        sink.write(blob)


def blob_sha(blob):
    return hashlib.sha256(blob).hexdigest()
