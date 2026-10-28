"""Qualification-only standard decoder, independent of codec implementation.

Receives the immutable baseline descriptors/context from the preregistered
corpus. No calls to production unpack/XOR/recipe functions. No physics/forward.
"""

import hashlib
import json
import lzma
import math
import struct

import numpy as np


def check(value, message):
    if not value:
        raise ValueError(message)


def independent_decode(blob, context, references, expected):
    check(
        type(blob) is bytes
        and 12 <= len(blob) <= 32 * 1024**2
        and blob[:8] == b"JOXMC001",
        "Independent codec container",
    )
    count = struct.unpack("<I", blob[8:12])[0]
    check(
        0 < count <= 512 * 1024 and 12 + count < len(blob),
        "Independent codec header bound",
    )

    def pairs(items):
        result = {}
        for k, v in items:
            check(k not in result, "Independent duplicate JSON key")
            result[k] = v
        return result

    h = json.loads(blob[12 : 12 + count], object_pairs_hook=pairs)
    canonical = json.dumps(
        context, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    check(
        h["context"] == context
        and h["context_sha256"] == hashlib.sha256(canonical).hexdigest(),
        "Independent external context",
    )
    check(
        h["logical"] == expected
        and h["dependencies"] == []
        and h["schema"] == "joxing-motor-measurement-codec-v1"
        and h["transform"] == "axis0-byte-XOR_then_item-byte-planes",
        "Independent logical schema",
    )
    derived = {
        "raw",
        "normalized",
        "excitation",
        "witness_qpos",
        "witness_qvel",
        "witness_act",
    }
    names = [v["name"] for v in h["members"]]
    check(
        len(set(names)) == len(names)
        and set(names) == (set(expected) - derived) | {"reference_targets"},
        "Independent member denominator",
    )
    total = 0
    for row in h["members"]:
        d = (
            context["references"]
            if row["name"] == "reference_targets"
            else expected[row["name"]]
        )
        check(
            row["descriptor"] == d and d["dtype"] in ("<f4", "<f8", "<i8", "|b1"),
            "Independent member type",
        )
        check(
            len(d["shape"]) <= 8
            and all(type(i) is int and 0 <= i <= 1000000 for i in d["shape"]),
            "Independent shape bound",
        )
        check(
            math.prod(d["shape"]) * np.dtype(d["dtype"]).itemsize == d["bytes"],
            "Independent shape bytes",
        )
        total += d["bytes"]
    check(
        total == h["decoded_bytes"] and total <= 128 * 1024**2,
        "Independent aggregate bound",
    )
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=16 * 1024**2)
    data = dec.decompress(blob[12 + count :], max_length=total + 1)
    check(
        len(data) == total and dec.eof and not dec.unused_data,
        "Independent LZMA extent",
    )
    result, cursor = {}, 0
    for row in h["members"]:
        d = row["descriptor"]
        dtype, shape = np.dtype(d["dtype"]), tuple(d["shape"])
        part = data[cursor : cursor + d["bytes"]]
        cursor += d["bytes"]
        if math.prod(shape):
            # Undo byte-plane transpose, then cumulative byte XOR row by row.
            byte_rows = (
                np.frombuffer(part, np.uint8).reshape(dtype.itemsize, -1).T.copy()
            )
            time_rows = byte_rows.reshape(shape[0] if shape else 1, -1)
            for i in range(1, len(time_rows)):
                time_rows[i] ^= time_rows[i - 1]
            value = np.frombuffer(time_rows.tobytes(), dtype).reshape(shape).copy()
        else:
            value = np.empty(shape, dtype)
        check(
            hashlib.sha256(value.tobytes()).hexdigest() == d["sha256"],
            "Independent literal bytes",
        )
        result[row["name"]] = value
    check(
        result.pop("reference_targets").tobytes() == references.tobytes(),
        "Independent original references",
    )
    for k in ("qpos", "qvel", "act"):
        start, end = context["fields"][k]
        result["witness_" + k] = result["decision_engines"][:, start:end].copy()
    decisions = len(result["decision_ticks"])
    raw = np.empty((decisions, 2, 3601), np.float32)
    for i in range(decisions):
        for pi, p in enumerate(("A", "B")):
            ids = context["maps"][p]
            points = result["witness_key_xpos"][i, pi]
            features = [
                result["witness_qpos"][i, ids["qpos"]],
                result["witness_qvel"][i, ids["qvel"]],
                result["witness_qacc"][i, ids["qvel"]],
                result["witness_act"][i, ids["act"]],
                np.clip(
                    result["witness_actuator_force"][i, ids["muscles"]] / 1000,
                    -100,
                    100,
                ),
                result["witness_actuator_length"][i, ids["muscles"]],
                result["witness_actuator_velocity"][i, ids["muscles"]].clip(-100, 100),
                (points[1:] - points[0]).flatten(),
                references[i, pi],
            ]
            raw[i, pi] = np.hstack(features).astype(np.float32)
    result["raw"] = raw
    n = context["normalizer"]
    result["normalized"] = np.clip(
        (raw - np.asarray(n["mean"], np.float64))
        / np.sqrt(np.asarray(n["var"], np.float64) + n["epsilon"]),
        -n["clip"],
        n["clip"],
    ).astype(np.float32)
    command = result["commands"]
    result["excitation"] = np.asarray(
        1 / (1 + np.exp(-5 * (command - 0.5))), np.float64
    )
    check(set(result) == set(expected), "Independent full array denominator")
    for key, value in result.items():
        d = expected[key]
        check(
            value.dtype.str == d["dtype"]
            and list(value.shape) == d["shape"]
            and value.nbytes == d["bytes"]
            and hashlib.sha256(value.tobytes(order="C")).hexdigest() == d["sha256"],
            "Independent logical bytes: " + key,
        )
    return result
