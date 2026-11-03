"""H35 full-observation46action evidence; never pad a latent density to138.

Lossless bounded ZIP_STORED chunks with external file/array/metadata receipts.
Every1ms guard metric and ordered contact hash is retained, including endings.
Raw evidence stays outside Git. Reader and writer share a strict typed envelope.
"""

import copy
import json
import re
from pathlib import Path

import numpy as np

from .native_pair_motor_evidence import encode
from .native_tape_storage import load, plan, store
from .provenance import digest

SCHEMA = "joxing-native-arm-guard-full-chunks-v1"
MAX_ROW_TEXT = 2048
FIELDS = {
    "source_obs": ("float32", (2, 7901)),
    "next_obs": ("float32", (2, 7901)),
    "teacher_inputs": ("float32", (2, 3601)),
    "teacher_commands": ("float32", (2, 700)),
    "commands": ("float32", (2, 700)),
    "applied_actions": ("float32", (2, 46)),
    "raw_policy_actions": ("float32", (2, 46)),
    "values": ("float32", (2, 1)),
    "log_probs": ("float32", (2,)),
    "rewards": ("float64", (2,)),
    "terminated": ("bool", ()),
    "truncated": ("bool", ()),
    "engine_before": ("float64", (4555,)),
    "engine_after": ("float64", (4555,)),
    "metrics_1ms": ("float64", (20, 2, 7)),
    "contact_sha256_1ms": ("uint8", (20, 32)),
}
META = {
    "episode_id",
    "episode_serial",
    "logical_seed",
    "native_seed",
    "nonce",
    "tick",
    "behavior",
}


def checked_arrays(arrays, count):
    if set(arrays) != set(FIELDS) or type(count) is not int or not 1 <= count <= 64:
        raise ValueError("Guard trace fields/row bound")
    for key, (dtype, shape) in FIELDS.items():
        a = arrays[key]
        if (
            not isinstance(a, np.ndarray)
            or a.dtype != np.dtype(dtype)
            or a.shape != (count, *shape)
            or not np.isfinite(a).all()
        ):
            raise ValueError("Guard trace dtype/shape/finite: " + key)
    if not np.array_equal(
        np.clip(arrays["raw_policy_actions"], -1, 1), arrays["applied_actions"]
    ):
        raise ValueError("Guard trace raw46/clipped46 mismatch")
    if np.any(arrays["terminated"] & arrays["truncated"]):
        raise ValueError("Guard trace exclusive ending masks")


def checked_metadata(meta):
    if (
        type(meta) is not dict
        or set(meta) != META
        or len(encode(meta)) > MAX_ROW_TEXT
        or any(
            type(meta[k]) is not int or meta[k] < 0
            for k in ("tick", "logical_seed", "native_seed", "episode_serial")
        )
        or meta["tick"] < 20
        or meta["tick"] % 20
        or meta["episode_serial"] < 1
    ):
        raise ValueError("Guard trace metadata/clock")
    if (
        type(meta["episode_id"]) is not str
        or not re.fullmatch(r"motor-[0-9a-f]{32}", meta["episode_id"])
        or type(meta["nonce"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", meta["nonce"])
    ):
        raise ValueError("Guard trace ownership digest")
    behavior = meta["behavior"]
    if (
        type(behavior) is not dict
        or set(behavior) != {"boundary", "policy_sha256", "actor_hash"}
        or type(behavior["boundary"]) is not int
        or behavior["boundary"] < 0
    ):
        raise ValueError("Guard behavior checkpoint")
    for k in ("policy_sha256", "actor_hash"):
        v = behavior[k]
        if (
            type(v) is not str
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
        ):
            raise ValueError("Guard behavior hash")


def envelope_sha(envelope):
    return digest({k: v for k, v in envelope.items() if k != "sha256"})


class GuardTraceWriter:
    def __init__(self, output, *, contract_hash, budget, chunk_rows=64):
        if type(chunk_rows) is not int or not 1 <= chunk_rows <= 64:
            raise ValueError("Guard chunk row bound")
        self.output, self.budget = Path(output), budget
        if not self.output.resolve().is_relative_to(budget.root):
            raise ValueError("Guard trace output outside owned budget")
        self.output.mkdir(parents=True, exist_ok=False)
        self.contract_hash, self.chunk_rows = contract_hash, chunk_rows
        self.rows, self.metadata, self.chunks = [], [], []
        self.count, self.chain, self.closed = 0, "0" * 64, False

    def append(self, transition, *, raw_actions, values, log_probs):
        if self.closed:
            raise ValueError("Closed guard trace")
        t = transition
        pair = lambda key: np.stack([t[key][p] for p in "AB"])
        row = {
            "source_obs": pair("source_observations"),
            "next_obs": pair("next_observations"),
            "teacher_inputs": pair("teacher_inputs"),
            "teacher_commands": pair("teacher_commands"),
            "commands": pair("commands"),
            "applied_actions": pair("actions"),
            "raw_policy_actions": raw_actions.copy(),
            "values": values.copy(),
            "log_probs": log_probs.copy(),
            "rewards": np.array([t["rewards"][p] for p in "AB"], np.float64),
            "terminated": np.array(t["terminated"], np.bool_),
            "truncated": np.array(t["truncated"], np.bool_),
            **{
                k: t[k].copy()
                for k in (
                    "engine_before",
                    "engine_after",
                    "metrics_1ms",
                    "contact_sha256_1ms",
                )
            },
        }
        checked_arrays({k: a[None] for k, a in row.items()}, 1)
        meta = {
            k: copy.deepcopy(t["info"][k])
            for k in ("episode_id", "native_seed", "tick")
        }
        meta["behavior"] = copy.deepcopy(t["info"]["behavior"])
        # Episode identity in the transition is captured BEFORE Vec auto-reset.
        for key in ("episode_serial", "logical_seed", "nonce"):
            meta[key] = t[key]
        checked_metadata(meta)
        self.rows.append(row)
        self.metadata.append(meta)
        if len(self.rows) == self.chunk_rows:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in FIELDS}
        checked_arrays(arrays, len(self.rows))
        number = len(self.chunks)
        stem = self.output / f"chunk-{number:04d}"
        forecast = plan(arrays)
        # The store temp is renamed, not copied. Charge its exact total plus
        # the complete metadata receipt BEFORE either file is created.
        envelope = {
            "schema": SCHEMA,
            "contract_hash": self.contract_hash,
            "number": number,
            "first_row": self.count,
            "count": len(self.rows),
            "previous_chain": self.chain,
            "rows": self.metadata,
            "tape": forecast,
        }
        envelope["sha256"] = envelope_sha(envelope)
        text_bound = len(encode(envelope)) + 4096
        self.budget.check(forecast["archive_bytes"] + text_bound, text_bound)
        tape = store(
            stem.with_suffix(".npz"), arrays, maximum_archive_bytes=64 * 1024**2
        )
        envelope["tape"] = tape
        envelope["sha256"] = envelope_sha(envelope)
        receipt = self.budget.write_json(stem.with_suffix(".json"), envelope)
        self.chain = envelope["sha256"]
        self.chunks.append(
            {
                "path": stem.with_suffix(".json").name,
                **receipt,
                "first_row": self.count,
                "count": len(self.rows),
                "chain": self.chain,
                "tape_bytes": tape["archive_bytes"],
            }
        )
        self.count += len(self.rows)
        self.rows, self.metadata = [], []

    def close(self):
        if self.closed:
            raise ValueError("Guard trace already closed")
        self.flush()
        self.closed = True
        return {
            "schema": SCHEMA,
            "count": self.count,
            "chain": self.chain,
            "chunks": copy.deepcopy(self.chunks),
        }


def read_guard_chunk(path, receipt, *, contract_hash, first_row, previous_chain):
    from .native_tape_storage import sha_file

    path = Path(path)
    if (
        path.stat().st_size > 256 * 1024
        or path.stat().st_size != receipt["bytes"]
        or sha_file(path) != receipt["sha256"]
    ):
        raise ValueError("Guard chunk external identity")
    envelope = json.loads(path.read_bytes())
    expected = {
        "schema",
        "contract_hash",
        "number",
        "first_row",
        "count",
        "previous_chain",
        "rows",
        "tape",
        "sha256",
    }
    if (
        set(envelope) != expected
        or envelope["schema"] != SCHEMA
        or envelope["contract_hash"] != contract_hash
        or envelope["first_row"] != first_row
        or envelope["previous_chain"] != previous_chain
        or envelope["sha256"] != envelope_sha(envelope)
        or envelope["sha256"] != receipt["chain"]
        or envelope["count"] != receipt["count"]
        or len(envelope["rows"]) != envelope["count"]
    ):
        raise ValueError("Guard chunk envelope/sequence")
    for m in envelope["rows"]:
        checked_metadata(m)
    arrays = load(path.with_suffix(".npz"), envelope["tape"])
    checked_arrays(arrays, envelope["count"])
    return arrays, envelope["rows"], envelope["sha256"]
