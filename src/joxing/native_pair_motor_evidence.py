"""Opt-in lossless chunks for the native paired motor pilot.

This is the ZERO-delay, four-frame/20ms profile only. It stores raw sensor
frames, never inverses of tanh observations. Native state/teacher/action values
stay exact. Neither the original smoke recorder nor the physical model changes.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .history import TimedHistory, TimedSample
from .native_pair_motor_rl import actor_vector, exact_f32, feature_scale
from .native_tape_storage import MAX_BYTES, load, plan, sha_file, store
from .observations import DIMENSION, SCHEMA_HASH
from .provenance import digest

SCHEMA = "joxing-native-pair-causal-chunks-v1"
MAX_ROW_TEXT = 3072
TEXT_SUFFIXES = {".json", ".jsonl", ".xml", ".log", ".txt"}
ROW_FIELDS = {
    "teacher_inputs": (np.dtype("float32"), (2, 3601)),
    "teacher_commands": (np.dtype("float32"), (2, 700)),
    "applied_actions": (np.dtype("float32"), (2, 138)),
    "raw_policy_actions": (np.dtype("float32"), (2, 138)),
    "values": (np.dtype("float32"), (2, 1)),
    "log_probs": (np.dtype("float32"), (2,)),
    "commands": (np.dtype("float32"), (2, 700)),
    "rewards": (np.dtype("float64"), (2,)),
    "terminated": (np.dtype("bool"), ()),
    "truncated": (np.dtype("bool"), ()),
}


def encode(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def training_storage_forecast(
    *,
    joint_steps,
    seeds,
    state_dimension,
    checkpoint_bytes,
    continuation_bytes,
    other_raw_bytes,
    other_text_bytes,
    raw_limit=1024**3,
    text_limit=16 * 1024**2,
):
    """Conservative reservation, NOT measured throughput or a run protocol.

    Charge every possible 20ms episode reset, one full checkpoint+continuation
    per 128-row rollout plus initial, and fixed metadata/fault allowances. The
    caller MUST bound checkpoint serialization and all evaluation/render/source
    writes prospectively; passing this calculation does not implement that guard.
    """
    quantities = (
        joint_steps,
        seeds,
        state_dimension,
        checkpoint_bytes,
        continuation_bytes,
        other_raw_bytes,
        other_text_bytes,
        raw_limit,
        text_limit,
    )
    if any(type(n) is not int or n <= 0 for n in quantities) or joint_steps % 128:
        raise ValueError("Explicit positive pilot forecast quantities/128 rows")
    if state_dimension > 16384:
        raise ValueError("Forecast native state bound")
    n = 128
    arrays = {
        k: np.zeros((n, *shape), dtype=dtype)
        for k, (dtype, shape) in ROW_FIELDS.items()
    }
    arrays.update(
        next_frames=np.zeros((n, 2, DIMENSION), np.float32),
        after_states=np.zeros((n, state_dimension), np.float64),
        initial_values=np.zeros((2, 4, DIMENSION), np.float32),
        initial_times=np.zeros((2, 4, 2), np.int64),
        initial_state=np.zeros(state_dimension, np.float64),
        reset_rows=np.arange(1, n, dtype=np.int64),
        reset_frames=np.zeros((n - 1, 2, DIMENSION), np.float32),
        reset_states=np.zeros((n - 1, state_dimension), np.float64),
    )
    archive = plan(arrays)["archive_bytes"]
    chunks = seeds * (joint_steps // n)
    manifests = chunks * (n * MAX_ROW_TEXT + 32768)
    checkpoint_count = seeds * (joint_steps // n + 1)
    checkpoints = checkpoint_count * (checkpoint_bytes + continuation_bytes)
    raw = chunks * archive + manifests + checkpoints + other_raw_bytes
    text = manifests + other_text_bytes
    return {
        "scope": "Worst-reset storage reservation only; no native training outcome",
        "joint_steps_per_seed": joint_steps,
        "seeds": seeds,
        "chunk_rows": n,
        "chunks": chunks,
        "maximum_reset_rows_per_chunk": n - 1,
        "chunk_archive_bytes": archive,
        "manifest_text_upper_bytes": manifests,
        "checkpoint_count": checkpoint_count,
        "checkpoint_reservation_bytes": checkpoints,
        "other_raw_bytes": other_raw_bytes,
        "other_text_bytes": other_text_bytes,
        "raw_upper_bytes": raw,
        "text_upper_bytes": text,
        "raw_limit": raw_limit,
        "text_limit": text_limit,
        "fits": raw <= raw_limit and text <= text_limit and archive <= MAX_BYTES,
        "checkpoint_and_other_writes_guarded": False,
    }


def exact_array(value, dtype, shape):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != shape
    ):
        raise ValueError("Exact evidence dtype/shape required")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite evidence")
    return value.copy()


class EvidenceBudget:
    """Single-owner pre-write RAW and TEXT ceilings, including temporary bytes.

    Other writers must use this guard too; it cannot police arbitrary external
    writes. Symlinks are rejected. Scanning this small owned directory is bounded
    by the formal owner's separate file/process limits.
    """

    def __init__(self, root, *, raw_limit, text_limit, fault_reserve=65536):
        self.root = Path(root).resolve()
        self.raw_limit, self.text_limit = int(raw_limit), int(text_limit)
        self.reserve = int(fault_reserve)
        if not 0 <= self.reserve < min(self.raw_limit, self.text_limit):
            raise ValueError("Evidence budget/reserve")
        self.root.mkdir(parents=True, exist_ok=True)

    def used(self):
        raw = text = 0
        for p in self.root.rglob("*"):
            if p.is_symlink():
                raise ValueError("Evidence root must not contain symlinks")
            if p.is_file():
                n = p.stat().st_size
                raw += n
                # A partial text replacement is text too.
                if any(s in TEXT_SUFFIXES for s in p.suffixes):
                    text += n
        return raw, text

    def check(self, raw, text=0):
        if type(raw) is not int or type(text) is not int or not 0 <= text <= raw:
            raise ValueError("Evidence requested credit")
        used_raw, used_text = self.used()
        if used_raw + raw + self.reserve > self.raw_limit:
            raise TimeoutError("Native evidence RAW first limit")
        if used_text + text + self.reserve > self.text_limit:
            raise TimeoutError("Native evidence TEXT first limit")

    def path(self, path):
        p = Path(path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("Foreign evidence path")
        if p.exists() or p.with_name(p.name + ".tmp").exists():
            raise FileExistsError("Native evidence refuses previous output")
        return p

    def write_json(self, path, value):
        p = self.path(path)
        blob = encode(value)
        self.check(len(blob), len(blob))
        p.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation; failed prefixes remain owned and charged.
        with p.open("xb") as f:
            f.write(blob)
        return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def histories(snapshots, game, tick):
    if set(snapshots) != set("AB") or type(tick) is not int or tick < 0 or tick % 20:
        raise ValueError("Native evidence paired history/clock")
    result = {}
    expected = list(range(max(0, tick - 60), tick + 1, 20))
    for p in "AB":
        s = snapshots[p]
        if s["capacity"] != 4 or s["dimension"] != DIMENSION:
            raise ValueError("Native evidence four raw frames required")
        if [r["sampled_tick"] for r in s["samples"]] != expected:
            raise ValueError("Missing/reordered/future native causal frame")
        for r in s["samples"]:
            if r["received_tick"] != r["sampled_tick"]:
                raise ValueError("This evidence profile has zero reaction delay")
            exact_f32(r["values"], (DIMENSION,))
        result[p] = TimedHistory.restore(
            s, game_id=game, player=p, schema_hash=SCHEMA_HASH
        )
    return result


def history_arrays(h, tick):
    views = [h[p].at(tick) for p in "AB"]
    values = np.stack([v.values for v in views])
    times = np.full((2, 4, 2), -1, np.int64)
    for i, v in enumerate(views):
        times[i, :, 0] = v.sampled_ticks
        times[i, v.valid, 1] = tick - v.receipt_age_ticks[v.valid]
    return values, times


def unpack_history(values, times, game, tick):
    exact_array(values, np.dtype("float32"), (2, 4, DIMENSION))
    exact_array(times, np.dtype("int64"), (2, 4, 2))
    snapshots = {}
    for i, p in enumerate("AB"):
        h = TimedHistory(game, p, SCHEMA_HASH, DIMENSION, 4)
        for j, (sampled, received) in enumerate(times[i]):
            if sampled == -1 and received == -1:
                if values[i, j].tobytes() != np.zeros(DIMENSION, np.float32).tobytes():
                    raise ValueError("Noncanonical history padding")
                continue
            h.receive(
                TimedSample(
                    game, p, SCHEMA_HASH, int(sampled), int(received), values[i, j]
                ),
                now_tick=int(received),
            )
        snapshots[p] = h.snapshot()
    return histories(snapshots, game, tick)


def vectors(h, tick):
    return np.stack([actor_vector(h[p], tick, feature_scale()) for p in "AB"])


class CausalCursor:
    """Continuation across rows AND chunks, including terminal reset ownership."""

    def __init__(self):
        self.count = 0
        self.game = self.serial = self.tick = self.state = self.h = None
        self.done = False

    def begin(self, row, h, state):
        game, serial, tick = row["episode_id"], row["episode_serial"], row["start_tick"]
        if (
            not isinstance(game, str)
            or not game
            or type(serial) is not int
            or serial < 1
        ):
            raise ValueError("Native evidence episode identity")
        reset = self.game is not None and game != self.game
        if self.game is not None:
            if reset:
                if not self.done or tick != 0 or serial != self.serial + 1:
                    raise ValueError("Foreign reset without completed joint episode")
            elif self.done or serial != self.serial or tick != self.tick:
                raise ValueError("Native evidence skipped/reordered episode/clock")
            elif state.tobytes() != self.state.tobytes() or any(
                digest(h[p].snapshot()) != digest(self.h[p].snapshot()) for p in "AB"
            ):
                raise ValueError("Native causal/state continuation mismatch")
        if row["joint_step"] != self.count or row["end_tick"] != tick + 20:
            raise ValueError("Native evidence row/control ordering")
        return reset

    def end(self, row, h, state, done):
        self.game, self.serial, self.tick = (
            row["episode_id"],
            row["episode_serial"],
            row["end_tick"],
        )
        self.state, self.h, self.done = state.copy(), copy.deepcopy(h), bool(done)
        self.count += 1


class CausalChunkWriter:
    def __init__(
        self, output, *, contract_hash, state_dimension, budget, chunk_rows=128
    ):
        self.output, self.budget = Path(output), budget
        if self.output.exists() or not self.output.resolve().is_relative_to(
            budget.root
        ):
            raise ValueError("Fresh owned native chunk directory required")
        if not isinstance(contract_hash, str) or len(contract_hash) != 64:
            raise ValueError("Native evidence contract identity")
        if type(chunk_rows) is not int or not 1 <= chunk_rows <= 128:
            raise ValueError("Native evidence chunk row bound")
        if type(state_dimension) is not int or not 1 <= state_dimension <= 16384:
            raise ValueError("Native evidence state dimension bound")
        self.contract_hash, self.state_dimension, self.chunk_rows = (
            contract_hash,
            state_dimension,
            chunk_rows,
        )
        self.cursor, self.chain, self.chunks = CausalCursor(), "0" * 64, []
        self.buffer, self.metadata = [], []
        self.reset_rows, self.reset_frames, self.reset_states = [], [], []
        self.initial = None
        self.closed = False
        self.output.mkdir(parents=True)

    def append(self, t, *, raw_actions, values, log_probs):
        if self.closed:
            raise RuntimeError("Closed native evidence writer")
        info = copy.deepcopy(t["info"])
        end = info["tick"]
        row = {
            "joint_step": self.cursor.count,
            "episode_id": info["episode_id"],
            "episode_serial": info["episode_serial"],
            "start_tick": end - 20,
            "end_tick": end,
            "info": info,
        }
        source = histories(t["source_histories"], row["episode_id"], end - 20)
        target = histories(t["next_histories"], row["episode_id"], end)
        before = exact_array(
            t["engine_before"], np.dtype("float64"), (self.state_dimension,)
        )
        after = exact_array(
            t["engine_after"], np.dtype("float64"), (self.state_dimension,)
        )
        reset = self.cursor.begin(row, source, before)
        src, dst = vectors(source, end - 20), vectors(target, end)
        for i, p in enumerate("AB"):
            for obs, key in ((src, "source_observations"), (dst, "next_observations")):
                original = exact_array(t[key][p], np.dtype("float32"), (7901,))
                if original.tobytes() != obs[i].tobytes():
                    raise ValueError(
                        "Raw causal reconstruction differs from actual actor observation"
                    )
            expected = copy.deepcopy(source[p])
            newest = target[p].snapshot()["samples"][-1]
            expected.receive(
                TimedSample(
                    row["episode_id"],
                    p,
                    SCHEMA_HASH,
                    end,
                    end,
                    np.array(newest["values"], np.float32),
                ),
                now_tick=end,
            )
            if digest(expected.snapshot()) != digest(target[p].snapshot()):
                raise ValueError("Native target history is not the next causal frame")
        arrays = {
            k: np.stack([t[k][p] for p in "AB"])
            for k in ("teacher_inputs", "teacher_commands", "commands")
        }
        arrays.update(
            applied_actions=np.stack([t["actions"][p] for p in "AB"]),
            raw_policy_actions=raw_actions,
            values=values,
            log_probs=log_probs,
            rewards=np.array([t["rewards"][p] for p in "AB"], np.float64),
            terminated=np.array(t["terminated"]),
            truncated=np.array(t["truncated"]),
        )
        arrays = {k: exact_array(v, *ROW_FIELDS[k]) for k, v in arrays.items()}
        if arrays["terminated"].item() and arrays["truncated"].item():
            raise ValueError("Native motor termination/truncation must be exclusive")
        if (
            np.clip(arrays["raw_policy_actions"], -1, 1).tobytes()
            != arrays["applied_actions"].tobytes()
        ):
            raise ValueError("Native raw/applied action clipping mismatch")
        arrays.update(
            next_frames=np.stack([target[p].at(end).values[-1] for p in "AB"]),
            after_states=after,
        )
        row.update(
            source_obs_sha256=array_sha(src),
            next_obs_sha256=array_sha(dst),
            engine_before_sha256=array_sha(before),
            engine_after_sha256=array_sha(after),
            array_hashes={k: array_sha(a) for k, a in arrays.items()},
        )
        chain = digest({"previous": self.chain, "row": row})
        row["chain"] = chain
        if len(encode(row)) > MAX_ROW_TEXT:
            raise ValueError("Native evidence per-row text bound")
        # Bound ALL pending arrays and metadata before accepting another row. The
        # directory's actual ZIP+manifest lengths receive an exact second check.
        pending = sum(a.nbytes for r in self.buffer for a in r.values())
        pending += sum(a.nbytes for a in arrays.values())
        pending += sum(a.nbytes for a in self.reset_frames + self.reset_states)
        if reset and self.buffer:
            pending += before.nbytes + 2 * DIMENSION * 4 + 8
        initial_size = 2 * 4 * DIMENSION * 4 + 2 * 4 * 2 * 8 + before.nbytes
        text = len(encode(self.metadata + [row])) + 32768
        self.budget.check(pending + initial_size + text + 32768, text)
        if not self.buffer:
            self.initial = (*history_arrays(source, end - 20), before.copy())
        elif reset:
            self.reset_rows.append(len(self.buffer))
            self.reset_frames.append(
                np.stack([source[p].at(0).values[-1] for p in "AB"])
            )
            self.reset_states.append(before)
        self.buffer.append(arrays)
        self.metadata.append(row)
        self.cursor.end(row, target, after, bool(t["terminated"] or t["truncated"]))
        self.chain = chain
        if len(self.buffer) == self.chunk_rows:
            self.flush()

    def flush(self):
        if not self.buffer:
            return None
        arrays = {k: np.stack([r[k] for r in self.buffer]) for k in self.buffer[0]}
        arrays.update(
            initial_values=self.initial[0],
            initial_times=self.initial[1],
            initial_state=self.initial[2],
            reset_rows=np.array(self.reset_rows, np.int64),
            reset_frames=np.array(self.reset_frames, np.float32).reshape(
                -1, 2, DIMENSION
            ),
            reset_states=np.array(self.reset_states, np.float64).reshape(
                -1, self.state_dimension
            ),
        )
        proposal = plan(arrays)
        if proposal["archive_bytes"] > MAX_BYTES:
            raise ValueError("Native evidence chunk exceeds qualified ZIP bound")
        stem = f"chunk-{len(self.chunks):04d}"
        tape = self.budget.path(self.output / (stem + ".npz"))
        path = self.budget.path(self.output / (stem + ".json"))
        value = {
            "schema": SCHEMA,
            "contract_hash": self.contract_hash,
            "first_joint_step": self.metadata[0]["joint_step"],
            "rows": self.metadata,
            "previous_chain": self.chunks[-1]["chain"] if self.chunks else "0" * 64,
            "chain": self.chain,
            "tape": {**proposal, "sha256": "0" * 64},
            "state_dimension": self.state_dimension,
        }
        meta_size = len(encode(value))
        self.budget.check(proposal["archive_bytes"] + meta_size, meta_size)
        value["tape"] = store(tape, arrays)
        result = self.budget.write_json(path, value)
        receipt = {
            "path": path.name,
            **result,
            "rows": len(self.metadata),
            "first_joint_step": value["first_joint_step"],
            "chain": self.chain,
            "tape_bytes": value["tape"]["archive_bytes"],
        }
        self.chunks.append(receipt)
        self.buffer, self.metadata = [], []
        self.reset_rows, self.reset_frames, self.reset_states = [], [], []
        self.initial = None
        return receipt

    def close(self):
        if not self.closed:
            self.flush()
            self.closed = True
        return {
            "schema": SCHEMA,
            "contract_hash": self.contract_hash,
            "joint_transitions": self.cursor.count,
            "chain": self.chain,
            "chunks": copy.deepcopy(self.chunks),
        }


def read_chunk(
    path, *, expected_sha256, contract_hash, cursor=None, previous_chain="0" * 64
):
    """Return one bounded chunk of reconstructed FULL rows, plus causal cursor.

    The caller pins manifest hashes from its independently authenticated result;
    a self-signed manifest alone is not provenance or a physical replay audit.
    """
    path = Path(path)
    if path.stat().st_size > 2 * 1024**2 or sha_file(path) != expected_sha256:
        raise ValueError("Native evidence external manifest identity/bound")
    meta = json.loads(path.read_bytes())
    if meta["schema"] != SCHEMA or meta["contract_hash"] != contract_hash:
        raise ValueError("Foreign native evidence contract")
    rows = meta["rows"]
    if not 1 <= len(rows) <= 128 or meta["previous_chain"] != previous_chain:
        raise ValueError("Native evidence chunk ordering/bound")
    cursor = cursor or CausalCursor()
    if meta["first_joint_step"] != cursor.count:
        raise ValueError("Native evidence missing/reordered chunk")
    size = meta["state_dimension"]
    if type(size) is not int or not 1 <= size <= 16384:
        raise ValueError("Native evidence engine shape")
    arrays = load(path.with_suffix(".npz"), meta["tape"])
    specs = {
        k: (dtype, (len(rows), *shape)) for k, (dtype, shape) in ROW_FIELDS.items()
    }
    specs.update(
        next_frames=(np.dtype("float32"), (len(rows), 2, DIMENSION)),
        after_states=(np.dtype("float64"), (len(rows), size)),
    )
    resets = arrays["reset_rows"]
    exact_array(resets, np.dtype("int64"), (len(resets),))
    if resets.tolist() != sorted(set(resets.tolist())) or any(
        not 0 < r < len(rows) for r in resets
    ):
        raise ValueError("Native evidence reset row identity")
    for k, (dtype, shape) in specs.items():
        exact_array(arrays[k], dtype, shape)
    exact_array(arrays["initial_state"], np.dtype("float64"), (size,))
    exact_array(
        arrays["reset_frames"], np.dtype("float32"), (len(resets), 2, DIMENSION)
    )
    exact_array(arrays["reset_states"], np.dtype("float64"), (len(resets), size))
    expected_keys = set(specs) | {
        "reset_rows",
        "reset_frames",
        "reset_states",
        "initial_state",
        "initial_values",
        "initial_times",
    }
    if set(arrays) != expected_keys:
        raise ValueError("Native evidence array schema")
    h = unpack_history(
        arrays["initial_values"],
        arrays["initial_times"],
        rows[0]["episode_id"],
        rows[0]["start_tick"],
    )
    state = arrays["initial_state"]
    result, chain = [], previous_chain
    for i, row in enumerate(rows):
        if len(encode(row)) > MAX_ROW_TEXT or any(
            row["info"][k] != row[v]
            for k, v in (
                ("tick", "end_tick"),
                ("episode_id", "episode_id"),
                ("episode_serial", "episode_serial"),
            )
        ):
            raise ValueError("Native evidence scalar metadata identity/bound")
        if i in resets:
            j = resets.tolist().index(i)
            vals = np.zeros((2, 4, DIMENSION), np.float32)
            vals[:, -1] = arrays["reset_frames"][j]
            times = np.full((2, 4, 2), -1, np.int64)
            times[:, -1] = 0
            h = unpack_history(vals, times, row["episode_id"], row["start_tick"])
            state = arrays["reset_states"][j]
        reset = cursor.begin(row, h, state)
        if i and reset != (i in resets):
            raise ValueError("Native evidence missing/spurious reset marker")
        src = vectors(h, row["start_tick"])
        before = state.copy()
        a = {k: arrays[k][i] for k in specs}
        for k, v in a.items():
            if array_sha(v) != row["array_hashes"][k]:
                raise ValueError("Native evidence row array hash:" + k)
        if a["terminated"].item() and a["truncated"].item():
            raise ValueError("Native evidence terminal mask")
        if (
            np.clip(a["raw_policy_actions"], -1, 1).tobytes()
            != a["applied_actions"].tobytes()
        ):
            raise ValueError("Native evidence raw/applied actions")
        for j, p in enumerate("AB"):
            end = row["end_tick"]
            h[p].receive(
                TimedSample(
                    row["episode_id"], p, SCHEMA_HASH, end, end, a["next_frames"][j]
                ),
                now_tick=end,
            )
        dst = vectors(h, row["end_tick"])
        state = a["after_states"]
        if any(
            array_sha(v) != row[k]
            for v, k in (
                (src, "source_obs_sha256"),
                (dst, "next_obs_sha256"),
                (before, "engine_before_sha256"),
                (state, "engine_after_sha256"),
            )
        ):
            raise ValueError("Native evidence reconstructed actor/state hash")
        unsigned = {k: v for k, v in row.items() if k != "chain"}
        chain = digest({"previous": chain, "row": unsigned})
        if chain != row["chain"]:
            raise ValueError("Native evidence transition chain")
        cursor.end(row, h, state, a["terminated"].item() or a["truncated"].item())
        result.append(
            {
                **a,
                "source_obs": src,
                "next_obs": dst,
                "engine_before": before,
                "engine_after": state.copy(),
                "metadata": row,
            }
        )
    if chain != meta["chain"]:
        raise ValueError("Native evidence final chunk chain")
    return result, cursor, chain
