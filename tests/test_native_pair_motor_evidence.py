import copy
import json

import numpy as np
import pytest

from joxing.history import TimedHistory, TimedSample
from joxing.native_pair_motor_evidence import (
    CausalChunkWriter,
    EvidenceBudget,
    encode,
    read_chunk,
)
from joxing.native_pair_motor_rl import actor_vector, feature_scale
from joxing.native_tape_storage import load, sha_file, store
from joxing.observations import DIMENSION, SCHEMA_HASH

CONTRACT = "f" * 64


def transitions(count=9, horizon=60):
    rng = np.random.default_rng(123)
    h = None
    state = np.zeros(17, np.float64)
    game = None
    for index in range(count):
        tick = (index % (horizon // 20)) * 20
        serial = index // (horizon // 20) + 1
        if tick == 0:
            game = "world-" + str(serial)
            h = {p: TimedHistory(game, p, SCHEMA_HASH, DIMENSION, 4) for p in "AB"}
            state = rng.normal(size=17).astype(np.float64)
            for p in "AB":
                v = rng.normal(size=DIMENSION).astype(np.float32)
                v[0] = np.float32(-0.0)
                h[p].receive(TimedSample(game, p, SCHEMA_HASH, 0, 0, v), now_tick=0)
        before = {p: h[p].snapshot() for p in "AB"}
        obs = {p: actor_vector(h[p], tick, feature_scale()) for p in "AB"}
        old = state.copy()
        state = rng.normal(size=17).astype(np.float64)
        state[0] = -0.0
        for p in "AB":
            v = rng.normal(size=DIMENSION).astype(np.float32)
            v[0] = np.float32(-0.0)
            h[p].receive(
                TimedSample(game, p, SCHEMA_HASH, tick + 20, tick + 20, v),
                now_tick=tick + 20,
            )
        t = {
            "info": {"tick": tick + 20, "episode_id": game, "episode_serial": serial},
            "source_histories": before,
            "next_histories": {p: h[p].snapshot() for p in "AB"},
            "source_observations": obs,
            "next_observations": {
                p: actor_vector(h[p], tick + 20, feature_scale()) for p in "AB"
            },
            "engine_before": old,
            "engine_after": state.copy(),
            "teacher_inputs": {
                p: rng.normal(size=3601).astype(np.float32) for p in "AB"
            },
            "teacher_commands": {
                p: rng.normal(size=700).astype(np.float32) for p in "AB"
            },
            "commands": {p: rng.normal(size=700).astype(np.float32) for p in "AB"},
            "actions": {p: np.zeros(138, np.float32) for p in "AB"},
            "rewards": {"A": 0.1, "B": 0.2},
            "terminated": False,
            "truncated": tick + 20 == horizon,
        }
        yield t


def writer(tmp_path, *, chunk_rows=2, raw=64 * 1024**2, text=2 * 1024**2):
    budget = EvidenceBudget(tmp_path, raw_limit=raw, text_limit=text)
    return CausalChunkWriter(
        tmp_path / "chunks",
        contract_hash=CONTRACT,
        state_dimension=17,
        budget=budget,
        chunk_rows=chunk_rows,
    )


def append(w, t):
    w.append(
        t,
        raw_actions=np.zeros((2, 138), np.float32),
        values=np.zeros((2, 1), np.float32),
        log_probs=np.zeros(2, np.float32),
    )


def read_all(w, result):
    rows, cursor, chain = [], None, "0" * 64
    for receipt in result["chunks"]:
        r, cursor, chain = read_chunk(
            w.output / receipt["path"],
            expected_sha256=receipt["sha256"],
            contract_hash=CONTRACT,
            cursor=cursor,
            previous_chain=chain,
        )
        rows.extend(r)
    assert chain == result["chain"] and len(rows) == result["joint_transitions"]
    return rows


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 4, 7, 128])
def test_lossless_history_native_endpoint_reset_and_chunk_continuity(
    tmp_path, chunk_rows
):
    w = writer(tmp_path, chunk_rows=chunk_rows)
    original = list(transitions())
    for t in original:
        append(w, t)
    result = w.close()
    assert w.close() == result
    decoded = read_all(w, result)
    for t, r in zip(original, decoded, strict=True):
        for name, field in (
            ("source_obs", "source_observations"),
            ("next_obs", "next_observations"),
            ("teacher_inputs", "teacher_inputs"),
            ("commands", "commands"),
        ):
            assert r[name].tobytes() == np.stack([t[field][p] for p in "AB"]).tobytes()
        assert r["engine_before"].tobytes() == t["engine_before"].tobytes()
        assert r["engine_after"].tobytes() == t["engine_after"].tobytes()
        assert np.signbit(r["engine_after"][0])
        assert r["truncated"].item() == t["truncated"]
    with pytest.raises(RuntimeError):
        append(w, original[-1])
    assert max(r["tape_bytes"] for r in result["chunks"]) < 64 * 1024**2


@pytest.mark.parametrize(
    "fault",
    [
        "clock",
        "corner",
        "schema",
        "history",
        "future",
        "delay",
        "rounded",
        "dtype",
        "obs",
        "state",
        "mask",
        "raw_clipping",
    ],
)
def test_bad_row_rejected_without_accepting_it(tmp_path, fault):
    rows = list(transitions(2))
    w = writer(tmp_path, chunk_rows=128)
    append(w, rows[0])
    t = copy.deepcopy(rows[1])
    if fault == "clock":
        t["info"]["tick"] += 20
    elif fault == "corner":
        t["source_histories"]["B"]["player"] = "A"
    elif fault == "schema":
        t["source_histories"]["A"]["schema_hash"] = "x"
    elif fault == "history":
        t["source_histories"]["A"]["samples"].pop(0)
    elif fault == "future":
        t["source_histories"]["A"]["samples"][-1]["sampled_tick"] = 40
    elif fault == "delay":
        t["source_histories"]["A"]["samples"][-1]["received_tick"] += 1
    elif fault == "rounded":
        t["source_histories"]["A"]["samples"][-1]["values"][0] = 0.123456789123
    elif fault == "dtype":
        t["engine_before"] = t["engine_before"].astype(np.float32)
    elif fault == "obs":
        t["source_observations"]["A"][0] += 0.01
    elif fault == "state":
        t["engine_before"][0] += 0.01
    elif fault == "mask":
        t.update(terminated=True, truncated=True)
    elif fault == "raw_clipping":
        t["actions"]["B"][0] = 0.1
    with pytest.raises(ValueError):
        append(w, t)
    assert w.cursor.count == 1 and len(w.buffer) == 1
    append(w, rows[1])
    assert len(read_all(w, w.close())) == 2


def test_cannot_cross_an_unfinished_episode_or_skip_chunk(tmp_path):
    w = writer(tmp_path)
    t = next(transitions(1))
    append(w, t)
    foreign = next(transitions(1))
    foreign["info"].update(episode_id="foreign", episode_serial=2)
    for k in ("source_histories", "next_histories"):
        for p in "AB":
            foreign[k][p]["game_id"] = "foreign"
    with pytest.raises(ValueError, match="reset"):
        append(w, foreign)
    result = w.close()
    r = result["chunks"][0]
    with pytest.raises(ValueError, match="contract"):
        read_chunk(
            w.output / r["path"], expected_sha256=r["sha256"], contract_hash="0" * 64
        )


@pytest.mark.parametrize("kind", ["raw", "text"])
def test_preallocated_raw_and_text_first_limits_before_accepting_row(tmp_path, kind):
    w = writer(
        tmp_path,
        raw=200000 if kind == "raw" else 2000000,
        text=100000 if kind == "text" else 2000000,
    )
    if kind == "text":
        w.budget.write_json(tmp_path / "earlier.json", {"already": "x" * 1000})
    with pytest.raises(TimeoutError, match=kind.upper()):
        append(w, next(transitions(1)))
    assert w.cursor.count == 0 and not w.buffer
    assert not list(w.output.iterdir())


def test_file_ownership_existing_outputs_and_symlink_refused(tmp_path):
    b = EvidenceBudget(tmp_path, raw_limit=1000000, text_limit=1000000)
    b.write_json(tmp_path / "a.json", {"small": 1})
    with pytest.raises(FileExistsError):
        b.write_json(tmp_path / "a.json", {})
    with pytest.raises(ValueError, match="Foreign"):
        b.write_json(tmp_path.parent / "foreign.json", {})
    (tmp_path / "link").symlink_to(tmp_path / "a.json")
    with pytest.raises(ValueError, match="symlinks"):
        b.check(1)


@pytest.mark.parametrize(
    "fault",
    ["manifest", "truncate", "crc", "shape", "row_order", "reset", "signed_zero"],
)
def test_corrupt_or_resigned_evidence_rejected(tmp_path, fault):
    w = writer(tmp_path, chunk_rows=7)
    for t in transitions(6):
        append(w, t)
    r = w.close()["chunks"][0]
    path = w.output / r["path"]
    m = json.loads(path.read_text())
    expected = r["sha256"]
    if fault == "manifest":
        path.write_bytes(path.read_bytes() + b" ")
    elif fault in ("truncate", "crc"):
        tape = path.with_suffix(".npz")
        raw = bytearray(tape.read_bytes())
        if fault == "truncate":
            raw = raw[:-7]
        else:
            raw[200] ^= 1
        tape.write_bytes(raw)
    elif fault in ("shape", "reset", "signed_zero"):
        tape = path.with_suffix(".npz")
        a = load(tape, m["tape"])
        tape.unlink()
        if fault == "shape":
            a["next_frames"] = a["next_frames"].astype(np.float64)
        elif fault == "reset":
            a["reset_rows"][0] = 1
        else:
            a["after_states"][0, 0] = 0.0
        m["tape"] = store(tape, a)
        path.write_bytes(encode(m))
        expected = sha_file(path)
    else:
        m["rows"][0]["joint_step"] = 2
        path.write_bytes(encode(m))
        expected = sha_file(path)
    with pytest.raises((ValueError, KeyError)):
        read_chunk(path, expected_sha256=expected, contract_hash=CONTRACT)


def test_maximum_chunk_and_partial_close_are_bounded_and_lossless(tmp_path):
    w = writer(tmp_path, chunk_rows=128)
    for t in transitions(129, horizon=60):
        append(w, t)
    result = w.close()
    assert [r["rows"] for r in result["chunks"]] == [128, 1]
    assert len(read_all(w, result)) == 129


def test_oversized_scalar_receipt_is_rejected_before_row_acceptance(tmp_path):
    w = writer(tmp_path)
    t = next(transitions(1))
    t["info"]["unbounded_diagnostic"] = "x" * 3072
    with pytest.raises(ValueError, match="per-row text"):
        append(w, t)
    assert w.cursor.count == 0 and not w.buffer


def test_worst_reset_forecast_declines_oversized_scope_without_raising_caps():
    from joxing.native_pair_motor_evidence import training_storage_forecast

    kw = {
        "seeds": 2,
        "state_dimension": 4555,
        "checkpoint_bytes": 13 * 1024**2,
        "continuation_bytes": 1024**2,
        "other_raw_bytes": 64 * 1024**2,
        "other_text_bytes": 2 * 1024**2,
    }
    large = training_storage_forecast(joint_steps=2048, **kw)
    small = training_storage_forecast(joint_steps=1024, **kw)
    assert not large["fits"] and large["raw_upper_bytes"] > large["raw_limit"]
    assert small["fits"] and small["raw_upper_bytes"] < small["raw_limit"]
    assert large["raw_limit"] == small["raw_limit"] == 1024**3
    assert not small["checkpoint_and_other_writes_guarded"]
