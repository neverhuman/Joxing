"""Saved synthetic receipts only: no native construction or training."""

import copy
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_native_learner_context as audit
import qualify_native_learner_context as entry


@pytest.mark.parametrize("fault", ["none", "crc", "truncated", "trailing", "oversized"])
def test_independent_gzip_decoder_rejects_corruption_and_bombs(fault):
    blob = gzip.compress(b"exact" * 100, mtime=0)
    limit = 500
    if fault == "crc":
        blob = blob[:-8] + bytes([blob[-8] ^ 1]) + blob[-7:]
    elif fault == "truncated":
        blob = blob[:-1]
    elif fault == "trailing":
        blob += b"extra"
    elif fault == "oversized":
        limit = 499
    if fault == "none":
        assert audit.gunzip(blob, limit) == b"exact" * 100
    else:
        with pytest.raises((ValueError, audit.zlib.error)):
            audit.gunzip(blob, limit)


def load_fixture():
    spec = importlib.util.spec_from_file_location(
        "original_dq_tests", ROOT / "tests/test_deferred_qualification.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.complete_fixture


def fixtures(tmp_path):
    original = json.loads(
        (
            ROOT / "experiments/v0.0.11/deferred-native-terminal-v3-protocol.json"
        ).read_text()
    )
    for slot in original["slots"]:
        folder = tmp_path / slot["id"]
        load_fixture()(folder)
        target = folder / "semantic-result.json"
        values = json.loads(target.read_text())
        values["specification"] = {"model": "fixed"}
        values["counts"] = {"reward": 0.01, "physics": 99}
        target.write_text(json.dumps(values))
        target = folder / "traces/learner.jsonl"
        rows = audit.rows(target)
        for row in rows:
            row["semantic"].update(state="s" * 64, rng="r" * 64)
            if row["semantic"]["phase"] == "learn_after":
                row["semantic"]["replay"] = "b" * 64
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return original | {"expected_specification": {"model": "fixed"}}


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "state",
        "rng",
        "reward",
        "force",
        "missing_update",
        "extra_update",
        "missing_stream",
        "empty_stream",
        "replay",
        "actor",
        "unchanged_actor",
        "denominator",
        "specification",
        "signed_zero",
    ],
)
def test_independent_full_trace_and_tensor_denominators(tmp_path, fault):
    packet = fixtures(tmp_path)
    file = tmp_path / "candidateA/traces/learner.jsonl"
    records = audit.rows(file)
    hasher = lambda p: "initial" if p.name.startswith("initial") else "final"
    if fault in ("state", "rng", "force", "signed_zero"):
        records[0]["semantic"][fault] = -0.0 if fault == "signed_zero" else "changed"
    elif fault == "missing_update":
        records.pop(2)
    elif fault == "extra_update":
        records.append(copy.deepcopy(records[2]))
    elif fault == "replay":
        next(r["semantic"] for r in records if r["semantic"]["phase"] == "learn_after")[
            "replay"
        ] = "bad"
    elif fault == "actor":
        hasher = lambda p: "wrong"
    elif fault in ("reward", "unchanged_actor", "denominator", "specification"):
        target = tmp_path / "candidateA/semantic-result.json"
        v = audit.read(target)
        if fault == "reward":
            v["counts"]["reward"] += 1e-15
        elif fault == "unchanged_actor":
            v["final_actor_hash"] = v["initial_actor_hash"]
        elif fault == "specification":
            v["specification"]["model"] = "changed"
        else:
            v["training_transitions"] = 63
        target.write_text(json.dumps(v))
    elif fault in ("missing_stream", "empty_stream"):
        target = tmp_path / "candidateA/traces/native-world-3.jsonl"
        target.unlink() if fault == "missing_stream" else target.write_text("")
    file.write_text("".join(json.dumps(r) + "\n" for r in records))
    if fault == "none":
        result = audit.check_trials(tmp_path, packet, hasher)
        assert (
            result["transitions"],
            result["updates"],
            len(result["comparisons"]),
        ) == (256, 48, 20)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            audit.check_trials(tmp_path, packet, hasher)


@pytest.mark.parametrize("fault", ["none", "fixture", "budget", "schema", "reader"])
def test_entry_never_relaxes_original_math_or_limits(fault):
    original = json.loads(
        (
            ROOT / "experiments/v0.0.11/deferred-native-terminal-v3-protocol.json"
        ).read_text()
    )
    packet = copy.deepcopy(original)
    packet.update(
        schema=entry.SCHEMA,
        execution_profile=entry.PROFILE,
        simulation_workers=4,
        threads_per_process=1,
        process_ceiling=16,
        owned_rss_bytes=8 * 1024**3,
        planned_stop_rss_bytes=7 * 1024**3,
        max_monitor_gap_s=0.1,
        close_deadline_s=2,
        whole_wall_s=300,
        per_slot_wall_s=60,
        physics_limit=240000,
        receipt_bytes=2 * 1024**2,
        fixture_bytes=1024**3,
        control_wall_s=30,
        rss_reader="terminal-mm-reserve-v3",
        watchdog_output_backend="private-tmpfs-v1",
    )
    if fault == "fixture":
        packet["trial_config"]["learner"]["gamma"] = 0.98
    elif fault == "budget":
        packet["owned_rss_bytes"] += 1
    elif fault == "schema":
        packet["schema"] = "unknown"
    elif fault == "reader":
        packet["rss_reader"] = "strict-v2"
    if fault == "none":
        entry.validate(packet, original)
    else:
        with pytest.raises(ValueError):
            entry.validate(packet, original)


@pytest.mark.parametrize("fault", ["none", "cap", "projection", "reserve", "scope"])
def test_receipt_exhaustion_blocks_before_any_native_work(tmp_path, monkeypatch, fault):
    projection = {
        "scope": "synthetic stored-evidence sizing; zero native work",
        "projected_peak_receipt_bytes": 500000,
    }
    packet = {
        "receipt_projection_file": "sizing.json",
        "receipt_reserve_bytes": 1024**2,
        "receipt_bytes": 2 * 1024**2,
    }
    used = {"receipt_bytes": 20000}
    if fault == "cap":
        used["receipt_bytes"] = 1024**2 + 1
    elif fault == "projection":
        projection["projected_peak_receipt_bytes"] = 1024**2 + 1
    elif fault == "reserve":
        packet["receipt_reserve_bytes"] = 1
    elif fault == "scope":
        projection["scope"] = "unregistered"
    (tmp_path / "sizing.json").write_text(json.dumps(projection))
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    if fault == "none":
        assert entry.receipt_headroom(packet, used)["used"] == 20000
    else:
        with pytest.raises(ValueError, match="before native"):
            entry.receipt_headroom(packet, used)


def test_saved_actor_tensor_reader_matches_original_and_detects_mutation(tmp_path):
    import torch

    from joxing.wave import canonical_actor_hash

    model = torch.nn.Linear(2, 2)
    path = tmp_path / "actor.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    first = audit.actor_hash(path)
    assert first == canonical_actor_hash(model)
    with torch.no_grad():
        model.weight[0, 0] += 0.000001
    torch.save({"state_dict": model.state_dict()}, path)
    assert audit.actor_hash(path) != first


@pytest.mark.parametrize("fault", ["none", "chain", "unknown"])
def test_independent_credit_reconstructs_charged_snapshot(tmp_path, fault):
    from joxing.physics_credit import PhysicsCredit

    ledger = PhysicsCredit.create(tmp_path / "credit.json", "binding", 100)
    token = ledger.reserve(20, "legacyA", "preparation")
    ledger.commit(token, 10)
    if fault == "unknown":
        ledger.unknown(ledger.reserve(30, "legacyA", "step"), "interrupted")
    elif fault == "chain":
        path = tmp_path / "credit.jsonl"
        path.write_text(path.read_text().replace('"actual":10', '"actual":11'))
    if fault == "chain":
        with pytest.raises(ValueError):
            audit.check_credit(tmp_path / "credit.json", 100, True)
    else:
        assert (
            audit.check_credit(tmp_path / "credit.json", 100, fault == "none")
            == ledger.snapshot()
        )
        if fault == "unknown":
            with pytest.raises(ValueError):
                audit.check_credit(tmp_path / "credit.json", 100, True)


@pytest.mark.parametrize("fault", ["none", "main", "terminal", "epoch"])
def test_independent_wire_reader_binds_entire_terminal_and_original_epoch(
    tmp_path, fault
):
    from receipt_encoding import (
        V2,
        compact_bytes,
        encode_bootstrap,
        main_metadata,
        save_report,
        terminal_line,
    )

    folder = tmp_path / "runs/v0.0.11"
    out = folder / "packet"
    out.mkdir(parents=True)
    report = {
        "source_commit": "source",
        "protocol_sha256": "protocol",
        "started_monotonic_s": 12.25,
        "status": "halted",
        "rows": [],
        "wall_s": 5.125,
    }
    save_report(out / "result.json.gz", report, mode=V2, legacy_save=None)
    values = {
        k: report.get(k)
        for k in (
            "status",
            "error",
            "rows",
            "physics_credit",
            "wall_s",
            "final_live_recorded",
        )
    }
    if fault == "terminal":
        values["wall_s"] += 1e-10
    terminal = terminal_line(values, V2, main=main_metadata(out / "result.json.gz"))
    (folder / "native-learner-context-launch.log").write_text(terminal)
    bootstrap = {
        "source_commit": "source",
        "protocol_sha256": "protocol",
        "epoch_started_monotonic_s": 12.26 if fault == "epoch" else 12.25,
    }
    blob, metadata = encode_bootstrap(bootstrap)
    (folder / "native-learner-context-deployment.json.gz").write_bytes(blob)
    (folder / "native-learner-context-deployment.encoding.json").write_bytes(
        compact_bytes(metadata)
    )
    if fault == "main":
        (out / "result.json.gz").write_bytes(
            gzip.compress(compact_bytes(report | {"wall_s": 99}))
        )
    if fault == "none":
        assert (
            audit.verify_wire(tmp_path, out, {"receipt_wire_encoding": V2}, report)
            == bootstrap
        )
    else:
        with pytest.raises(ValueError):
            audit.verify_wire(tmp_path, out, {"receipt_wire_encoding": V2}, report)
