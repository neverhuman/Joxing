import multiprocessing

import pytest

from joxing import physics_credit as pc


def spend(path):
    ledger = pc.PhysicsCredit(path, "fixed")
    for _ in range(10):
        try:
            token = ledger.reserve(3, "A", "step")
        except RuntimeError:
            break
        ledger.commit(token, 3)


def test_atomic_processes_cannot_overspend(tmp_path):
    ledger = pc.PhysicsCredit.create(tmp_path / "credit.json", "fixed", 60)
    context = multiprocessing.get_context("spawn")
    jobs = [context.Process(target=spend, args=(str(ledger.path),)) for _ in range(4)]
    try:
        for job in jobs:
            job.start()
        for job in jobs:
            job.join(timeout=15)
            assert job.exitcode == 0
        result = ledger.snapshot()
        assert result["used"] == result["charged"] == 60 and not result["pending"]
    finally:
        for job in jobs:
            if job.is_alive():
                job.kill()
                job.join(timeout=2)


def test_unknown_reservation_stays_charged(tmp_path):
    ledger = pc.PhysicsCredit.create(tmp_path / "credit.json", "fixed", 20)
    token = ledger.reserve(15, "A", "preparation")
    ledger.unknown(token, "interrupted")
    assert ledger.snapshot()["charged"] == 15 and ledger.snapshot()["used"] == 0
    with pytest.raises(RuntimeError):
        ledger.reserve(6, "B", "step")
    with pytest.raises(ValueError):
        ledger.commit(token, 0)


def test_exact_refund_and_duplicate_cannot_invent_credit(tmp_path):
    ledger = pc.PhysicsCredit.create(tmp_path / "credit.json", "fixed", 20)
    token = ledger.reserve(10, "A", "step")
    with pytest.raises(ValueError):
        ledger.commit(token, 11)
    ledger.commit(token, 6)
    assert ledger.snapshot()["charged"] == 6
    with pytest.raises(ValueError):
        ledger.commit(token, 0)
    second = ledger.reserve(14, "B", "step")
    ledger.commit(second, 14)
    assert ledger.snapshot()["charged"] == 20


def test_foreign_stale_or_wrong_binding_denied(tmp_path, monkeypatch):
    ledger = pc.PhysicsCredit.create(tmp_path / "credit.json", "fixed", 20)
    with pytest.raises(ValueError):
        pc.PhysicsCredit(ledger.path, "wrong").snapshot()
    monkeypatch.setattr(
        pc, "process_identity", lambda: ledger.owner | {"start": "reused"}
    )
    with pytest.raises(ValueError):
        ledger.reserve(1, "A", "step")


def test_existing_ledger_never_overwritten(tmp_path):
    pc.PhysicsCredit.create(tmp_path / "credit.json", "fixed", 20)
    with pytest.raises(FileExistsError):
        pc.PhysicsCredit.create(tmp_path / "credit.json", "fixed", 20)
