import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from host_admission import assess_pair, parse_psi, reasons

POLICY = {
    "fresh_s": 2,
    "spacing_s": 10,
    "load_per_affinity_cpu": 0.5,
    "cpu_some_avg10": 10,
    "io_full_avg10": 1,
    "memory_full_avg10": 1,
    "minimum_memory_bytes": 16 * 1024**3,
    "minimum_disk_bytes": 10 * 1024**3,
}


def sample(t=10):
    return {
        "monotonic_s": t,
        "load": [1, 1, 1],
        "affinity_cpus": 8,
        "available_memory_bytes": 32 * 1024**3,
        "disk_free_bytes": 30 * 1024**3,
        "jobs": [],
        "psi": {
            k: {"some": {"avg10": 0}, "full": {"avg10": 0}}
            for k in ("cpu", "io", "memory")
        },
    }


@pytest.mark.parametrize(
    "case", ["valid", "duplicate", "nan", "negative", "large", "missing"]
)
def test_psi_parse_rejects_ambiguous_data(case):
    raw = "some avg10=0.5 avg60=1.0 avg300=2.0 total=123\nfull avg10=0 avg60=0 avg300=0 total=0\n"
    if case == "duplicate":
        raw += raw.splitlines()[0] + "\n"
    if case == "nan":
        raw = raw.replace("0.5", "nan")
    if case == "negative":
        raw = raw.replace("0.5", "-1")
    if case == "large":
        raw = raw.replace("0.5", "101")
    if case == "missing":
        raw = raw.replace(" avg60=1.0", "")
    if case == "valid":
        assert parse_psi(raw)["some"]["total"] == 123
    else:
        with pytest.raises(ValueError):
            parse_psi(raw)


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "load",
        "cpu",
        "io",
        "memory",
        "affinity",
        "ram",
        "disk",
        "jobs",
        "stale",
        "future",
        "missing",
        "nan",
    ],
)
def test_admission_reasons_prevent_silent_success(case):
    s = sample()
    if case == "load":
        s["load"][0] = 4.01
    if case in ("cpu", "io", "memory"):
        s["psi"][case]["some" if case == "cpu" else "full"]["avg10"] = (
            11 if case == "cpu" else 1.1
        )
    if case == "affinity":
        s["affinity_cpus"] = 0
    if case == "ram":
        s["available_memory_bytes"] = 0
    if case == "disk":
        s["disk_free_bytes"] = 0
    if case == "jobs":
        s["jobs"] = [{"pid": 42, "start_ticks": 99}]
    if case == "stale":
        s["monotonic_s"] = 5
    if case == "future":
        s["monotonic_s"] = 20
    if case == "missing":
        del s["psi"]["memory"]
    if case == "nan":
        s["load"][0] = float("nan")
    assert bool(reasons(s, POLICY, 10)) == (case != "valid")


@pytest.mark.parametrize(
    "case", ["valid", "short", "old", "first_busy", "missing", "count"]
)
def test_pair_spacing_freshness_and_deferral(case):
    snapshots = [sample(10), sample(20)]
    if case == "short":
        snapshots[1]["monotonic_s"] = 19
    if case == "old":
        snapshots[1]["monotonic_s"] = 15
    if case == "first_busy":
        snapshots[0]["load"][0] = 100
    if case == "missing":
        snapshots[1] = {}
    if case == "count":
        snapshots.pop()
    before = copy.deepcopy(snapshots)
    result = assess_pair(snapshots, POLICY, 20)
    assert result["status"] == ("admitted" if case == "valid" else "deferred")
    assert snapshots == before
