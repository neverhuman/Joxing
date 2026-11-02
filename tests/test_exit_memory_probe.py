import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def probe():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "probe_exit_memory", scripts / "probe_exit_memory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "case", ["exiting", "live_missing", "nonzero_mm", "other_thread_live"]
)
def test_exit_hypothesis_requires_all_evidence(probe, case):
    row = {
        "vmrss_bytes": None,
        "vmsize_present": False,
        "statm_pages": [0] * 7,
        "threads": [{"flags": 4}],
        "after": {"flags": 4},
    }
    if case == "live_missing":
        row["after"]["flags"] = 0
    elif case == "nonzero_mm":
        row["statm_pages"][1] = 1
    elif case == "other_thread_live":
        row["threads"].append({"flags": 0})
    assert probe.exiting_empty_mm(row) == (case == "exiting")


@pytest.mark.parametrize(
    "mode,size", [("normal_exit", 64 * 1024**2), ("direct_exit", 512 * 1024**2)]
)
@pytest.mark.skipif(
    sys.platform != "linux", reason="Actual Linux exit-memory diagnostic"
)
def test_actual_owned_exit_memory(probe, tmp_path, mode, size):
    report = probe.run_case(tmp_path / "probe", mode, size)
    assert (
        report["status"] == "completed"
        and report["returncode"] == 0
        and report["process_reaped"]
    )
    assert report["native_physics_steps"] == report["learning_updates"] == 0
    # Not observing the hypothesis is a retained negative/limited diagnostic result.
    assert report["hypothesis"] in (
        "supported-synthetic-exit",
        "not_observed",
        "unclassified_missing_rss",
    )
