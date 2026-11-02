import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from runtime_import_check import run


@pytest.mark.parametrize(
    "case", ["valid", "import_fault", "wrong_version", "early_fault"]
)
def test_runtime_phases_cannot_infer_missing_success(case):
    expected = {
        "python": sys.version.split()[0],
        "numpy": "n",
        "mujoco": "m",
        "torch": "t",
        "sb3": "s",
    }
    versions = {"numpy": "n", "mujoco": "m", "torch": "t", "stable_baselines3": "s"}
    calls, events = [], []

    def importer(name):
        calls.append(name)
        if (case == "import_fault" and name == "torch") or (
            case == "early_fault" and name == "numpy"
        ):
            raise ImportError("Declared import fault")
        return SimpleNamespace(
            __version__="wrong"
            if case == "wrong_version" and name == "torch"
            else versions[name]
        )

    def observer():
        return {
            "monotonic_s": len(events),
            "process_cpu_s": 0,
            "pid": 17,
            "start_ticks": 99,
        }

    if case == "valid":
        assert (
            run(expected, importer=importer, observer=observer, emit=events.append)
            == expected
        )
    else:
        with pytest.raises((ImportError, ValueError)):
            run(expected, importer=importer, observer=observer, emit=events.append)
    assert events[0]["phase"] == "start"
    if case == "valid":
        assert events[-1]["phase"] == "qualified" and calls == list(versions)
    else:
        assert not any(e["phase"] == "qualified" for e in events)
    if case in ("early_fault", "import_fault"):
        assert events[-1]["phase"] == "import_fault"


def test_module_has_no_numerical_import_at_load():
    import subprocess

    code = "import sys;sys.path.insert(0,'scripts');import runtime_import_check;assert not {'torch','numpy','mujoco','stable_baselines3'}&sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5)


@pytest.mark.parametrize("case", ["complete", "cutoff", "extra"])
def test_compressed_events_retain_flushed_prefix_on_cutoff(case):
    import gzip
    import io

    from runtime_import_check import recover_events, write_event

    buffer = io.BytesIO()
    stream = gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=1)
    rows = [{"phase": "start"}, {"phase": "import_begin", "module": "torch"}]
    for row in rows:
        write_event(stream, row)
    partial = buffer.getvalue()
    stream.close()
    blob = partial if case == "cutoff" else buffer.getvalue()
    if case == "extra":
        blob += b"declared raw stderr"
    observed = recover_events(blob)
    assert observed["events"] == rows and observed["finalized"] == (case != "cutoff")
    assert observed["extra_bytes_hex"] == (
        b"declared raw stderr".hex() if case == "extra" else ""
    )
