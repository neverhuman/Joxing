"""Import/version check with causal phase receipts; no native model or learning."""

import argparse
import gzip
import importlib
import json
import os
import resource
import sys
import time
import zlib
from pathlib import Path


def observe():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": os.getpid(),
        "start_ticks": int(stat[19]),
        "monotonic_s": time.monotonic(),
        "process_cpu_s": time.process_time(),
        "user_s": usage.ru_utime,
        "system_s": usage.ru_stime,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_switches": usage.ru_nvcsw,
        "involuntary_switches": usage.ru_nivcsw,
        "io": Path("/proc/self/io").read_text(),
        "status": "\n".join(
            x
            for x in Path("/proc/self/status").read_text().splitlines()
            if x.startswith(("VmRSS:", "VmSize:", "Threads:"))
        ),
        "pressure": {
            k: Path("/proc/pressure", k).read_text() for k in ("cpu", "io", "memory")
        },
        "load": list(os.getloadavg()),
    }


def run(expected, *, importer=importlib.import_module, observer=observe, emit=None):
    emit = emit or (
        lambda value: print(
            json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True
        )
    )
    versions = {"python": sys.version.split()[0]}
    emit({"phase": "start", "observation": observer()})
    for name, key in (
        ("numpy", "numpy"),
        ("mujoco", "mujoco"),
        ("torch", "torch"),
        ("stable_baselines3", "sb3"),
    ):
        emit({"phase": "import_begin", "module": name, "observation": observer()})
        try:
            module = importer(name)
            versions[key] = module.__version__
        except BaseException as error:
            emit(
                {
                    "phase": "import_fault",
                    "module": name,
                    "error": type(error).__name__ + ": " + str(error),
                    "observation": observer(),
                }
            )
            raise
        emit(
            {
                "phase": "import_end",
                "module": name,
                "version": versions[key],
                "observation": observer(),
            }
        )
    emit(
        {
            "phase": "version_assert",
            "versions": versions,
            "expected": expected,
            "observation": observer(),
        }
    )
    if versions != expected:
        raise ValueError("Runtime versions mismatch")
    emit({"phase": "qualified", "versions": versions, "observation": observer()})
    return versions


def write_event(stream, value):
    stream.write(
        (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    )
    stream.flush(zlib.Z_SYNC_FLUSH)


def recover_events(blob, limit=65536):
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(blob, limit + 1)
    if len(raw) > limit or decoder.unconsumed_tail:
        raise ValueError("Runtime event bound")
    lines = raw.splitlines(keepends=True)
    tail = lines.pop() if lines and not lines[-1].endswith(b"\n") else b""
    return {
        "events": [json.loads(x) for x in lines],
        "finalized": decoder.eof,
        "partial_json": tail.decode(errors="replace"),
        "extra_bytes_hex": decoder.unused_data.hex(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument(
        "--event-wire", choices=["plain", "gzip-jsonl-v1"], default="plain"
    )
    args = parser.parse_args()
    if args.event_wire == "gzip-jsonl-v1":
        with gzip.GzipFile(
            fileobj=sys.stdout.buffer, mode="wb", mtime=0, compresslevel=1
        ) as stream:
            run(
                json.loads(args.expected), emit=lambda value: write_event(stream, value)
            )
    else:
        run(json.loads(args.expected))
