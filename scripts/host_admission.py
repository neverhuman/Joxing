"""Read-only, prospective host admission; no service/priority/affinity changes."""

import math
import os
import shutil
import time
from pathlib import Path


def parse_psi(raw):
    result = {}
    for line in raw.splitlines():
        name, *fields = line.split()
        if name in result:
            raise ValueError("Duplicate PSI row")
        values = {}
        for field in fields:
            key, value = field.split("=")
            if key in values:
                raise ValueError("Duplicate PSI field")
            number = int(value) if key == "total" else float(value)
            if (
                not math.isfinite(number)
                or number < 0
                or (key != "total" and number > 100)
            ):
                raise ValueError("Invalid PSI value")
            values[key] = number
        if not {"avg10", "avg60", "avg300", "total"} <= values.keys():
            raise ValueError("Incomplete PSI row")
        result[name] = values
    return result


def snapshot(*, ignored_pids=(), base=Path("/home/ubuntu/joxing")):
    jobs = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in ignored_pids:
            continue
        try:
            cmd = (d / "cmdline").read_bytes()
            if str((d / "cwd").resolve()).startswith(str(base) + "/") and (
                b"python" in cmd or b"/uv " in cmd
            ):
                stat = (d / "stat").read_text().rsplit(")", 1)[1].split()
                jobs.append({"pid": int(d.name), "start_ticks": int(stat[19])})
        except (OSError, ValueError):
            continue
    memory = (
        int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    return {
        "monotonic_s": time.monotonic(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "load": list(os.getloadavg()),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "available_memory_bytes": memory,
        "disk_free_bytes": shutil.disk_usage(base).free,
        "psi": {
            k: parse_psi(Path("/proc/pressure", k).read_text())
            for k in ("cpu", "io", "memory")
        },
        "jobs": jobs,
    }


def reasons(sample, policy, now):
    bad = []
    try:
        age = now - sample["monotonic_s"]
        if not math.isfinite(age) or not 0 <= age <= policy["fresh_s"]:
            bad.append("stale_or_future")
        cpus = sample["affinity_cpus"]
        if type(cpus) is not int or cpus <= 0:
            bad.append("affinity")
        load = sample["load"][0]
        if (
            not math.isfinite(load)
            or load < 0
            or load > policy["load_per_affinity_cpu"] * cpus
        ):
            bad.append("load")
        for resource, row, limit in (
            ("cpu", "some", "cpu_some_avg10"),
            ("io", "full", "io_full_avg10"),
            ("memory", "full", "memory_full_avg10"),
        ):
            value = sample["psi"][resource][row]["avg10"]
            if not math.isfinite(value) or not 0 <= value <= policy[limit]:
                bad.append(resource + "_pressure")
        if sample["available_memory_bytes"] < policy["minimum_memory_bytes"]:
            bad.append("memory_floor")
        if sample["disk_free_bytes"] < policy["minimum_disk_bytes"]:
            bad.append("disk_floor")
        if sample["jobs"]:
            bad.append("owned_jobs")
    except (KeyError, TypeError, IndexError, ValueError):
        bad.append("missing_or_invalid_data")
    return bad


def assess_pair(samples, policy, now):
    failures = []
    if len(samples) != 2:
        failures.append("two_snapshots_required")
    else:
        try:
            failures += [
                "first:" + x
                for x in reasons(samples[0], policy, samples[0]["monotonic_s"])
            ]
            failures += ["second:" + x for x in reasons(samples[1], policy, now)]
            if (
                samples[1]["monotonic_s"] - samples[0]["monotonic_s"]
                < policy["spacing_s"]
            ):
                failures.append("spacing")
        except (KeyError, TypeError, ValueError):
            failures.append("invalid_pair_data")
    return {
        "status": "deferred" if failures else "admitted",
        "reasons": failures,
        "samples": samples,
        "policy": policy,
        "assessed_monotonic_s": now,
    }


class AdmissionDeferred(RuntimeError):
    pass
