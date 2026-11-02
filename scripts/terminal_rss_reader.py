"""Opt-in conservative charges for narrowly verified single-thread mm teardown."""

import time
from pathlib import Path

from probe_exit_memory import stat_fields


def terminal_evidence(bound, proc_root=Path("/proc")):
    folder = proc_root / str(bound["pid"])
    expected = [str(bound["pid"])]
    before = stat_fields(folder / "stat")
    if before["start_ticks"] != bound["start_ticks"] or not before["flags"] & 4:
        raise RuntimeError(
            f"Unsupported terminal identity/flag pid={bound['pid']} start={bound['start_ticks']}"
        )
    tasks = sorted(p.name for p in (folder / "task").iterdir())
    fields = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in (folder / "status").read_text().splitlines()
        if ":" in line
    }
    pages = [int(s) for s in (folder / "statm").read_text().split()]
    task = stat_fields(folder / "task" / str(bound["pid"]) / "stat")
    after = stat_fields(folder / "stat")
    after_tasks = sorted(p.name for p in (folder / "task").iterdir())
    if (
        tasks != expected
        or after_tasks != expected
        or fields.get("Threads") != "1"
        or "VmRSS" in fields
        or "VmSize" in fields
        or pages != [0] * 7
        or any(
            s["start_ticks"] != bound["start_ticks"] or not s["flags"] & 4
            for s in (before, task, after)
        )
    ):
        raise RuntimeError(
            f"Unsupported terminal RSS evidence pid={bound['pid']} start={bound['start_ticks']}"
        )
    return {
        "pid": bound["pid"],
        "start_ticks": bound["start_ticks"],
        "before": before,
        "after": after,
        "task": task,
        "tasks": tasks,
        "after_tasks": after_tasks,
        "status_threads": 1,
        "vmrss_absent": True,
        "vmsize_absent": True,
        "statm_pages": pages,
    }


class TerminalReader:
    def __init__(
        self, read_rss, *, evidence=terminal_evidence, clock=time.monotonic, hold_s=0.1
    ):
        self.read_rss, self.evidence, self.clock, self.hold_s = (
            read_rss,
            evidence,
            clock,
            hold_s,
        )
        self.highwater = {}
        self.retiring = {}
        self.records = {}

    def sample(self, bound):
        key = bound["pid"], bound["start_ticks"]
        name = f"{key[0]}:{key[1]}"
        try:
            value = self.read_rss(bound)
        except RuntimeError:
            try:
                proof = self.evidence(bound)
            except OSError:
                # A disappearing task must be verified by the original bound reader.
                return self.sample_after_race(bound)
            if proof["after"]["state"] in ("Z", "X"):
                return self.observed_or_released(bound, None)
            if self.highwater.get(key, 0) <= 0:
                raise RuntimeError("Terminal RSS lacks prior observation") from None
            now = self.clock()
            since = self.retiring.setdefault(key, now)
            age = now - since
            record = self.records.setdefault(
                name,
                {
                    "pid": key[0],
                    "start_ticks": key[1],
                    "first_retiring_s": since,
                    "evidence": proof,
                    "highwater_charge_bytes": self.highwater[key],
                    "samples": 0,
                    "max_hold_s": 0.0,
                    "released": False,
                },
            )
            record["samples"] += 1
            record["max_hold_s"] = max(record["max_hold_s"], age)
            if age > self.hold_s:
                raise RuntimeError("Retiring RSS hold first limit")
            return {
                "rss_bytes": None,
                "retiring_rss_charge_bytes": self.highwater[key],
                "capacity_charge_bytes": self.highwater[key],
                "kind": "retiring",
            }
        return self.observed_or_released(bound, value)

    def sample_after_race(self, bound):
        return self.observed_or_released(bound, self.read_rss(bound))

    def observed_or_released(self, bound, value):
        key = bound["pid"], bound["start_ticks"]
        name = f"{key[0]}:{key[1]}"
        now = self.clock()
        if key in self.retiring:
            age = now - self.retiring[key]
            self.records[name]["max_hold_s"] = max(
                self.records[name]["max_hold_s"], age
            )
            if age > self.hold_s:
                raise RuntimeError("Retiring RSS hold first limit at release")
            if value is not None:
                raise RuntimeError("Unexpected live RSS after terminal evidence")
            self.records[name].update(released=True, released_s=now)
            del self.retiring[key]
        if value is None:
            return None
        self.highwater[key] = max(self.highwater.get(key, 0), value)
        return {
            "rss_bytes": value,
            "retiring_rss_charge_bytes": 0,
            "capacity_charge_bytes": value,
            "kind": "observed",
        }


def aggregate(samples):
    observed = sum(s["rss_bytes"] or 0 for s in samples)
    reserved = sum(s.get("retiring_rss_charge_bytes", 0) for s in samples)
    return observed, reserved, observed + reserved
