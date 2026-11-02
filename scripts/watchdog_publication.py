"""Bounded private report IPC and post-exit persistence; no native imports."""

import hashlib
import json
import os
import resource
import shutil
import stat
import tempfile
import time
from pathlib import Path


class PhaseTiming:
    def __init__(self):
        self.data = {
            "max_wall_s": {},
            "max_cpu_s": {},
            "max_sleep_excess_s": 0,
            "recent": [],
            "self_final_write_included": False,
        }

    def begin(self, wall=None):
        return (
            time.monotonic() if wall is None else wall,
            time.process_time(),
            resource.getrusage(resource.RUSAGE_SELF),
        )

    def end(self, phase, began, *, wall=None, requested_s=None):
        now = time.monotonic() if wall is None else wall
        usage = resource.getrusage(resource.RUSAGE_SELF)
        row = {
            "phase": phase,
            "begin_s": began[0],
            "end_s": now,
            "wall_s": now - began[0],
            "cpu_s": time.process_time() - began[1],
            "minor_faults": usage.ru_minflt - began[2].ru_minflt,
            "major_faults": usage.ru_majflt - began[2].ru_majflt,
            "voluntary_switches": usage.ru_nvcsw - began[2].ru_nvcsw,
            "involuntary_switches": usage.ru_nivcsw - began[2].ru_nivcsw,
        }
        if requested_s is not None:
            row["requested_s"] = requested_s
            row["excess_s"] = max(0, row["wall_s"] - requested_s)
            self.data["max_sleep_excess_s"] = max(
                self.data["max_sleep_excess_s"], row["excess_s"]
            )
        for key in ("wall_s", "cpu_s"):
            values = self.data["max_" + key]
            values[phase] = max(values.get(phase, 0), row[key])
        self.data["recent"].append(row)
        self.data["recent"] = self.data["recent"][-32:]


def tmpfs_proof(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError("Private IPC base must be a real absolute mount")
    rows = [
        line
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if line.split()[4] == str(path)
    ]
    if len(rows) != 1 or rows[0].split(" - ", 1)[1].split()[0] != "tmpfs":
        raise ValueError("Private IPC requires existing tmpfs")
    return {"mountinfo": rows[0], "base": str(path)}


def own_identity():
    pid = os.getpid()
    f = Path(f"/proc/{pid}/stat")
    return (
        {"pid": pid, "start": f.read_text().rsplit(")", 1)[1].split()[19]}
        if f.exists()
        else {"pid": pid, "start": "development-only"}
    )


class ReportScope:
    def __init__(
        self, canonical, *, base=Path("/dev/shm"), cap=1024**2, mount_check=tmpfs_proof
    ):
        self.canonical = Path(canonical)
        if self.canonical.exists() or self.canonical.with_suffix(".ready").exists():
            raise ValueError("Canonical report already exists")
        self.mount = mount_check(base)
        if not 256 * 1024 <= cap <= 1024**2 or shutil.disk_usage(base).free < cap:
            raise ValueError("Private IPC credit/capacity")
        self.cap = cap
        self.owner = own_identity()
        self.directory = Path(tempfile.mkdtemp(prefix="joxing-watch-", dir=base))
        s = self.directory.stat()
        self.key = (s.st_dev, s.st_ino)
        self.allocation = {
            "uid": s.st_uid,
            "mode": stat.S_IMODE(s.st_mode),
            "directory": stat.S_ISDIR(s.st_mode),
        }
        self.path = self.directory / "report.json"
        self.receipt = self.canonical.with_name(
            self.canonical.stem + "-publication.json"
        )
        self.peak_bytes = 0
        self.finished = False
        self.failed = False
        self._validate()
        self._save({"status": "allocated", "allocated_s": time.monotonic()})

    def _save(self, values):
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": "joxing-private-watch-report-v1",
            "owner": self.owner,
            "directory": str(self.directory),
            "directory_device_inode": list(self.key),
            "allocation_stat": self.allocation,
            "volatile_report": str(self.path),
            "canonical_report": str(self.canonical),
            "mount": self.mount,
            "cap_bytes": self.cap,
            "peak_ipc_bytes": self.peak_bytes,
            **values,
        }
        tmp = self.receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
        tmp.replace(self.receipt)

    def _validate(self):
        if own_identity() != self.owner or self.directory.is_symlink():
            raise RuntimeError("Foreign/reused IPC owner")
        s = self.directory.stat()
        if (
            (s.st_dev, s.st_ino) != self.key
            or s.st_uid != os.getuid()
            or stat.S_IMODE(s.st_mode) != 0o700
        ):
            raise RuntimeError("Private IPC directory identity/mode drift")
        total = 0
        for f in self.directory.iterdir():
            try:
                member = f.lstat()
            except FileNotFoundError:
                continue  # Atomic writer rename verified this old name disappeared.
            if (
                f.name not in ("report.json", "report.tmp", "report.ready")
                or not stat.S_ISREG(member.st_mode)
                or member.st_uid != os.getuid()
            ):
                raise RuntimeError("Unexpected/unowned IPC member")
            size = member.st_size
            if size > 256 * 1024:
                raise RuntimeError("Report file byte credit")
            total += size
        self.peak_bytes = max(self.peak_bytes, total)
        if total > self.cap:
            raise RuntimeError("Aggregate IPC byte credit")
        return total

    def finish(self, *, task_live, guard_live, copier=None):
        if self.failed:
            raise RuntimeError("Previous publication fault; IPC retained")
        if task_live or guard_live:
            raise RuntimeError("Cannot persist/remove active report IPC")
        copied = []
        began = time.monotonic()
        try:
            self._validate()
            if not self.path.exists():
                raise RuntimeError("No volatile report to salvage")
            for name, target in [
                ("report.json", self.canonical),
                ("report.ready", self.canonical.with_suffix(".ready")),
                ("report.tmp", self.canonical.with_suffix(".partial")),
            ]:
                source = self.directory / name
                if not source.exists():
                    continue
                raw = source.read_bytes()
                expected = hashlib.sha256(raw).hexdigest()
                if target.exists():
                    raise RuntimeError("Canonical copy would overwrite evidence")
                if copier is None:
                    target.write_bytes(raw)
                else:
                    copier(source, target)
                if (
                    target.stat().st_size != len(raw)
                    or hashlib.sha256(target.read_bytes()).hexdigest() != expected
                ):
                    raise RuntimeError("Volatile/durable copy mismatch")
                copied.append(
                    {
                        "source": name,
                        "target": str(target),
                        "bytes": len(raw),
                        "sha256": expected,
                    }
                )
            self._validate()
            for f in self.directory.iterdir():
                f.unlink()
            self.directory.rmdir()
            self._save(
                {
                    "status": "copied_and_removed",
                    "copy_begin_s": began,
                    "copy_end_s": time.monotonic(),
                    "copies": copied,
                    "task_live": False,
                    "guard_live": False,
                }
            )
            self.finished = True
        except BaseException as error:
            self.failed = True
            self._save(
                {
                    "status": "fault_ipc_retained",
                    "copy_begin_s": began,
                    "fault_s": time.monotonic(),
                    "copies": copied,
                    "error": type(error).__name__ + ": " + str(error),
                }
            )
            raise
