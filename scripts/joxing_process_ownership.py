"""Read actual ancestry rather than compare two racing process inventories."""

from pathlib import Path

from supervise_motor_pilot import identity


def belongs_to_supervisor(pid, supervisor, *, read_identity=identity):
    """True=owned, False=foreign, None=vanished during inspection.

    Bind the root to PID AND start ticks. A child created after a previous
    inventory still belongs when its current ancestry reaches the owned root.
    """
    seen = set()
    while pid > 1 and pid not in seen and len(seen) < 128:
        seen.add(pid)
        try:
            current = read_identity(pid)
        except OSError:
            return None
        if pid == supervisor["pid"]:
            return current["start_ticks"] == supervisor["start_ticks"]
        pid = current["parent_pid"]
    return False


def unknown_joxing_processes(supervisor, *, ignored_pids=()):
    unknown = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ignored_pids:
            continue
        try:
            cmd = (proc / "cmdline").read_bytes()
            cwd = str((proc / "cwd").resolve())
            if "/joxing/" not in cwd or not (b"python" in cmd or b"/uv " in cmd):
                continue
            row = identity(int(proc.name))
            if belongs_to_supervisor(row["pid"], supervisor) is False:
                unknown.append(row | {"cwd": cwd})
        except OSError:
            continue
    return unknown
