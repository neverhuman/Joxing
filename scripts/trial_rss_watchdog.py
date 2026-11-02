"""Fast, stdlib-only RSS guard. PSS collection belongs in another process."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

from supervise_motor_pilot import identity, live


def bound_rss(bound, *, proc_root=Path("/proc"), read_identity=identity):
    """A live unreadable status invalidates monitoring; never substitute zero."""
    for attempt in range(3):
        try:
            current = read_identity(bound["pid"])
            if (
                current["start_ticks"] != bound["start_ticks"]
                or current["state"] == "Z"
            ):
                return None
            raw = (proc_root / str(bound["pid"]) / "status").read_text()
            value = (
                int(
                    next(
                        s.split()[1] for s in raw.splitlines() if s.startswith("VmRSS:")
                    )
                )
                * 1024
            )
            after = read_identity(bound["pid"])
            if after["start_ticks"] != bound["start_ticks"] or after["state"] == "Z":
                return None
            return value
        except (OSError, StopIteration) as error:
            try:
                current = read_identity(bound["pid"])
            except OSError:
                return None
            if (
                current["start_ticks"] == bound["start_ticks"]
                and current["state"] != "Z"
            ):
                if attempt < 2:
                    time.sleep(0.001)
                    continue
                raise RuntimeError(
                    f"Unreadable live owned RSS pid={bound['pid']} start={bound['start_ticks']} after3reads"
                ) from error
            return None


def fast_tree(root, *, proc_root=Path("/proc"), read_identity=identity):
    """Read children of every owned thread; no expensive host-wide inventory."""
    found, pending = {}, [root]
    while pending:
        bound = pending.pop()
        try:
            current = read_identity(bound["pid"])
            if (
                current["start_ticks"] != bound["start_ticks"]
                or current["state"] == "Z"
            ):
                continue
            key = current["pid"], current["start_ticks"]
            if key in found:
                continue
            found[key] = current
            for task in (proc_root / str(current["pid"]) / "task").iterdir():
                try:
                    children = (task / "children").read_text().split()
                except FileNotFoundError:
                    continue
                for value in children:
                    try:
                        child = read_identity(int(value))
                    except ProcessLookupError:
                        continue
                    if child["parent_pid"] == current["pid"]:
                        pending.append(child)
        except FileNotFoundError:
            continue
        except ProcessLookupError:
            continue
    return found


def stop_owned(root, seen):
    """Never signal supervisor/watchdog or foreign/reused identities."""
    if live(root) and os.getpgid(root["pid"]) == root["pid"]:
        try:
            os.killpg(root["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    for row in seen.values():
        if live(row):
            try:
                os.kill(row["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 0.5
    while any(live(row) for row in seen.values()) and time.monotonic() < deadline:
        time.sleep(0.01)
    for row in seen.values():
        if live(row):
            try:
                os.kill(row["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def save(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if len(raw) > 256 * 1024:
        raise ValueError("Watchdog report credit")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(raw)
    temporary.replace(path)


def monitor(
    root,
    supervisor,
    out,
    *,
    planned_rss,
    hard_rss,
    process_cap,
    wall_s,
    max_gap_s=0.1,
    poll_s=0.02,
    identity_cap=128,
):
    if not (0 < planned_rss < hard_rss and 0 < poll_s < max_gap_s and wall_s > 0):
        raise ValueError("Watchdog registered limits")
    if out.exists() or out.with_suffix(".ready").exists():
        raise ValueError("Watchdog output already exists")
    own = identity(os.getpid())
    if len({root["pid"], supervisor["pid"], own["pid"]}) != 3:
        raise ValueError("Watchdog roles must be separate processes")
    if not live(root) or not live(supervisor) or os.getpgid(root["pid"]) != root["pid"]:
        raise ValueError("Watchdog root/supervisor/group identity")
    began, last_sample, last_write = time.monotonic(), None, 0.0
    seen, recent = {(root["pid"], root["start_ticks"]): root}, []
    report = {
        "schema": "joxing-fast-rss-watchdog-v1",
        "status": "monitoring",
        "root": root,
        "supervisor": supervisor,
        "watchdog": own,
        "limits": {
            "planned_rss": planned_rss,
            "hard_rss": hard_rss,
            "process_cap": process_cap,
            "wall_s": wall_s,
            "max_gap_s": max_gap_s,
            "poll_s": poll_s,
            "identity_cap": identity_cap,
        },
        "samples": 0,
        "peak_sampled_rss_bytes": 0,
        "max_sample_gap_s": 0,
        "max_probe_s": 0,
        "peak_processes": 0,
    }
    root_gone = None
    try:
        while True:
            probe_started = time.monotonic()
            if not live(supervisor):
                raise RuntimeError("Supervisor exited")
            seen.update(fast_tree(root))
            if len(seen) > identity_cap:
                raise RuntimeError("Owned identity credit")
            samples = []
            for row in [supervisor, own, *seen.values()]:
                value = bound_rss(row)
                if value is not None:
                    samples.append(
                        {
                            "pid": row["pid"],
                            "start_ticks": row["start_ticks"],
                            "rss_bytes": value,
                        }
                    )
            now = time.monotonic()
            gap = 0 if last_sample is None else now - last_sample
            last_sample = now
            total = sum(s["rss_bytes"] for s in samples)
            report.update(
                samples=report["samples"] + 1,
                wall_s=now - began,
                peak_sampled_rss_bytes=max(report["peak_sampled_rss_bytes"], total),
                max_sample_gap_s=max(report["max_sample_gap_s"], gap),
                max_probe_s=max(report["max_probe_s"], now - probe_started),
                peak_processes=max(report["peak_processes"], len(samples)),
                last_sample_rss_bytes=total,
                last_sample_monotonic_s=now,
            )
            recent.append(
                {
                    "monotonic_s": now,
                    "rss_bytes": total,
                    "gap_s": gap,
                    "processes": len(samples),
                }
            )
            recent = recent[-64:]
            reason = (
                "hard_rss"
                if total > hard_rss
                else "process_cap"
                if len(samples) > process_cap
                else "stale_monitor"
                if max(gap, now - probe_started) > max_gap_s
                else "wall_cap"
                if now - began > wall_s
                else "planned_rss"
                if total >= planned_rss
                else None
            )
            if reason:
                report.update(status="stopped", reason=reason, stop_sample=samples)
                stop_owned(root, seen)
                break
            if not live(root):
                root_gone = now if root_gone is None else root_gone
                if not any(live(i) for i in seen.values()):
                    report.update(status="exited", reason="all_owned_exited")
                    break
                if now - root_gone > 0.5:
                    raise RuntimeError("Descendants survive root exit")
            if now - last_write >= 0.25 or report["samples"] == 1:
                report.update(
                    owned_identities=list(seen.values()), recent_samples=recent
                )
                save(out, report)
                if report["samples"] == 1:
                    out.with_suffix(".ready").touch(exist_ok=False)
                last_write = now
            time.sleep(poll_s)
    except BaseException as error:  # noqa: BLE001 -- stop only the registered task
        report.update(status="fault", reason=type(error).__name__ + ": " + str(error))
        stop_owned(root, seen)
    finally:
        report.update(
            owned_identities=list(seen.values()),
            recent_samples=recent,
            final_live=[i for i in seen.values() if live(i)],
            wall_s=time.monotonic() - began,
        )
        save(out, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    for key in (
        "root_pid",
        "root_start",
        "supervisor_pid",
        "supervisor_start",
        "planned_rss",
        "hard_rss",
        "process_cap",
    ):
        parser.add_argument("--" + key.replace("_", "-"), type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-s", type=float, required=True)
    parser.add_argument("--max-gap-s", type=float, default=0.1)
    args = parser.parse_args()
    report = monitor(
        {"pid": args.root_pid, "start_ticks": args.root_start},
        {"pid": args.supervisor_pid, "start_ticks": args.supervisor_start},
        args.output,
        planned_rss=args.planned_rss,
        hard_rss=args.hard_rss,
        process_cap=args.process_cap,
        wall_s=args.wall_s,
        max_gap_s=args.max_gap_s,
    )
    if report["status"] != "exited":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
