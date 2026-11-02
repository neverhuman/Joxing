"""Fresh-process module qualification with exact case identity and owned limits."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from joxing_process_ownership import unknown_joxing_processes
from source_shard_support import check_junit, create_fixture_root, partition
from supervise_motor_pilot import descendants, identity, live
from supervise_upper_task_tests import canonical, rss, save, stop
from upper_task_gate_support import KNOWN_XFAIL, partition_usage

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    p = json.loads((ROOT / args.packet).read_text())
    if (
        p["schema"] != "joxing-isolated-source-v1"
        or p["mode"] not in ("diagnostic", "full_source")
        or canonical(p) != args.protocol_sha256
        or json.loads(
            subprocess.check_output(["git", "show", "HEAD:" + args.packet], cwd=ROOT)
        )
        != p
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != args.source_sha
        or p["expected_xfails"] not in ([], [KNOWN_XFAIL])
        or (p["mode"] == "full_source" and p["expected_xfails"] != [KNOWN_XFAIL])
        or p["shards"] != partition(p["node_ids"], p["expected_xfails"])
    ):
        raise ValueError("Frozen isolated-source identity/partition")

    def verify():
        if (
            subprocess.check_output(
                ["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, text=True
            ).strip()
            != "false"
        ):
            raise ValueError("Source audit requires complete superproject history")
        for name, expected in p["required_git_objects"].items():
            if (
                hashlib.sha256(
                    subprocess.check_output(["git", "show", name], cwd=ROOT)
                ).hexdigest()
                != expected
            ):
                raise ValueError("Historical executable: " + name)
        for name, d in p["source_files"].items():
            f = ROOT / name
            if (
                f.stat().st_size != d["bytes"]
                or hashlib.sha256(f.read_bytes()).hexdigest() != d["sha256"]
            ):
                raise ValueError("Isolated source file drift: " + name)
        for name, revision in p["upstream_commits"].items():
            if (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT / "third_party" / name,
                    text=True,
                ).strip()
                != revision
            ):
                raise ValueError("Upstream pin drift")
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=False
        ).returncode:
            raise ValueError("Isolated source dirty checkout")

    verify()
    out = ROOT / p["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".isolated-source.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    fixtures = create_fixture_root(out)
    supervisor = identity(os.getpid())
    ancestors, pid = set(), os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = identity(pid)["parent_pid"]

    def capacity():
        foreign = unknown_joxing_processes(supervisor, ignored_pids=ancestors)
        if foreign:
            raise RuntimeError("Other owned Joxing process: " + str(foreign))
        available = (
            int(
                next(
                    s.split()[1]
                    for s in Path("/proc/meminfo").read_text().splitlines()
                    if s.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        disk = shutil.disk_usage(ROOT).free
        scratch, receipts = partition_usage(out.parent, fixtures)
        if available < p["minimum_memory_bytes"] or disk < p["minimum_disk_bytes"]:
            raise RuntimeError(
                f"Shared-host capacity: available={available}, disk={disk}"
            )
        if scratch > p["fixture_bytes"] or receipts > p["receipt_bytes"]:
            raise RuntimeError(
                f"Storage: fixture={scratch}/{p['fixture_bytes']}, receipts={receipts}/{p['receipt_bytes']}"
            )
        return {
            "available_memory_bytes": available,
            "free_disk_bytes": disk,
            "fixture_bytes": scratch,
            "receipt_bytes": receipts,
            "load": os.getloadavg(),
        }

    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT), MUJOCO_GL="egl")
    python = str(ROOT / "experiments/v0.0.11/runtime/.venv/bin/python")
    commands = [
        {
            "stage": "sync",
            "command": [
                shutil.which("uv") or "/home/ubuntu/.local/bin/uv",
                "sync",
                "--project",
                "experiments/v0.0.11/runtime",
                "--locked",
            ],
        },
        {
            "stage": "runtime",
            "command": [
                python,
                "-c",
                "import json,sys,numpy,mujoco; v={'python':sys.version.split()[0],'numpy':numpy.__version__,'mujoco':mujoco.__version__}; print(json.dumps(v)); assert v=="
                + repr(p["runtime_versions"]),
            ],
        },
    ]
    commands.append(
        {
            "stage": "collection",
            "command": [
                python,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                *(
                    ["tests", "grokbot/tests"]
                    if p["mode"] == "full_source"
                    else [s["file"] for s in p["shards"]]
                ),
            ],
        }
    )
    for index, shard in enumerate(p["shards"]):
        name = f"module-{index:03d}"
        commands.append(
            {
                "stage": name,
                "shard_index": index,
                "command": [
                    python,
                    "-m",
                    "pytest",
                    "-vv",
                    "-rx",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp",
                    str(fixtures / name),
                    "--junitxml",
                    str(out / (name + ".xml")),
                    *shard["node_ids"],
                ],
            }
        )
    report = {
        "schema": "joxing-isolated-source-supervision-v1",
        "source_commit": args.source_sha,
        "protocol_sha256": args.protocol_sha256,
        "process": supervisor,
        "status": "running",
        "scope": p["scope"],
        "formal_native_evaluations": 0,
        "formal_training_runs": 0,
        "initial_capacity": capacity(),
        "rows": [{"node": n, "status": "not_run"} for n in p["node_ids"]],
        "stages": [dict(c, status="not_run") for c in commands],
    }
    began = time.monotonic()
    test_started = None
    active, seen, row = None, {}, None

    def interrupted(signum, _frame):
        raise TimeoutError("Owned isolated-source interruption: " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save(out / "result.json", report)
    try:
        for row in report["stages"]:
            name = row["stage"]
            is_module = "shard_index" in row
            if is_module and test_started is None:
                test_started = time.monotonic()
            limit = p["per_module_wall_s"] if is_module else p["setup_wall_s"][name]
            limit = min(limit, p["whole_wall_s"] - (time.monotonic() - began))
            if is_module:
                limit = min(
                    limit, p["total_test_wall_s"] - (time.monotonic() - test_started)
                )
            limit -= p["stop_reserve_s"]
            if limit <= 0:
                raise TimeoutError("No isolated-source stage credit")
            started = time.monotonic()
            row.update(status="running", limit_s=limit, peak_owned_rss_bytes=0)
            with (out / (name + ".log")).open("w") as log:
                active = subprocess.Popen(
                    row["command"],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                row["process"] = identity(active.pid)
                seen = {}
                while True:
                    for child in descendants(active.pid):
                        seen[(child["pid"], child["start_ticks"])] = child
                    living = [i for i in seen.values() if live(i)]
                    used_rss = rss(supervisor) + sum(rss(i) for i in living)
                    row.update(
                        owned_identities=list(seen.values()),
                        elapsed_s=time.monotonic() - started,
                        current_owned_rss_bytes=used_rss,
                        live_owned_processes=1 + len(living),
                        peak_owned_rss_bytes=max(row["peak_owned_rss_bytes"], used_rss),
                        capacity=capacity(),
                    )
                    save(out / "result.json", report)
                    if (
                        row["elapsed_s"] >= limit
                        or used_rss > p["owned_rss_bytes"]
                        or 1 + len(living) > p["process_ceiling"]
                    ):
                        raise TimeoutError(
                            f"First stage limit: elapsed={row['elapsed_s']}/{limit}, rss={used_rss}/{p['owned_rss_bytes']}, processes={1 + len(living)}/{p['process_ceiling']}"
                        )
                    if active.poll() is not None:
                        break
                    time.sleep(0.5)
                row.update(returncode=active.wait(), wall_s=time.monotonic() - started)
                cleanup_started = time.monotonic()
                while (
                    any(live(i) for i in seen.values())
                    and time.monotonic() - cleanup_started < 0.5
                ):
                    time.sleep(0.05)
                row.update(
                    cleanup_wait_s=time.monotonic() - cleanup_started,
                    wall_s=time.monotonic() - started,
                )
                if any(live(i) for i in seen.values()):
                    raise RuntimeError("Surviving isolated-source descendant")
                active = None
            if is_module:
                shard = p["shards"][row["shard_index"]]
                checked = check_junit(
                    (out / (name + ".xml")).read_text(),
                    shard["node_ids"],
                    shard["expected_xfails"],
                )
                statuses = {r["node"]: r["status"] for r in checked["rows"]}
                for case in report["rows"]:
                    if case["node"] in statuses:
                        case["status"] = statuses[case["node"]]
                row["junit_counts"] = checked["counts"]
                if not checked["qualified"]:
                    raise ValueError(
                        "Module contains unsuccessful or unexpected source case"
                    )
            elif name == "collection":
                actual = [
                    s
                    for s in (out / (name + ".log")).read_text().splitlines()
                    if s.startswith(("tests/", "grokbot/tests/")) and "::" in s
                ]
                if actual != p["node_ids"]:
                    raise ValueError(
                        "Frozen actual collection differs from registered denominator"
                    )
            if row["returncode"]:
                raise RuntimeError("Isolated-source stage nonzero exit")
            row["status"] = "completed"
            save(out / "result.json", report)
        verify()
        if any(
            r["status"] != ("xfail" if r["node"] in p["expected_xfails"] else "passed")
            for r in report["rows"]
        ):
            raise ValueError("Incomplete exact source case denominator")
        report.update(
            status="completed",
            criterion_passed=True,
            full_source_gate_qualified=p["mode"] == "full_source",
        )
    except Exception as error:  # noqa: BLE001 -- retain all failed/not_run attempts
        if active is not None:
            stop(active, seen)
            row.update(
                returncode=active.poll(),
                wall_s=time.monotonic() - started,
                owned_identities=list(seen.values()),
            )
        if row is not None and row["status"] == "running":
            row["status"] = "failed"
        if row is not None and "shard_index" in row:
            uncertain = set(p["shards"][row["shard_index"]]["node_ids"])
            for case in report["rows"]:
                if case["node"] in uncertain and case["status"] == "not_run":
                    case["status"] = "unconfirmed"
        report.update(
            status="halted",
            criterion_passed=False,
            full_source_gate_qualified=False,
            error=type(error).__name__ + ": " + str(error),
        )
    finally:
        report.update(
            wall_s=time.monotonic() - began,
            test_wall_s=0 if test_started is None else time.monotonic() - test_started,
            final_live_recorded=[i for i in seen.values() if live(i)],
        )
        report["fixture_bytes"], report["receipt_bytes"] = partition_usage(
            out.parent, fixtures
        )
        save(out / "result.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "stages")}))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
