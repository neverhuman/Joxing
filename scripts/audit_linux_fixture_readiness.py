"""Independent saved H29 source evidence audit; no simulation or learner imports."""

import argparse
import hashlib
import json
import os
import resource
import time
from itertools import pairwise
from pathlib import Path

from audit_lean_worker_probe import check_semantics
from audit_native_learner_context import (
    check_runtime,
    digest,
    gunzip,
    require,
    verify_admission,
    verify_wire,
)
from qualify_linux_fixture_readiness import ROOT, verify
from source_receipt_audit import audit_junit
from supervise_motor_pilot import identity, live
from upper_task_gate_support import partition_usage

from joxing.receipt_fixture_paths import validate_root


def check_pair(fixtures):
    proofs = list(fixtures.glob("controls/**/proof.json"))
    require(len(proofs) == 1, "Exactly one fresh source-pair proof required")
    v = json.loads(proofs[0].read_text())
    require(
        v["native_physics"] == 0 and len(v["initialization"]) == 2,
        "Zero-native initialization denominator",
    )
    require(
        v["initialization"][0] == v["initialization"][1] == v["rows"][0]["response"],
        "Initialization response/state binding",
    )
    require(
        v["imports"]
        == [
            {"torch": True, "stable_baselines3": True, "mujoco": False},
            {"torch": False, "stable_baselines3": False, "mujoco": False},
        ],
        "Fake pair import isolation",
    )
    require(v["peer_rows"] == v["rows"], "Entire peer command/state/RNG equality")
    require(
        len(v["readiness_wall_s"]) == 2
        and all(0 <= t <= 8 for t in v["readiness_wall_s"]),
        "Existing8s readiness cap",
    )
    semantic = check_semantics(v["rows"])
    processes = []
    for target in ("_worker", "lean_pipe_worker"):
        rows = [
            json.loads(x)
            for x in proofs[0]
            .with_name(target + "-init.jsonl")
            .read_text()
            .splitlines()
        ]
        require(
            [r["phase"] for r in rows]
            == ["factory_entered", "factory_ready", "managed_finalizer"],
            "Incomplete startup/finalizer events",
        )
        require(
            all(r["identity"] == rows[0]["identity"] for r in rows),
            "Initialization identity changed",
        )
        require(
            all(a["monotonic_s"] <= b["monotonic_s"] for a, b in pairwise(rows)),
            "Initialization chronology",
        )
        bound = {
            "pid": rows[0]["identity"]["pid"],
            "start_ticks": int(rows[0]["identity"]["start"]),
        }
        require(not live(bound), "Fake worker not exited")
        processes.append({"target": target, "identity": bound, "events": rows})
    return {
        "commands": semantic["commands"],
        "initialization_responses": 2,
        "phases": processes,
        "scope": "Source fake IPC semantics only; no formal RSS/native result",
    }


def main():
    ap = argparse.ArgumentParser()
    for name in ("packet", "folder", "source-sha", "ready-file"):
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args()
    started = time.monotonic()
    out = Path(a.folder)
    p = json.loads((ROOT / a.packet).read_text())
    me = identity(os.getpid())
    value = {
        "schema": "joxing-independent-linux-source-readiness-audit-v1",
        "audit_process": me,
        "qualified_source_only": False,
        "checks": {},
        "native_physics": 0,
        "formal_probes": 0,
        "policy_actors": 0,
        "optimizer_updates": 0,
    }
    try:
        while not Path(a.ready_file).exists():
            if time.monotonic() - started > 2:
                raise TimeoutError("Audit/controller readiness barrier")
            time.sleep(0.01)
        verify(ROOT, p, a.source_sha, digest(p))
        r = json.loads(gunzip((out / "result.json.gz").read_bytes(), 512 * 1024))
        controls = [s for s in r["stages"] if s["stage"] == "controls"]
        rc = controls[0].get("returncode") if controls else None
        value["source_gate"] = audit_junit(
            out, p["test_node_ids"], rc, started=bool(controls)
        )
        bootstrap = verify_wire(ROOT, out, p, r, prefix="portable-source")
        require(
            r["source_commit"] == a.source_sha and r["protocol_sha256"] == digest(p),
            "Saved source binding",
        )
        require(
            r["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "Original host boot",
        )
        controller = json.loads(
            (out.parent / "portable-source-controller.json").read_text()
        )
        require(
            controller["status"] == "audit_running"
            and controller["audit_process"] == me,
            "Bound detached audit owner",
        )
        require(
            controller["controller"] == r["controller"]
            and controller["controller"]["pid"] == os.getppid(),
            "Audit controller ancestry",
        )
        require(
            bootstrap["process"]["pid"] == r["controller"]["pid"]
            and bootstrap["process"]["start_ticks"] == r["controller"]["start_ticks"],
            "Bootstrap owner identity",
        )
        require(not live(r["supervisor"]), "Source writer still live")
        owned = {
            (i["pid"], i["start_ticks"]): i
            for i in r["owned_identities"] + controller["owned_identities"]
        }
        require(
            not any(live(i) for i in owned.values()) and not r["final_live_recorded"],
            "Owned source/control writer live",
        )
        value["checks"]["owned_exits"] = list(owned.values())
        admission = bootstrap["admission"]
        samples = admission["samples"]
        require(
            admission["status"] == "admitted" and len(samples) == 2,
            "Missing fixed initial admission",
        )
        for s in samples:
            verify_admission(s, p["admission_policy"], s["monotonic_s"])
        require(
            samples[1]["monotonic_s"] - samples[0]["monotonic_s"]
            >= p["admission_policy"]["spacing_s"],
            "Initial admission spacing",
        )
        verify_admission(
            bootstrap["post_checkout_admission"],
            p["admission_policy"],
            bootstrap["post_checkout_assessed_monotonic_s"],
        )
        verify_admission(
            r["runtime_admission"]["sample"],
            p["admission_policy"],
            r["runtime_admission"]["assessed_monotonic_s"],
        )
        value["checks"]["admission"] = (
            "fixed policy/fresh initial/post-checkout/runtime samples"
        )
        value["checks"]["runtime"] = check_runtime(out, p, r)
        _, manifest, _ = validate_root(
            out / "fixtures/receipts", p["fixture_bundle"]["manifest_sha256"]
        )
        require(
            manifest == p["fixture_bundle"]["manifest"] and len(manifest["files"]) == 8,
            "Exact eight original inputs/negative siblings",
        )
        value["checks"]["fixtures"] = {
            "files": manifest["files"],
            "absent": manifest["absent"],
            "charged_bytes": p["fixture_bundle"]["charged_bytes"],
            "aliases_created": False,
        }
        controls = [s for s in r["stages"] if s["stage"] == "controls"]
        rc = controls[0].get("returncode") if controls else None
        value["source_gate"] = audit_junit(
            out, p["test_node_ids"], rc, started=bool(controls)
        )
        for stage in r["stages"]:
            require(
                stage["peak_sampled_rss_bytes"] <= p["owned_rss_bytes"]
                and stage["peak_processes"] <= p["process_ceiling"],
                "Source resource first limit",
            )
        if controls:
            require(
                controls[0].get("wall_s", controls[0]["elapsed_s"])
                <= p["control_wall_s"],
                "Source30s limit",
            )
        require(
            r["wall_s"] <= 103
            and all(
                r[k] == 0
                for k in (
                    "native_physics",
                    "native_models",
                    "policy_actors",
                    "optimizer_updates",
                    "formal_probes",
                )
            ),
            "Source wall/no native scope",
        )
        if r["status"] == "completed":
            require(
                value["source_gate"]["qualified"]
                and value["checks"]["runtime"]["status"] == "qualified",
                "False completed prerequisite",
            )
            require(
                r["rows"] == [{"id": "linux-source", "status": "completed"}],
                "Source denominator",
            )
            value["checks"]["pair"] = check_pair(out / "fixtures")
            value["qualified_source_only"] = True
        else:
            require(
                r["status"] in ("halted", "deferred")
                and not value["source_gate"]["qualified"],
                "Unexplained source halt",
            )
        manifest = {}
        for file in sorted(out.parent.rglob("*")):
            if file.is_symlink():
                manifest[str(file.relative_to(out.parent))] = {
                    "symlink": os.readlink(file)
                }
            elif file.is_file():
                data = file.read_bytes()
                manifest[str(file.relative_to(out.parent))] = {
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
        value.update(
            raw_manifest=manifest,
            source_status=r["status"],
            source_commit=a.source_sha,
            protocol_sha256=digest(p),
            audit_completed=True,
        )
    except Exception as error:  # noqa: BLE001 -- preserve an independently reviewable fault receipt
        value.update(
            audit_completed=False,
            qualified_source_only=False,
            error=type(error).__name__ + ": " + str(error),
        )
    value.update(
        audit_wall_s=time.monotonic() - started,
        peak_audit_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    if "r" in locals():
        value["original_epoch_elapsed_s"] = time.monotonic() - r["started_monotonic_s"]
    raw, receipts = partition_usage(out.parent, out / "fixtures")
    value["usage_before_audit_receipt"] = {"fixtures": raw, "receipts": receipts}
    blob = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if (
        raw > p["fixture_bytes"]
        or receipts + len(blob) + 8192 > p["receipt_bytes"]
        or value["audit_wall_s"] > p["audit_wall_s"]
        or value.get("original_epoch_elapsed_s", 121) > 118
    ):
        value.update(
            audit_completed=False,
            qualified_source_only=False,
            limit_error="Independent storage/wall first limit",
        )
        blob = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
    with (out / "independent-audit.json").open("xb") as stream:
        stream.write(blob)
    print(
        json.dumps(
            {
                k: value.get(k)
                for k in (
                    "audit_completed",
                    "qualified_source_only",
                    "error",
                    "audit_wall_s",
                    "original_epoch_elapsed_s",
                )
            }
        )
    )
    if not value["audit_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
