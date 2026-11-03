"""Independent, stdlib saved-evidence audit. No fake/native environment created."""

import argparse
import hashlib
import json
import os
import resource
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from audit_native_learner_context import (
    check_runtime,
    digest,
    gunzip,
    proc_proof,
    read,
    require,
    rows,
    verify_admission,
    verify_wire,
)


def encoded_dict(value):
    return dict(value["dict"])


def check_semantics(values):
    require(len(values) == 26, "Complete26-command history")
    names = [v["command"]["tuple"][0] for v in values]
    require(
        names
        == [
            "get_spaces",
            "reset",
            "has_attr",
            "has_attr",
            "get_attr",
            "set_attr",
            "get_attr",
            "env_method",
            "is_wrapped",
            "is_wrapped",
            "render",
            "step",
        ]
        + ["reset", "step", "step", "step"] * 3
        + ["reset", "render"],
        "Command schedule",
    )
    require(
        values[2]["response"] is True
        and values[3]["response"] is False
        and values[4]["response"] == 3
        and values[6]["response"] == 17,
        "Attribute/has_attr semantics",
    )
    require(
        values[8]["response"] is True and values[9]["response"] is False,
        "Wrapper search semantics",
    )
    for offset, mode in ((12, "terminated"), (16, "truncated"), (20, "both")):
        normal, terminal, following = [
            values[offset + j]["response"]["tuple"] for j in (1, 2, 3)
        ]
        require(
            normal[2] is False and terminal[2] is True and following[2] is False,
            "Termination/autoreset",
        )
        info, next_info = encoded_dict(terminal[3]), encoded_dict(following[3])
        require(
            info["TimeLimit.truncated"] is (mode == "truncated")
            and "terminal_observation" in info
            and "terminal_observation" not in next_info,
            "Truncation/terminal observation",
        )
        require(terminal[4] == following[4], "Reset info not retained")
        observation = encoded_dict(info["terminal_observation"])["state"]["array"]
        require(
            observation["shape"] == [2]
            and observation["dtype"] == "<f4"
            and struct.unpack("<2f", bytes.fromhex(observation["hex"]))[0] == 2,
            "Terminal observation before reset",
        )
    # Every recorded before/after state carries explicit dedicated RNG and state;
    # source/peer comparison checks their entire typed JSON, including signed zero.
    for value in values:
        for label in ("before", "after"):
            state = encoded_dict(value[label])
            require(
                set(state) == {"tick", "resets", "mode", "rng", "value", "observation"},
                "Missing causal fake state",
            )
            require("state" in encoded_dict(state["rng"]), "Missing RNG state")
    return {"commands": 26, "known_oracle_cases": 3, "full_state_rng_retained": True}


def check_modules(folder, value, root, profile):
    proc_proof(value["proc"], root)
    module_path = Path(value["modules_path"])
    require(
        module_path.resolve().is_relative_to(folder.resolve()), "Foreign module receipt"
    )
    modules = read(module_path)
    require(
        modules == sorted(set(modules)) and digest(modules) == value["modules_sha256"],
        "Full module-list digest",
    )
    present = {
        name: any(s == name or s.startswith(name + ".") for s in modules)
        for name in ("torch", "stable_baselines3", "mujoco")
    }
    require(present == value["imports"], "Import flags differ from full list")
    expected = {
        "torch": profile == "sb3",
        "stable_baselines3": profile == "sb3",
        "mujoco": False,
    }
    require(
        present == expected
        and ("libtorch" in value["proc"]["heavy_mappings"]) is (profile == "sb3")
        and "mujoco" not in value["proc"]["heavy_mappings"],
        "Actual child import isolation",
    )
    status = next(
        d for name, d in value["proc"]["proc_evidence"].items() if name == "status.txt"
    )
    text = Path(status["path"]).read_text()
    rss = (
        int(next(x.split()[1] for x in text.splitlines() if x.startswith("VmRSS:")))
        * 1024
    )
    require(rss == value["rss_bytes"] > 0, "Actual live RSS binding")
    return rss


def audit_probe(folder, packet, root, watch, complete):
    result_path = folder / "probe-result.json"
    value = {"result_present": result_path.exists()}
    if (folder / "semantic.json").exists():
        semantic = read(folder / "semantic.json")
        value["semantic"] = check_semantics(semantic)
        value["semantic_sha256"] = digest(semantic)
    if not result_path.exists():
        require(not complete, "Completed row missing result")
        return value
    result = read(result_path)
    context = result["context"]
    owner = result["worker"]
    require(
        all(result[k] == 0 for k in ("physics", "actors", "inference", "updates")),
        "Unexpected work",
    )
    require(
        context["method"] == "forkserver" and context["preload"] == [],
        "Wrong context/preload",
    )
    for key in ("root_before", "server_before", "server_after"):
        proc_proof(context[key], root, True)
    require(
        context["server_before"]["parent_pid"] == context["root"]["pid"],
        "Server ancestry",
    )
    before, after = context["worker_before"], context["worker_after"]
    for proof in (before, after):
        require(
            proof["worker"] == owner
            and proof["proc"]["identity"] == owner
            and proof["proc"]["parent_pid"] == context["server"]["pid"],
            "Child ancestry/identity",
        )
    first = check_modules(folder, before, root, result["profile"])
    last = check_modules(folder, after, root, result["profile"])
    require(
        first == result["worker_rss_before"] and last == result["worker_rss_after"],
        "RSS result binding",
    )
    semantic = read(folder / "semantic.json")
    closed = read(folder / "closed-state.json")
    require(
        digest(semantic) == result["semantic_sha256"]
        and closed == semantic[-1]["after"]
        and digest(closed) == result["closed_state_sha256"],
        "Whole semantic/close state digest",
    )
    close = rows(folder / "serial-close.jsonl")
    final = rows(folder / "context-finalizer.jsonl")
    require(
        [(r["phase"], r.get("worker")) for r in close]
        == [("bound", None), ("close_send", 0), ("joined", 0), ("closed", None)],
        "One-worker serial close phases",
    )
    require(
        len(final) == 1
        and final[0]["identity"] == owner
        and final[0]["phase"] == "managed_finalizer",
        "Managed finalizer",
    )
    duration = close[2]["monotonic_s"] - close[1]["monotonic_s"]
    require(
        0 <= duration == result["close_to_join_s"] <= 2
        and close[1]["monotonic_s"]
        <= final[0]["monotonic_s"]
        <= close[2]["monotonic_s"]
        and close[2]["exitcode"] == 0,
        "Actual close/finalizer/join deadline",
    )
    if complete:
        retirement = watch["retirement_records"].get(
            str(owner["pid"]) + ":" + owner["start"]
        )
        require(
            retirement
            and retirement["released"]
            and retirement["highwater_charge_bytes"] > 0
            and retirement["max_hold_s"] <= 0.1,
            "Actual positive retirement proof missing",
        )
    value.update(
        profile=result["profile"],
        worker=owner,
        rss_before=first,
        rss_after=last,
        closed_state_sha256=result["closed_state_sha256"],
        close_to_join_s=duration,
    )
    return value


def audit_comparisons(out):
    values = {
        name: read(out / "fixtures" / name / "semantic.json")
        for name in ("sb3A", "leanA", "sb3B", "leanB")
    }
    pairs = []
    for label in ("A", "B"):
        a, b = "sb3" + label, "lean" + label
        # Canonical JSON hashes independently preserve numeric/bool types.
        require(
            digest(values[a]) == digest(values[b]),
            "Independent full fake pair mismatch",
        )
        left, right = [read(out / "fixtures" / n / "probe-result.json") for n in (a, b)]
        require(
            left["closed_state_sha256"] == right["closed_state_sha256"],
            "Paired close state",
        )
        ratio = max(right["worker_rss_before"], right["worker_rss_after"]) / min(
            left["worker_rss_before"], left["worker_rss_after"]
        )
        require(0 < ratio <= 0.75, "Independent perworkerRSS gate")
        pairs.append({"pair": label, "commands": 26, "rss_ratio_upper_bound": ratio})
    return pairs


def main():
    parser = argparse.ArgumentParser()
    for key in ("packet", "folder", "source-sha"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--ready-file")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    p = read(args.packet)
    out = Path(args.folder).resolve()
    began = time.monotonic()
    if args.ready_file:
        barrier = Path(args.ready_file).resolve()
        require(barrier.parent == out, "Owned audit readiness barrier")
        end = time.monotonic() + 2
        while not barrier.exists():
            require(time.monotonic() < end, "Audit controller readiness timeout")
            time.sleep(0.005)
    r = json.loads(gunzip((out / "result.json.gz").read_bytes(), 512 * 1024))
    require(
        out == (root / p["output"]).resolve()
        and r["source_commit"] == args.source_sha
        and r["protocol_sha256"] == digest(p),
        "Frozen source/output/protocol",
    )
    require(
        r["status"] in ("completed", "halted", "deferred")
        and r["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "Writer/boot state",
    )
    require(
        0 <= began - r["started_monotonic_s"] < p["whole_wall_s"],
        "Original audit epoch",
    )
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        == args.source_sha
        and not subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"], cwd=root, text=True
        ).strip(),
        "Source changed",
    )
    require(
        subprocess.check_output(
            ["git", "rev-parse", "--is-shallow-repository"], cwd=root, text=True
        ).strip()
        == "false",
        "Full history",
    )
    for name, d in {
        **p["source_files"],
        **p["input_files"],
        **p["runtime_file_hashes"],
    }.items():
        path = Path(name) if name.startswith("/") else root / name
        raw = path.read_bytes()
        require(
            len(raw) == d["bytes"] and hashlib.sha256(raw).hexdigest() == d["sha256"],
            "Source/runtime/input hash:" + name,
        )
    for obj, h in p["required_git_objects"].items():
        require(
            hashlib.sha256(
                subprocess.check_output(["git", "show", obj], cwd=root)
            ).hexdigest()
            == h,
            "Historical object",
        )
    for name, commit in p["upstream_commits"].items():
        require(
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root / "third_party" / name, text=True
            ).strip()
            == commit,
            "Vendor pin",
        )
    boot = verify_wire(root, out, p, r, prefix="lean-worker-probe")
    policy = p["admission_policy"]
    ad = boot["admission"]
    require(
        ad["status"] == "admitted" and not ad["reasons"] and ad["policy"] == policy,
        "Initial host admission",
    )
    first, second = ad["samples"]
    verify_admission(first, policy, first["monotonic_s"])
    verify_admission(second, policy, ad["assessed_monotonic_s"])
    require(
        second["monotonic_s"] - first["monotonic_s"] >= policy["spacing_s"],
        "Admission spacing",
    )
    verify_admission(
        boot["post_checkout_admission"],
        policy,
        boot["post_checkout_assessed_monotonic_s"],
    )
    if not r.get("runtime_admission", {}).get("reasons") and "runtime_admission" in r:
        ad = r["runtime_admission"]
        verify_admission(ad["sample"], policy, ad["assessed_monotonic_s"])
    require(
        [s["id"] for s in r["rows"]] == [s["id"] for s in p["slots"]],
        "Whole row denominator",
    )
    require(
        not (out / "fixtures/native-credit.json").exists(),
        "Zero-native probe unexpectedly allocated physics",
    )
    xml_path = out / "controls.xml.gz"
    cases = []
    if xml_path.exists():
        raw = gunzip(xml_path.read_bytes(), 1024 * 1024)
        meta = r["controls_encoding"]
        require(
            hashlib.sha256(raw).hexdigest() == meta["decoded_sha256"]
            and hashlib.sha256(xml_path.read_bytes()).hexdigest()
            == meta["encoded_sha256"],
            "JUnit bytes",
        )
        cases = list(ET.fromstring(raw).iter("testcase"))
        expected = {
            (
                n.split("::")[0][:-3].replace("/", ".")
                + "".join("." + x for x in n.split("::")[1:-1]),
                n.split("::")[-1],
            )
            for n in p["test_node_ids"]
        }
        require(
            len(cases) == len(expected)
            and {(c.get("classname"), c.get("name")) for c in cases} == expected
            and all(not list(c) for c in cases),
            "Complete source controls",
        )
    elif r["status"] == "completed":
        raise ValueError("Source gate missing")
    ids = [r["supervisor"], *r["owned_identities"]]
    proofs = []
    for slot in r["rows"]:
        wfile = out / (slot["id"] + "-watchdog.json")
        watch = read(wfile) if wfile.exists() else None
        if watch:
            ids.extend(watch["owned_identities"])
            require(
                not watch["final_live"]
                and watch["peak_capacity_charge_bytes"] <= p["owned_rss_bytes"],
                "Guard cleanup/capacity",
            )
            if slot["status"] == "completed":
                require(
                    watch["status"] == "exited"
                    and not watch["pending_retirements"]
                    and max(watch["max_probe_s"], watch["max_sample_gap_s"]) <= 0.1
                    and watch["limits"]["poll_s"] == 0.001,
                    "Guard exact timing/sampling profile",
                )
            for rec in watch["retirement_records"].values():
                e = rec["evidence"]
                require(
                    e["tasks"] == e["after_tasks"] == [str(rec["pid"])]
                    and e["statm_pages"] == [0] * 7
                    and e["vmrss_absent"]
                    and e["vmsize_absent"]
                    and e["status_threads"] == 1,
                    "Raw terminal-mm evidence",
                )
                require(
                    all(
                        v["start_ticks"] == rec["start_ticks"] and v["flags"] & 4
                        for v in (e["before"], e["after"], e["task"])
                    ),
                    "Terminal identity/flags",
                )
            publication = read(out / (slot["id"] + "-watchdog-publication.json"))
            require(
                publication["status"] == "copied_and_removed"
                and not Path(publication["directory"]).exists()
                and not publication["task_live"]
                and not publication["guard_live"],
                "Exact after-exit publication",
            )
            for rec in publication["copies"]:
                raw = Path(rec["target"]).read_bytes()
                require(
                    len(raw) == rec["bytes"]
                    and hashlib.sha256(raw).hexdigest() == rec["sha256"],
                    "Exact canonical copy",
                )
        folder = out / "fixtures" / slot["id"]
        if folder.exists():
            proofs.append(
                {
                    "id": slot["id"],
                    "formal_status": slot["status"],
                    **audit_probe(
                        folder, p, root, watch, slot["status"] == "completed"
                    ),
                }
            )
    for ident in ids:
        path = Path("/proc", str(ident["pid"]), "stat")
        if path.exists():
            require(
                int(path.read_text().rsplit(")", 1)[1].split()[19])
                != ident["start_ticks"],
                "Owned writer still exists",
            )
    pairs = audit_comparisons(out) if r["status"] == "completed" else []
    fixture = receipts = 0
    manifest = hashlib.sha256()
    count = 0
    for path in sorted(out.parent.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            count += 1
            if path.is_relative_to(out / "fixtures"):
                fixture += len(raw)
            else:
                receipts += len(raw)
            manifest.update(
                json.dumps(
                    [
                        str(path.relative_to(root)),
                        len(raw),
                        hashlib.sha256(raw).hexdigest(),
                        str(path.readlink()) if path.is_symlink() else None,
                    ]
                ).encode()
            )
    mirror = root / "artifacts/generated/h28-receipt-inputs"
    marker = read(mirror / "INPUT_MIRROR_NOT_EXECUTION.json")
    require(marker == p["input_mirror"]["marker"], "Explicit input mirror provenance")
    mirrored = sum(f.stat().st_size for f in mirror.rglob("*") if f.is_file())
    require(mirrored == p["mirrored_input_bytes"], "Input mirror byte accounting")
    for alias in p["input_mirror"]["aliases"]:
        link = Path("/home/ubuntu/joxing") / alias
        require(
            link.is_symlink() and link.resolve() == (mirror / alias).resolve(),
            "Read-only input alias identity",
        )
    fixture += mirrored
    result = {
        "status": "verified_" + r["status"],
        "source_commit": args.source_sha,
        "protocol_sha256": r["protocol_sha256"],
        "source_files": len(p["source_files"]),
        "controls": len(cases),
        "rows": r["rows"],
        "probes": proofs,
        "pairs": pairs,
        "physics": 0,
        "native_environments": 0,
        "actors": 0,
        "updates": 0,
        "source_identities_exited": len({(i["pid"], i["start_ticks"]) for i in ids}),
        "fixture_bytes": fixture,
        "included_mirrored_input_bytes": mirrored,
        "receipts_before_audit": receipts,
        "raw_files_before_audit": count,
        "raw_manifest_sha256": manifest.hexdigest(),
        "original_epoch_through_audit_s": time.monotonic() - r["started_monotonic_s"],
        "audit_wall_s": time.monotonic() - began,
        "audit_identity": {
            "pid": os.getpid(),
            "start_ticks": int(
                Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
            ),
        },
        "audit_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "runtime": check_runtime(out, p, r)
        if any(s["stage"] == "runtime" for s in r["stages"])
        else {"status": "not_run"},
    }
    data = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination = out / "independent-audit.json"
    require(not destination.exists(), "Fresh audit output")
    require(
        fixture <= p["fixture_bytes"]
        and receipts + len(data) <= p["receipt_bytes"]
        and result["audit_peak_rss_bytes"] <= p["owned_rss_bytes"],
        "Audited output/RSS cap",
    )
    require(
        time.monotonic() - began < 30
        and time.monotonic() - r["started_monotonic_s"] < 300,
        "Original whole/audit deadline",
    )
    destination.write_bytes(data)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "controls",
                    "pairs",
                    "source_identities_exited",
                    "original_epoch_through_audit_s",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
