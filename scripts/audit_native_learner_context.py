"""Independent saved-evidence H27 audit; never re-simulates native physics."""

import argparse
import base64
import hashlib
import json
import math
import resource
import subprocess
import time
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from itertools import pairwise
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def typed(value):
    if type(value) is float:
        return ("float", value.hex())
    if isinstance(value, dict):
        return ("dict", sorted((k, typed(v)) for k, v in value.items()))
    if isinstance(value, list):
        return ("list", [typed(v) for v in value])
    return (type(value).__name__, value)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def gunzip(blob, limit):
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(blob, limit + 1)
    require(
        len(raw) <= limit
        and decoder.eof
        and not decoder.unused_data
        and not decoder.unconsumed_tail,
        "Corrupt/truncated/oversized/noncanonical gzip stream",
    )
    return raw


def check_credit(path, maximum, completed):
    credit = read(path)
    log = Path(path).with_suffix(".jsonl").read_text().splitlines()
    used, pending, settled, chain = 0, {}, {}, "0" * 64
    for number, line in enumerate(log[: credit["sequence"]], 1):
        e = json.loads(line)
        require(e["sequence"] == number, "Credit sequence")
        chain = hashlib.sha256((chain + line).encode()).hexdigest()
        token = e["token"]
        if e["operation"] == "reserve":
            require(token not in pending and e["amount"] > 0, "Credit reservation")
            pending[token] = {
                "owner": e["owner"],
                "reserved": e["amount"],
                "slot": e["slot"],
                "kind": e["kind"],
                "state": "reserved",
            }
        elif e["operation"] == "commit":
            prior = pending[token]
            require(
                prior["owner"] == e["owner"]
                and prior["state"] == "reserved"
                and 0 <= e["actual"] <= prior["reserved"],
                "Credit commit",
            )
            used += e["actual"]
            aggregate = settled.setdefault(
                prior["slot"] + ":" + prior["kind"], {"operations": 0, "used": 0}
            )
            aggregate["operations"] += 1
            aggregate["used"] += e["actual"]
            del pending[token]
        else:
            require(
                e["operation"] == "unknown" and pending[token]["owner"] == e["owner"],
                "Credit unknown work",
            )
            pending[token].update(state="unknown", error=e["error"])
        require(
            used + sum(p["reserved"] for p in pending.values()) <= maximum,
            "Physics credit exceeded",
        )
    require(
        credit["limit"] == maximum
        and used == credit["used"]
        and pending == credit["pending"]
        and settled == credit["settled"]
        and chain == credit["chain"],
        "Credit state/log mismatch",
    )
    if completed:
        require(
            not pending and len(log) == credit["sequence"],
            "Incomplete successful physical credit",
        )
    return credit | {"charged": used + sum(r["reserved"] for r in pending.values())}


def verify_wire(root, out, packet, report, prefix="native-learner-context"):
    blob = (out / "result.json.gz").read_bytes()
    raw = gunzip(blob, 512 * 1024)
    require(
        len(raw) <= 512 * 1024
        and raw
        == (
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode(),
        "Canonical main receipt",
    )
    terminal = read(root / f"runs/v0.0.11/{prefix}-launch.log")
    encoded = base64.b64decode(terminal["data_base64"], validate=True)
    decoded = gunzip(encoded, 512 * 1024)
    require(
        terminal["schema"] == "joxing-terminal-capsule-v1"
        and terminal["encoding"] == packet["receipt_wire_encoding"],
        "Terminal encoding",
    )
    require(
        len(encoded) == terminal["encoded_bytes"]
        and hashlib.sha256(encoded).hexdigest() == terminal["encoded_sha256"]
        and len(decoded) == terminal["decoded_bytes"] <= 512 * 1024
        and hashlib.sha256(decoded).hexdigest() == terminal["decoded_sha256"],
        "Entire terminal bytes",
    )
    values = {
        k: report.get(k)
        for k in (
            "status",
            "error",
            "rows",
            "physics_credit",
            "wall_s",
            "final_live_recorded",
        )
    }
    require(
        typed(json.loads(decoded)) == typed(values),
        "Terminal fields/type/float mismatch",
    )
    require(
        terminal["main_report"]
        == {
            "encoding": packet["receipt_wire_encoding"],
            "decoded_bytes": len(raw),
            "decoded_sha256": hashlib.sha256(raw).hexdigest(),
            "encoded_bytes": len(blob),
            "encoded_sha256": hashlib.sha256(blob).hexdigest(),
        },
        "Final main byte binding",
    )
    bootstrap_blob = (root / f"runs/v0.0.11/{prefix}-deployment.json.gz").read_bytes()
    bootstrap_raw = gunzip(bootstrap_blob, 1048576)
    meta = read(root / f"runs/v0.0.11/{prefix}-deployment.encoding.json")
    require(
        len(bootstrap_blob) == meta["encoded_bytes"]
        and hashlib.sha256(bootstrap_blob).hexdigest() == meta["encoded_sha256"]
        and len(bootstrap_raw) == meta["decoded_bytes"] <= 1048576
        and hashlib.sha256(bootstrap_raw).hexdigest() == meta["decoded_sha256"],
        "Bootstrap exact bytes",
    )
    bootstrap = json.loads(bootstrap_raw)
    require(
        bootstrap["source_commit"] == report["source_commit"]
        and bootstrap["protocol_sha256"] == report["protocol_sha256"]
        and bootstrap["epoch_started_monotonic_s"] == report["started_monotonic_s"],
        "Original epoch/bootstrap binding",
    )
    return bootstrap


def verify_admission(sample, policy, now):
    require(0 <= now - sample["monotonic_s"] <= policy["fresh_s"], "Stale admission")
    require(
        0
        <= sample["load"][0]
        <= policy["load_per_affinity_cpu"] * sample["affinity_cpus"]
        and not sample["jobs"],
        "Host admission load/jobs",
    )
    for name, key, limit in (
        ("cpu", "some", "cpu_some_avg10"),
        ("io", "full", "io_full_avg10"),
        ("memory", "full", "memory_full_avg10"),
    ):
        value = sample["psi"][name][key]["avg10"]
        require(
            math.isfinite(value) and 0 <= value <= policy[limit],
            "Host pressure:" + name,
        )
    require(
        sample["available_memory_bytes"] >= policy["minimum_memory_bytes"]
        and sample["disk_free_bytes"] >= policy["minimum_disk_bytes"],
        "Host floor",
    )


def check_runtime(out, packet, report):
    stage = next(s for s in report["stages"] if s["stage"] == "runtime")
    if stage["status"] != "completed":
        return {"status": "incomplete"}
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress((out / "runtime.log").read_bytes(), 65537)
    require(
        len(raw) <= 65536
        and decoder.eof
        and not decoder.unused_data
        and not decoder.unconsumed_tail,
        "Runtime event prefix/finalization",
    )
    events = [json.loads(x) for x in raw.splitlines()]
    require(
        [e["phase"] for e in events]
        == ["start"]
        + ["import_begin", "import_end"] * 4
        + ["version_assert", "qualified"],
        "Runtime phase order",
    )
    require(
        [events[1 + 2 * i]["module"] for i in range(4)]
        == ["numpy", "mujoco", "torch", "stable_baselines3"],
        "Runtime module order",
    )
    require(
        events[-2]["versions"]
        == events[-2]["expected"]
        == events[-1]["versions"]
        == packet["runtime_versions"]
        and stage["wall_s"] <= 13,
        "Runtime version/deadline",
    )
    require(
        all(
            e["observation"]["pid"] == stage["process"]["pid"]
            and e["observation"]["start_ticks"] == stage["process"]["start_ticks"]
            for e in events
        ),
        "Runtime process identity",
    )
    return {"status": "qualified", "events": len(events), "wall_s": stage["wall_s"]}


def actor_hash(path):
    # Own source-frozen SB3 output, not an external/untrusted pickle or Actor.load.
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        require(
            array.dtype.kind == "f" and bool(tensor.isfinite().all()),
            "Nonfinite actor tensor",
        )
        result.update(json.dumps([name, list(array.shape), str(array.dtype)]).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def check_trials(folder, packet, hash_actor=actor_hash):
    """Reconstruct semantic comparisons independently of compare_slots()."""
    folder = Path(folder)
    streams = {
        "learner",
        "native-evaluation",
        *[f"{prefix}-{i}" for prefix in ("worker", "native-world") for i in range(4)],
    }
    slots = {}
    for slot in packet["slots"]:
        path = folder / slot["id"]
        semantic = read(path / "semantic-result.json")
        require(
            semantic["training_transitions"] == 64 and semantic["updates"] == 12,
            "Missing actual64/12 learner schedule",
        )
        require(
            semantic["scheduled_training_complete"] and semantic["source_unchanged"],
            "Incomplete or changed learner",
        )
        require(
            semantic["initial_actor_hash"] != semantic["final_actor_hash"],
            "Unchanged actor",
        )
        require(
            semantic["specification"] == packet["expected_specification"],
            "Changed native specification",
        )
        for prefix in ("initial", "final"):
            require(
                hash_actor(path / "trial" / (prefix + "-actor.pt"))
                == semantic[prefix + "_actor_hash"],
                "Saved actor tensor hash differs",
            )
        actual = {p.stem for p in (path / "traces").glob("*.jsonl")}
        require(actual == streams, "Missing/foreign semantic stream")
        traces = {
            name: [r["semantic"] for r in rows(path / "traces" / (name + ".jsonl"))]
            for name in streams
        }
        require(all(traces.values()), "Empty semantic stream")
        learner = traces["learner"]
        require(
            Counter(r["phase"] for r in learner)
            == {
                "model_initial": 1,
                "learn_before": 1,
                "learn_after": 1,
                "update_before": 12,
                "update_after": 12,
                "evaluation_before": 2,
                "evaluation_after": 2,
            },
            "Incomplete learner history",
        )
        require(
            [r["updates"] for r in learner if r["phase"] == "update_after"]
            == list(range(1, 13)),
            "Wrong update order",
        )
        require(
            [r["updates"] for r in learner if r["phase"] == "update_before"]
            == list(range(12)),
            "Wrong pre-update order",
        )
        require(
            len(next(r["replay"] for r in learner if r["phase"] == "learn_after"))
            == 64,
            "Missing occupied replay digest",
        )
        for i in range(4):
            t, native = traces[f"worker-{i}"], traces[f"native-world-{i}"]
            require(
                sum(r["phase"] == "step" for r in t) == 16
                and t[-1]["phase"] == "close",
                "Incomplete worker history",
            )
            require(
                sum(r["phase"] == "step" for r in native) == 16
                and sum(r["phase"] == "reset" for r in native) >= 2,
                "Incomplete native history",
            )
        evaluation = traces["native-evaluation"]
        require(
            sum(r["phase"] == "reset" for r in evaluation) == 3
            and sum(r["phase"] == "step" for r in evaluation) >= 2,
            "Incomplete evaluation history",
        )
        slots[slot["id"]] = {"semantic": semantic, "traces": traces}
    comparisons = []
    for arm in ("A", "B"):
        left, right = slots["legacy" + arm], slots["candidate" + arm]
        require(
            typed(left["semantic"]) == typed(right["semantic"]),
            "Independent semantic mismatch:" + arm,
        )
        for name in sorted(streams):
            require(
                typed(left["traces"][name]) == typed(right["traces"][name]),
                "Independent full trace mismatch:" + arm + ":" + name,
            )
            comparisons.append(
                {
                    "arm": arm,
                    "stream": name,
                    "rows": len(left["traces"][name]),
                    "exact": True,
                }
            )
    return {
        "qualified": True,
        "comparisons": comparisons,
        "transitions": 256,
        "updates": 48,
        "actor_payloads_read": 8,
    }


def proc_proof(value, root, clean=False):
    captures = {}
    for name, record in value["proc_evidence"].items():
        path = Path(record["path"])
        require(path.resolve().is_relative_to(root.resolve()), "Foreign proc evidence")
        raw = path.read_bytes()
        require(
            len(raw) == record["bytes"]
            and hashlib.sha256(raw).hexdigest() == record["sha256"],
            "Proc evidence hash",
        )
        captures[name] = raw.decode()
    require(
        set(captures)
        == {"stat-before.txt", "stat-after.txt", "status.txt", "maps.txt"},
        "Proc evidence denominator",
    )
    identity = value["identity"]
    stat = [
        captures[k].rsplit(")", 1)[1].split()
        for k in ("stat-before.txt", "stat-after.txt")
    ]
    require(
        all(
            s[19] == identity["start"]
            and s[0] not in ("Z", "X")
            and not int(s[6]) & 4
            and int(s[17]) == 1
            for s in stat
        ),
        "Proc identity/thread/exit mismatch",
    )
    require(
        all(
            int(captures[k].split("(", 1)[0]) == identity["pid"]
            for k in ("stat-before.txt", "stat-after.txt")
        ),
        "Proc PID mismatch",
    )
    status = dict(
        line.split(":", 1)
        for line in captures["status.txt"].splitlines()
        if ":" in line
    )
    require(int(status["Threads"].strip()) == 1, "Proc multithreaded")
    require(value["parent_pid"] == int(stat[-1][1]), "Proc ancestry mismatch")
    heavy = [
        k
        for k in ("libtorch", "numpy", "mujoco", "openblas", "mkl")
        if k in captures["maps.txt"].lower()
    ]
    require(
        heavy == value["heavy_mappings"]
        and hashlib.sha256(captures["maps.txt"].encode()).hexdigest()
        == value["maps_sha256"],
        "Proc mappings mismatch",
    )
    if clean:
        require(not heavy, "Dirty parent/server")


def check_context(folder, packet, watch, root):
    """Read dynamic state/lifetime/ancestry without importing training_context."""
    before, final = (
        read(folder / "trained-before-close.json"),
        read(folder / "training-context-result.json"),
    )
    context = final["context"]
    cold = read(folder / "training-constructor-states.json")
    require(
        len(cold) == 4
        and all(
            s["muscles"] == 700 and s["state"] == packet["expected_state_hash"]
            for s in cold
        ),
        "Original cold native state changed",
    )
    require(
        len(before["states"]) == 4
        and len({(w["pid"], w["start"]) for w in before["workers"]}) == 4,
        "Dynamic worker denominator",
    )
    require(
        final["execution_profile"] == packet["execution_profile"]
        and len(final["worlds"]) == len(context["workers"]) == 4,
        "Trained context denominator/profile",
    )
    request = read(folder / "requested-context.json")
    require(
        request["requested"] == "spawn" and request["actual"] == "forkserver",
        "Undeclared context substitution",
    )
    require(
        context["method"] == "forkserver" and context["preload"] == [],
        "Context/preload",
    )
    for key in ("root_before", "server_before", "server_after"):
        proc_proof(context[key], root, True)
    require(
        context["server_before"]["parent_pid"] == context["root"]["pid"],
        "Server parent",
    )
    require(
        before["workers"] == [r["worker"] for r in context["workers"]],
        "Before-close worker drift",
    )
    close = rows(folder / "serial-close.jsonl")
    require(
        [(r["phase"], r.get("worker")) for r in close]
        == [("bound", None)]
        + [(p, i) for i in range(4) for p in ("close_send", "joined")]
        + [("closed", None)],
        "Serial close order",
    )
    duration = close[-2]["monotonic_s"] - close[1]["monotonic_s"]
    require(
        0 <= duration == final["close_to_join_s"] <= 2
        and all(a["monotonic_s"] <= b["monotonic_s"] for a, b in pairwise(close)),
        "Serial deadline/order",
    )
    retirement = []
    for i, value in enumerate(context["workers"]):
        owner, state = value["worker"], before["states"][i]
        proc_proof(value["before"]["proc"], root)
        proc_proof(value["after"], root)
        require(
            value["before"]["worker"] == value["after"]["identity"] == owner
            and value["after"]["parent_pid"] == context["server"]["pid"],
            "Actual worker ancestry",
        )
        require(
            read(folder / f"world-{i}/training-factory.json")
            == {
                "module": "joxing.training_context",
                "profile": packet["execution_profile"],
                "training_allowed": True,
            },
            "Fresh training-capable factory missing",
        )
        phases, finals = (
            rows(folder / f"world-{i}/resource-release.jsonl"),
            rows(folder / f"world-{i}/context-finalizer.jsonl"),
        )
        trace = rows(folder / f"traces/worker-{i}.jsonl")
        last = trace[-1]
        require(
            last["owner"] == owner
            and last["semantic"]["phase"] == "close"
            and last["semantic"]["state"] == state["state"]
            and last["semantic"]["reset_rng"] == state["reset_rng"],
            "Trained close/RNG mismatch",
        )
        require(
            [r["phase"] for r in phases]
            == ["before_close", "inner_closed", "graph_checked", "released"]
            and all(r["identity"] == owner for r in phases),
            "Release phases/owners",
        )
        require(
            phases[0]["state_hash"] == phases[1]["state_hash"] == state["state"]
            and phases[0]["muscles"] == state["muscles"] == 700,
            "Trained native state changed",
        )
        require(
            phases[2]["weakrefs_dead"] == {"env": True, "model": True, "data": True},
            "Trained graph retained",
        )
        for record in (phases[0], phases[2]):
            memory = record["memory"]
            require(
                type(memory["rss_bytes"]) is int and memory["rss_bytes"] > 0,
                "Unobserved native RSS",
            )
            require(
                all(
                    z["pid"] == owner["pid"]
                    and z["start"] == owner["start"]
                    and z["state"] not in ("Z", "X")
                    and not z["flags"] & 4
                    for z in (memory["before"], memory["after"])
                ),
                "RSS process binding",
            )
        require(
            phases[2]["observed_rss_change_bytes"]
            == phases[2]["memory"]["rss_bytes"] - phases[0]["memory"]["rss_bytes"],
            "RSS delta",
        )
        require(
            len(finals) == 1
            and finals[0]["phase"] == "managed_finalizer"
            and finals[0]["identity"] == owner,
            "Managed finalizer",
        )
        sent, joined = close[1 + 2 * i : 3 + 2 * i]
        require(
            all(
                e["identity"]["pid"] == owner["pid"]
                and e["identity"]["start"] == owner["start"]
                for e in (sent, joined)
            ),
            "Joined process identity",
        )
        times = [
            sent["monotonic_s"],
            phases[0]["monotonic_s"],
            last["monotonic_s"],
            *[r["monotonic_s"] for r in phases[1:]],
            finals[0]["monotonic_s"],
            joined["monotonic_s"],
        ]
        require(
            all(a <= b for a, b in pairwise(times)) and joined["exitcode"] == 0,
            "Dynamic release/finalizer/join order",
        )
        record = watch["retirement_records"].get(f"{owner['pid']}:{owner['start']}")
        require(
            record
            and record["released"]
            and record["highwater_charge_bytes"] > 0
            and record["max_hold_s"] <= 0.1,
            "Actual positive trained retirement missing",
        )
        retirement.append({"worker": owner, "max_hold_s": record["max_hold_s"]})
    return {
        "workers": before["workers"],
        "close_to_join_s": duration,
        "retirements": retirement,
    }


def main():
    parser = argparse.ArgumentParser()
    for key in ("packet", "folder", "source-sha"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.folder).resolve()
    packet = read(args.packet)
    report = json.loads(gunzip((out / "result.json.gz").read_bytes(), 512 * 1024))
    began = time.monotonic()
    require(out == (root / packet["output"]).resolve(), "Foreign output root")
    require(
        report["boot_id"]
        == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "Host boot changed",
    )
    require(
        0 <= began - report["started_monotonic_s"] <= packet["whole_wall_s"],
        "Original epoch audit deadline",
    )
    require(
        report["status"] in ("completed", "halted", "deferred"), "Task still active"
    )
    require(
        report["source_commit"] == args.source_sha
        and report["protocol_sha256"] == digest(packet),
        "Source/protocol identity",
    )
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        == args.source_sha
        and not subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"], cwd=root, text=True
        ).strip(),
        "Changed source checkout",
    )
    for name, expected in packet["source_files"].items():
        raw = (root / name).read_bytes()
        require(
            len(raw) == expected["bytes"]
            and hashlib.sha256(raw).hexdigest() == expected["sha256"],
            "Source file:" + name,
        )
    for obj, expected in packet["required_git_objects"].items():
        require(
            hashlib.sha256(
                subprocess.check_output(["git", "show", obj], cwd=root)
            ).hexdigest()
            == expected,
            "Historical object",
        )
    for name, expected in packet.get("input_files", {}).items():
        path = Path(name) if name.startswith("/") else root / name
        raw = path.read_bytes()
        require(
            len(raw) == expected["bytes"]
            and hashlib.sha256(raw).hexdigest() == expected["sha256"],
            "Input hash:" + name,
        )
    for name, expected in packet["upstream_commits"].items():
        require(
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root / "third_party" / name, text=True
            ).strip()
            == expected,
            "Upstream pin",
        )
    require(
        subprocess.check_output(
            ["git", "rev-parse", "--is-shallow-repository"], cwd=root, text=True
        ).strip()
        == "false",
        "Incomplete Git history",
    )
    bootstrap = verify_wire(root, out, packet, report)
    admission = bootstrap["admission"]
    policy = packet["admission_policy"]
    require(
        admission["status"] == "admitted"
        and not admission["reasons"]
        and admission["policy"] == policy,
        "Initial admission",
    )
    first, second = admission["samples"]
    verify_admission(first, policy, first["monotonic_s"])
    verify_admission(second, policy, admission["assessed_monotonic_s"])
    require(
        second["monotonic_s"] - first["monotonic_s"] >= policy["spacing_s"],
        "Admission spacing",
    )
    verify_admission(
        bootstrap["post_checkout_admission"],
        policy,
        bootstrap["post_checkout_assessed_monotonic_s"],
    )
    if "runtime_admission" in report:
        a = report["runtime_admission"]
        require(not a["reasons"], "Runtime admission")
        verify_admission(a["sample"], policy, a["assessed_monotonic_s"])
    require(
        [r["id"] for r in report["rows"]] == [s["id"] for s in packet["slots"]],
        "Missing/reordered outcome row",
    )
    cases = []
    if (out / "controls.xml.gz").exists():
        encoded = (out / "controls.xml.gz").read_bytes()
        xml = gunzip(encoded, 1024 * 1024)
        meta = report["controls_encoding"]
        require(
            len(encoded) == meta["encoded_bytes"]
            and hashlib.sha256(encoded).hexdigest() == meta["encoded_sha256"]
            and len(xml) == meta["decoded_bytes"]
            and hashlib.sha256(xml).hexdigest() == meta["decoded_sha256"],
            "Control wire hashes",
        )
        cases = list(ET.fromstring(xml).iter("testcase"))
        mapping = {
            (
                n.split("::")[0][:-3].replace("/", ".")
                + "".join("." + x for x in n.split("::")[1:-1]),
                n.split("::")[-1],
            )
            for n in packet["test_node_ids"]
        }
        require(
            len(cases) == len(mapping)
            and {(c.get("classname"), c.get("name")) for c in cases} == mapping
            and all(not list(c) for c in cases),
            "Incomplete/false control gate",
        )
    elif report["status"] == "completed":
        raise ValueError("Missing successful source controls")
    checked = {
        "rows": [
            {
                k: s[k]
                for k in (
                    "id",
                    "arm",
                    "variant",
                    "status",
                    "reason",
                    "training_transitions",
                    "updates",
                    "initial_actor_hash",
                    "final_actor_hash",
                )
                if k in s
            }
            for s in report["rows"]
        ],
        "source_controls": len(cases),
        "new_physics": 0,
        "slots": [],
    }
    ids = [report["supervisor"], *report["owned_identities"]]
    for stage in report["stages"]:
        if "process" in stage:
            ids.append(stage["process"])
    for ident in ids:
        f = Path(f"/proc/{ident['pid']}/stat")
        if f.exists():
            require(
                int(f.read_text().rsplit(")", 1)[1].split()[19])
                != ident["start_ticks"],
                "Source writer still live",
            )
    # Guards are process-local in this fresh independent audit, not source edits.
    import mujoco

    forbidden_calls = []

    def forbidden(*a, **k):
        forbidden_calls.append("mj_step")
        raise AssertionError("No audit physics permitted")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, forbidden)
    credit_path = out / "fixtures/native-credit.json"
    if credit_path.exists():
        checked["credit"] = check_credit(
            credit_path, packet["physics_limit"], report["status"] == "completed"
        )
        require(
            typed(checked["credit"]) == typed(report["physics_credit"]),
            "Reported physical credit differs",
        )
    checked["runtime"] = (
        check_runtime(out, packet, report)
        if any(s["stage"] == "runtime" for s in report["stages"])
        else {"status": "not_run"}
    )
    for slot in report["rows"]:
        if slot["status"] != "completed":
            continue
        watch = read(out / (slot["id"] + "-watchdog.json"))
        require(
            watch["status"] == "exited"
            and not watch["final_live"]
            and not watch["pending_retirements"]
            and watch["peak_capacity_charge_bytes"] <= packet["owned_rss_bytes"],
            "Watchdog resource fault",
        )
        require(
            max(watch["max_probe_s"], watch["max_sample_gap_s"]) <= 0.1,
            "Watchdog sampling fault",
        )
        ids.extend(watch["owned_identities"])
        for record in watch["retirement_records"].values():
            e = record["evidence"]
            require(
                e["tasks"] == e["after_tasks"] == [str(record["pid"])]
                and e["statm_pages"] == [0] * 7
                and e["status_threads"] == 1
                and e["vmrss_absent"]
                and e["vmsize_absent"],
                "Terminal-mm evidence",
            )
            require(
                all(
                    z["start_ticks"] == record["start_ticks"] and z["flags"] & 4
                    for z in (e["before"], e["after"], e["task"])
                ),
                "Terminal-mm identity/exit flags",
            )
        for sample in watch["recent_samples"]:
            require(
                sample["capacity_charge_bytes"]
                == sample["observed_rss_bytes"] + sample["retiring_rss_charge_bytes"],
                "RSS charge arithmetic",
            )
        publication = read(out / (slot["id"] + "-watchdog-publication.json"))
        require(
            publication["status"] == "copied_and_removed"
            and not Path(publication["directory"]).exists()
            and not publication["task_live"]
            and not publication["guard_live"],
            "Private publication lifetime",
        )
        for record in publication["copies"]:
            raw = Path(record["target"]).read_bytes()
            require(
                len(raw) == record["bytes"]
                and hashlib.sha256(raw).hexdigest() == record["sha256"],
                "Private publication exact bytes",
            )
        checked["slots"].append(
            {
                "id": slot["id"],
                **check_context(out / "fixtures" / slot["id"], packet, watch, root),
            }
        )
    if report["status"] == "completed":
        require(
            all(s["status"] == "completed" for s in report["rows"]),
            "False complete packet",
        )
        checked["comparison"] = check_trials(out / "fixtures", packet)
    require(not forbidden_calls, "Audit attempted physics")
    require(
        time.monotonic() - began <= 30
        and time.monotonic() - report["started_monotonic_s"] <= packet["whole_wall_s"],
        "Audit/whole epoch deadline",
    )
    require(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        <= packet["owned_rss_bytes"],
        "Audit RSS cap",
    )
    for identity in ids:
        f = Path(f"/proc/{identity['pid']}/stat")
        if f.exists():
            require(
                int(f.read_text().rsplit(")", 1)[1].split()[19])
                != identity["start_ticks"],
                "Owned identity still live",
            )
    checked.update(
        status="verified_" + report["status"],
        source_commit=args.source_sha,
        protocol_sha256=report["protocol_sha256"],
        exited_identities=len({(x["pid"], x["start_ticks"]) for x in ids}),
        task_wall_s=report["wall_s"],
        audit_wall_s=time.monotonic() - began,
        epoch_wall_through_audit_s=time.monotonic() - report["started_monotonic_s"],
    )
    destination = out / "independent-audit.json"
    require(not destination.exists(), "Fresh audit output required")
    fixtures, receipts = 0, 0
    manifest = hashlib.sha256()
    files = 0
    for path in sorted(out.parent.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            # Match the source storage denominator, including dereferenced test
            # symlink payloads; preserve link metadata in the safe transfer.
            if path.is_relative_to(out / "fixtures"):
                fixtures += len(raw)
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
            files += 1
    checked["saved_evidence"] = {
        "files": files,
        "manifest_sha256": manifest.hexdigest(),
        "fixture_bytes": fixtures,
        "receipt_bytes_before_audit": receipts,
    }
    raw = (json.dumps(checked, sort_keys=True, separators=(",", ":")) + "\n").encode()
    require(
        fixtures <= packet["fixture_bytes"]
        and receipts + len(raw) <= packet["receipt_bytes"],
        "Audited storage first limit",
    )
    require(
        time.monotonic() - report["started_monotonic_s"] <= packet["whole_wall_s"],
        "Original epoch through evidence inventory",
    )
    destination.write_bytes(raw)
    print(
        json.dumps(
            {
                k: checked[k]
                for k in (
                    "status",
                    "source_controls",
                    "exited_identities",
                    "task_wall_s",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
