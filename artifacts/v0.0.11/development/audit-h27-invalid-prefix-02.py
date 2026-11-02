import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

root = Path("/home/ubuntu/joxing/native-learner-context-parity-20261102-01")
out = root / "runs/v0.0.11/native-learner-context-parity-01"
trial = out / "fixtures/legacyA"
sys.path[:0] = [str(root / "scripts"), str(root / "src")]
import audit_native_learner_context as a

p = a.read(root / "experiments/v0.0.11/native-learner-context-parity-protocol.json")
r = json.loads(gzip.decompress((out / "result.json.gz").read_bytes()))
start = time.monotonic()
assert 0 <= start - r["started_monotonic_s"] < 295
import mujoco


def forbidden(*args, **kwargs):
    raise AssertionError("No forensic physics")


for name in ("mj_step", "mj_step1", "mj_step2"):
    setattr(mujoco, name, forbidden)
learner = [x["semantic"] for x in a.rows(trial / "traces/learner.jsonl")]
assert len(learner) == 31
assert [x["updates"] for x in learner if x["phase"] == "update_before"] == list(
    range(12)
)
assert [x["updates"] for x in learner if x["phase"] == "update_after"] == list(
    range(1, 13)
)
assert len(next(x["replay"] for x in learner if x["phase"] == "learn_after")) == 64
cold = a.read(trial / "training-constructor-states.json")
before = a.read(trial / "trained-before-close.json")
assert len(cold) == len(before["states"]) == 4
assert all(x["muscles"] == 700 and x["state"] == p["expected_state_hash"] for x in cold)
workers = []
for i, st in enumerate(before["states"]):
    records = a.rows(trial / f"traces/worker-{i}.jsonl")
    native = a.rows(trial / f"traces/native-world-{i}.jsonl")
    phases = a.rows(trial / f"world-{i}/resource-release.jsonl")
    last = records[-1]["semantic"]
    assert sum(x["semantic"]["phase"] == "step" for x in records) == 16
    assert sum(x["semantic"]["phase"] == "step" for x in native) == 16
    assert (
        last["phase"] == "close"
        and last["state"] == st["state"] == native[-1]["semantic"]["state"]
        and last["reset_rng"] == st["reset_rng"]
    )
    assert phases[0]["state_hash"] == st["state"] and phases[0]["muscles"] == 700
    if len(phases) > 1:
        assert phases[1]["state_hash"] == st["state"]
    graph = next((x for x in phases if x["phase"] == "graph_checked"), None)
    if graph:
        assert graph["weakrefs_dead"] == {"env": True, "model": True, "data": True}
    final = trial / f"world-{i}/context-finalizer.jsonl"
    fin = a.rows(final) if final.exists() else []
    proc = (
        a.read(trial / f"world-{i}/context-worker.json")
        if (trial / f"world-{i}/context-worker.json").exists()
        else None
    )
    workers.append(
        {
            "index": i,
            "identity": before["workers"][i],
            "steps": 16,
            "trained_state_hash": st["state"],
            "reset_rng_hash": st["reset_rng"],
            "release_phases": [x["phase"] for x in phases],
            "graph_dead": graph["weakrefs_dead"] if graph else None,
            "rss_before_close": phases[0]["memory"]["rss_bytes"],
            "rss_after_native_release": graph["memory"]["rss_bytes"] if graph else None,
            "finalizers": fin,
        }
    )
hashes = {
    name: a.actor_hash(trial / "trial" / f"{name}-actor.pt")
    for name in ("initial", "final")
}
assert hashes["initial"] != hashes["final"]
w = a.read(out / "legacyA-watchdog.json")
credit = a.check_credit(out / "fixtures/native-credit.json", p["physics_limit"], False)
assert credit["used"] == 5086 and not credit["pending"]
traces = {
    f.name: {
        "bytes": f.stat().st_size,
        "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        "rows": len(a.rows(f)),
    }
    for f in sorted((trial / "traces").glob("*.jsonl"))
}
assert len(traces) == 10
value = {
    "status": "invalid_prefix_verified_not_parity",
    "source_commit": r["source_commit"],
    "protocol_sha256": r["protocol_sha256"],
    "rows": r["rows"],
    "transitions_observed": 64,
    "updates_observed": 12,
    "valid_trials": 0,
    "comparison_qualified": False,
    "new_audit_physics": 0,
    "actors": hashes,
    "credit": credit,
    "workers": workers,
    "traces": traces,
    "watcher_fault": w["reason"],
    "retirements": w["retirement_records"],
    "serial_close_events": a.rows(trial / "serial-close.jsonl"),
    "forensic_pid": os.getpid(),
    "started_monotonic_s": start,
    "original_epoch": r["started_monotonic_s"],
    "epoch_through_forensics_s": time.monotonic() - r["started_monotonic_s"],
    "scope": "Separate within-original-epoch saved invalid prefix audit; source trial remains invalid; no counterpart or parity/motor claim.",
}
assert time.monotonic() - start < 30 and value["epoch_through_forensics_s"] < 300
(out / "invalid-prefix-forensics.json").write_text(
    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
)
print(
    json.dumps(
        {
            k: value[k]
            for k in (
                "status",
                "transitions_observed",
                "updates_observed",
                "actors",
                "epoch_through_forensics_s",
            )
        }
    )
)
