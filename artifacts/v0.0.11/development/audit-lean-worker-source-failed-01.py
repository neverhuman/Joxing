"""Recheck saved failed Linux JUnit against frozen378-node registration."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

protocol = Path("experiments/v0.0.11/lean-rollout-worker-protocol.json")
folder = Path("artifacts/v0.0.11/lean-worker-source-failed-01")
p = json.loads(protocol.read_text())
raw = (folder / "controls.xml").read_bytes()
prior = json.loads((folder / "prerequisite-failure-audit.json").read_text())
expected = {
    (
        n.split("::")[0][:-3].replace("/", ".")
        + "".join("." + x for x in n.split("::")[1:-1]),
        n.split("::")[-1],
    ): n
    for n in p["test_node_ids"]
}
cases = list(ET.fromstring(raw).iter("testcase"))
assert len(cases) == len(expected) == 378
actual = []
for c in cases:
    status = (
        "failed"
        if c.find("failure") is not None or c.find("error") is not None
        else "skipped"
        if c.find("skipped") is not None
        else "passed"
    )
    actual.append(
        {"node": expected[c.get("classname"), c.get("name")], "status": status}
    )
assert (
    actual == prior["cases"]
    and hashlib.sha256(raw).hexdigest() == prior["junit_sha256"]
)
assert len({r["node"] for r in actual}) == 378
assert {
    s: sum(r["status"] == s for r in actual) for s in ("passed", "failed", "skipped")
} == {"passed": 375, "failed": 3, "skipped": 0}
assert (
    prior["protocol_sha256"]
    == hashlib.sha256(
        json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
)
assert len(prior["rows"]) == 4 and all(s["status"] == "not_run" for s in prior["rows"])
print(
    json.dumps(
        {
            "status": "saved_failed_gate_reverified",
            "cases": 378,
            "passed": 375,
            "failed": 3,
            "skipped": 0,
            "formal_probes": 0,
            "native_physics": 0,
        }
    )
)
