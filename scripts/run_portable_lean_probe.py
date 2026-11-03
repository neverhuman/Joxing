"""H30 child entry: validate outer/inner binding before fake-only dispatch."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_lean_probe_support import dispatch, validate


def main():
    ap = argparse.ArgumentParser()
    for n in ("packet", "slot", "folder", "ready", "source-sha", "protocol-sha256"):
        ap.add_argument("--" + n, required=True)
    a = ap.parse_args()
    p = json.loads((ROOT / a.packet).read_text())
    validate(p)
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != a.source_sha
    ):
        raise ValueError("Trial source binding")
    from joxing.lean_worker_probe import run

    value = dispatch(
        ROOT, p, a.slot, a.folder, a.ready, a.source_sha, a.protocol_sha256, run
    )
    print(
        json.dumps(
            {
                "slot": a.slot,
                "physics": value["physics"],
                "actors": value["actors"],
                "updates": value["updates"],
                "inner_packet_sha256": p["inner_packet_sha256"],
            }
        )
    )


if __name__ == "__main__":
    raise SystemExit(
        "H30 DRAFT: independent audit/protocol not registered; formal dispatch disabled. Native motor RL is the priority; see docs/NATIVE_PAIR_RL_PILOT_MISSION.md"
    )
