"""Execute only the frozen two-seed motor pilot with durable entered counters."""

import argparse
import json
import signal
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    a = ap.parse_args()
    p = json.loads((ROOT / a.packet).read_text())
    if (
        p["schema"] != "joxing-native-pair-motor-pilot-v1"
        or p["joint_steps"] != 1024
        or p["seeds"] != [761, 983]
    ):
        raise ValueError("Frozen bounded motor profile")
    from joxing.native_pair_motor_evidence import EvidenceBudget
    from joxing.native_pair_motor_pilot import run_native_pilot

    out = ROOT / p["output"]
    budget = EvidenceBudget(
        out, raw_limit=p["raw_bytes"], text_limit=p["receipt_bytes"]
    )
    started = time.monotonic()
    entered = 0
    ledger = (out / "primary-entered.bin").open("xb+", buffering=0)

    def guard(n):
        nonlocal entered
        if entered + n + p["source_fixture_physics_reserve"] > p["physics_steps"]:
            raise RuntimeError("Primary physics first limit")
        if time.monotonic() - started > p["native_wall_s"] - 5:
            raise TimeoutError("Native whole stage first limit")
        if n:
            entered += n
            ledger.seek(0)
            ledger.write(struct.pack("<QQ", entered, entered ^ ((1 << 64) - 1)))

    def interrupted(*_):
        raise TimeoutError("Owned native stage interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        r = run_native_pilot(
            p["teacher_inputs"], out / "native", budget=budget, guard=guard
        )
        if r["native_physics"] != entered:
            raise ValueError("World/entered counter mismatch")
        print(json.dumps({"status": r["status"], "entered_physics": entered}))
    finally:
        ledger.close()
        reserve = budget.reserve
        budget.reserve = 0
        try:
            budget.write_json(
                out / "physics-final.json",
                {
                    "entered_physics": entered,
                    "source_reserved": p["source_fixture_physics_reserve"],
                    "wall_s": time.monotonic() - started,
                },
            )
        finally:
            budget.reserve = reserve


if __name__ == "__main__":
    main()
