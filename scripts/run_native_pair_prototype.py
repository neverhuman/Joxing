"""Registered small real-checkpoint PPO witness; no competitive training."""

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    for n in ("packet", "source-sha", "protocol-sha256"):
        ap.add_argument("--" + n, required=True)
    a = ap.parse_args()
    p = json.loads((ROOT / a.packet).read_text())
    sha = hashlib.sha256(
        json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        p["schema"] != "joxing-native-pair-mac-prototype-v1"
        or sha != a.protocol_sha256
        or p["joint_steps"] != 32
        or p["horizon_ticks"] != 1000
    ):
        raise ValueError("Explicit frozen native prototype required")
    from joxing.native_pair_motor_train import run_native_smoke

    entered = 0
    started = time.monotonic()
    events = ROOT / p["output"] / "physics.jsonl"

    def guard(n):
        nonlocal entered
        if entered + n + p["source_fixture_physics_reserve"] > p["physics_steps"]:
            raise RuntimeError("Entered physics first limit")
        if time.monotonic() - started > p["native_wall_s"]:
            raise TimeoutError("Native prototype wall first limit")
        if n:
            entered += n
            if entered % 20 == 0:
                with events.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "entered_physics": entered,
                                "monotonic_s": time.monotonic(),
                            }
                        )
                        + "\n"
                    )

    def interrupted(*_):
        raise TimeoutError("Owned native prototype interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        r = run_native_smoke(
            p["teacher_inputs"],
            ROOT / p["output"] / "native",
            seed=p["seed"],
            joint_steps=p["joint_steps"],
            horizon_ticks=p["horizon_ticks"],
            guard=guard,
        )
        if r["native_physics"] != entered:
            raise ValueError("World/entry physics counters differ")
        print(
            json.dumps(
                {
                    "status": r["status"],
                    "physics": entered,
                    "training": r["training"]["actor_samples"],
                    "updates": r["training"]["rollout_updates"],
                }
            )
        )
    finally:
        (ROOT / p["output"] / "physics-final.json").write_text(
            json.dumps(
                {
                    "entered_physics": entered,
                    "source_fixture_reserved_not_measured": p[
                        "source_fixture_physics_reserve"
                    ],
                    "hard_physics_limit": p["physics_steps"],
                    "wall_s": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
