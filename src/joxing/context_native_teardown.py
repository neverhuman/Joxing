"""H21 ONE original native reset, with observed forkserver transport/release."""

import argparse
import json
import re
from pathlib import Path

from .deferred_qualification import wait_ready
from .native_context import NativeContextProof
from .process_exit_probe import write_json


def run(folder, slot, packet, ledger_path, binding, ready):
    if packet["schema"] != "joxing-native-context-transfer-v1" or not re.fullmatch(
        "[0-9a-f]{64}", packet["expected_state_hash"]
    ):
        raise ValueError("Explicit native context packet/state required")
    wait_ready(ready, 15)
    proof = NativeContextProof(Path(str(folder) + "-context"))
    proof.prepare()  # Before any numerical imports into this fresh root.
    from .release_native_teardown import run as release_run

    result = release_run(
        folder,
        slot,
        packet,
        ledger_path,
        binding,
        ready,
        start_method=proof.method,
        worker_factory=proof.factory(),
        after_construct=proof.after_construct,
        expected_state=packet["expected_state_hash"],
    )
    result = result | {
        "schema": packet["schema"],
        "context": proof.qualify(result, folder),
    }
    write_json(Path(folder) / "constructor-result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("folder", "slot", "packet", "ledger", "binding", "ready"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    run(
        args.folder,
        args.slot,
        json.loads(Path(args.packet).read_text()),
        args.ledger,
        args.binding,
        args.ready,
    )
