"""Named close-only adapter. No native imports or changes to the base class."""

import json
import math
import time
from pathlib import Path


def read_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "start": fields[19], "state": fields[0]}


def serial_vec_class(
    base,
    receipt_path,
    close_deadline_s=2.0,
    *,
    identity_reader=read_identity,
    clock=time.monotonic,
):
    """Return an opt-in subclass; base reset/step/constructor behavior is retained."""
    if not math.isfinite(close_deadline_s) or close_deadline_s <= 0:
        raise ValueError("Positive finite serial close deadline required")
    path = Path(receipt_path)

    class SerialClose(base):
        def __init__(self, *args, **kwargs):
            self._serial_failed = False
            self._serial_bound = []
            super().__init__(*args, **kwargs)
            if not self.processes or len(self.processes) != len(self.remotes):
                raise ValueError("Serial endpoint/process denominator")
            for process in self.processes:
                bound = identity_reader(process.pid)
                if (
                    bound is None
                    or bound["state"] in ("Z", "X")
                    or not process.is_alive()
                ):
                    raise RuntimeError("Serial worker not live at binding")
                self._serial_bound.append(bound)
            if len({b["pid"] for b in self._serial_bound}) != len(self.processes):
                raise ValueError("Duplicate serial worker identity")
            self._serial_emit("bound", workers=self._serial_bound)

        def _serial_emit(self, phase, **fields):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {"phase": phase, "monotonic_s": clock(), **fields},
                        sort_keys=True,
                    )
                    + "\n"
                )

        def close(self):
            if self.closed:
                return
            if self._serial_failed:
                raise RuntimeError(
                    "Serial close previously failed; owned cleanup required"
                )
            deadline = clock() + close_deadline_s

            def remaining():
                left = deadline - clock()
                if left <= 0:
                    raise TimeoutError("Serial close overall deadline")
                return left

            try:
                if len(self._serial_bound) != len(self.processes) or len(
                    self.remotes
                ) != len(self.processes):
                    raise ValueError("Serial close incomplete binding")
                if self.waiting:
                    for i, remote in enumerate(self.remotes):
                        if not remote.poll(remaining()):
                            raise TimeoutError("Serial pending response deadline")
                        remote.recv()
                        self._serial_emit("drained", worker=i)
                    self.waiting = False
                for i, (remote, process, bound) in enumerate(
                    zip(self.remotes, self.processes, self._serial_bound, strict=True)
                ):
                    remaining()
                    current = identity_reader(bound["pid"])
                    if (
                        current is None
                        or current["start"] != bound["start"]
                        or current["state"] in ("Z", "X")
                        or not process.is_alive()
                    ):
                        raise RuntimeError("Serial worker identity/liveness drift")
                    self._serial_emit("close_send", worker=i, identity=bound)
                    remote.send(("close", None))
                    process.join(timeout=remaining())
                    if process.is_alive():
                        raise TimeoutError("Serial worker join deadline")
                    remaining()
                    after = identity_reader(bound["pid"])
                    if after is not None and after["start"] == bound["start"]:
                        raise RuntimeError("Serial joined identity still present")
                    if process.exitcode != 0:
                        raise RuntimeError("Serial worker nonzero exit")
                    self._serial_emit(
                        "joined", worker=i, identity=bound, exitcode=process.exitcode
                    )
                    remote.close()
                self.closed = True
                self._serial_emit("closed")
            except BaseException as error:
                self._serial_failed = True
                self._serial_emit(
                    "fault", error=type(error).__name__ + ": " + str(error)
                )
                raise

    return SerialClose
