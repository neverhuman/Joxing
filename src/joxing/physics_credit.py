"""Atomic diagnostic physics credits. Unsettled reservations remain charged."""

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path


def process_identity():
    pid = os.getpid()
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return {"pid": pid, "start": stat.read_text().rsplit(")", 1)[1].split()[19]}
    # Development portability only; formal qualification requires Linux /proc.
    return {
        "pid": pid,
        "start": subprocess.check_output(
            ["ps", "-o", "lstart=", "-p", str(pid)], text=True
        ).strip(),
    }


class PhysicsCredit:
    def __init__(self, path, binding):
        self.path, self.binding = Path(path), binding
        self.owner = process_identity()

    @classmethod
    def create(cls, path, binding, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError("Positive integer credit limit required")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(
                {
                    "schema": "joxing-physics-credit-v1",
                    "binding": binding,
                    "limit": limit,
                    "used": 0,
                    "pending": {},
                    "settled": {},
                    "sequence": 0,
                    "chain": "0" * 64,
                },
                stream,
            )
        return cls(path, binding)

    @contextlib.contextmanager
    def locked(self):
        if process_identity() != self.owner:
            raise ValueError("Foreign or stale ledger object owner")
        with self.path.with_suffix(".lock").open("a+") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text())
            if (
                data["schema"] != "joxing-physics-credit-v1"
                or data["binding"] != self.binding
            ):
                raise ValueError("Physics credit binding drift")
            yield data

    def write(self, data, event):
        data["sequence"] += 1
        event = {"sequence": data["sequence"], "owner": self.owner, **event}
        raw = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
        data["chain"] = hashlib.sha256((data["chain"] + raw).encode()).hexdigest()
        with self.path.with_suffix(".jsonl").open("a") as stream:
            stream.write(raw + "\n")
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, sort_keys=True) + "\n")
        tmp.replace(self.path)

    def reserve(self, amount, slot, kind):
        if (
            type(amount) is not int
            or amount <= 0
            or kind not in ("preparation", "step")
        ):
            raise ValueError("Invalid physics reservation")
        with self.locked() as data:
            charged = data["used"] + sum(
                r["reserved"] for r in data["pending"].values()
            )
            if charged + amount > data["limit"]:
                raise RuntimeError("Physics first limit")
            token = str(data["sequence"] + 1)
            data["pending"][token] = {
                "owner": self.owner,
                "reserved": amount,
                "slot": slot,
                "kind": kind,
                "state": "reserved",
            }
            self.write(
                data,
                {
                    "operation": "reserve",
                    "token": token,
                    "amount": amount,
                    "slot": slot,
                    "kind": kind,
                },
            )
            return token

    def commit(self, token, actual):
        with self.locked() as data:
            row = data["pending"].get(token)
            if row is None or row["owner"] != self.owner or row["state"] != "reserved":
                raise ValueError("Foreign/stale/unknown reservation settlement")
            if type(actual) is not int or not 0 <= actual <= row["reserved"]:
                raise ValueError("Invalid actual physics credit")
            data["used"] += actual
            key = row["slot"] + ":" + row["kind"]
            aggregate = data["settled"].setdefault(key, {"operations": 0, "used": 0})
            aggregate["operations"] += 1
            aggregate["used"] += actual
            del data["pending"][token]
            self.write(data, {"operation": "commit", "token": token, "actual": actual})

    def unknown(self, token, error):
        with self.locked() as data:
            row = data["pending"].get(token)
            if row is None or row["owner"] != self.owner:
                raise ValueError("Foreign/stale unknown reservation")
            row["state"] = "unknown"
            row["error"] = str(error)
            self.write(
                data, {"operation": "unknown", "token": token, "error": str(error)}
            )

    def snapshot(self):
        with self.locked() as data:
            return data | {
                "charged": data["used"]
                + sum(r["reserved"] for r in data["pending"].values())
            }
