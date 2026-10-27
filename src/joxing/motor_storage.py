"""Pre-write bounded raw credits for disjoint D2 output owners."""

import io
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

CREDITS = {}
PRIVATE_PREFIXES = ("worker-", "initial-", "final-", "replay-")


class RawCredit:
    def __init__(self, root, maximum, parent=False):
        self.root = Path(root).resolve()
        self.maximum = int(maximum)
        self.parent = parent
        self.reserve = 2 * 1024**2
        if self.maximum <= self.reserve:
            raise ValueError("Raw credit must include fault reserve")
        self.emergency_mode = False
        CREDITS[self.root] = self

    def owns(self, path):
        try:
            parts = Path(path).resolve().relative_to(self.root).parts
        except ValueError:
            return False
        return not (
            self.parent and any(p.startswith(PRIVATE_PREFIXES) for p in parts[:-1])
        )

    def used(self):
        return sum(
            p.stat().st_size
            for p in self.root.rglob("*")
            if p.is_file() and self.owns(p)
        )

    def check(self, path, size, append=False):
        path = Path(path)
        prior = path.stat().st_size if path.exists() and not append else 0
        # Atomic replacements briefly retain both old and new bytes on disk.
        peak = self.used() + size
        limit = self.maximum if self.emergency_mode else self.maximum - self.reserve
        if peak > limit:
            raise TimeoutError("Pre-write raw credit exhausted: " + str(self.root))
        return self.used() - prior + size

    @contextmanager
    def emergency(self):
        before = self.emergency_mode
        self.emergency_mode = True
        try:
            yield
        finally:
            self.emergency_mode = before


def credit(path):
    matches = [c for c in CREDITS.values() if c.owns(path)]
    return max(matches, key=lambda c: len(c.root.parts)) if matches else None


def write_bytes(path, value):
    path = Path(path)
    owner = credit(path)
    if owner:
        owner.check(path, len(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_json(path, value):
    write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def append_json(path, value):
    path = Path(path)
    value = (json.dumps(value, allow_nan=False) + "\n").encode()
    owner = credit(path)
    if owner:
        owner.check(path, len(value), append=True)
    with path.open("ab") as file:
        file.write(value)


def save_npz(path, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    write_bytes(path, buffer.getvalue())
