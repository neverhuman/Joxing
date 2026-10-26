"""Small, inspectable experiment manifests; no credentials or process environments."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def revision(path: Path) -> str:
    top = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"],capture_output=True,text=True)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != path.resolve():
        return "uncommitted"
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def upstream_revision(name: str) -> str:
    value = revision(ROOT / "third_party" / name)
    if value != "uncommitted":
        return value
    # Bundle-based compute deployments retain root Git metadata and gitlink pins.
    row = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "HEAD", "third_party/"+name],capture_output=True,text=True)
    parts = row.stdout.split()
    return parts[2] if len(parts) >= 3 and parts[0] == "160000" else "uncommitted"


def source_hash() -> str:
    files = sorted((ROOT/"src/joxing").glob("*.py"))
    return digest({str(p.relative_to(ROOT)):p.read_text() for p in files})


def manifest() -> dict:
    return {
        "schema": "joxing-run-v1", "utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": revision(ROOT), "source_hash":source_hash(), "python": platform.python_version(),
        "mujoco": mujoco.__version__, "numpy": np.__version__,
        "host": platform.node(), "platform": platform.platform(),
        "upstream": {name: upstream_revision(name) for name in ("ms-human-700", "msgym")},
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
