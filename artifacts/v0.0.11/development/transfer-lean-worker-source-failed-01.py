"""Hash-verified owned regular payload transfer; preserve link metadata."""

import gzip
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

meta = json.loads(Path("runs/v0.0.11/h28-raw-manifest.json").read_text())
root = "/home/ubuntu/joxing/lean-worker-probe-20261102-01"
names = sorted(meta["files"])
assert all(
    n.startswith("runs/v0.0.11/")
    and ".." not in Path(n).parts
    and not Path(n).is_absolute()
    for n in names
)
cmd = [
    "tar",
    "--dereference",
    "--hard-dereference",
    "-C",
    root,
    "-cf",
    "-",
    "--null",
    "-T",
    "-",
]
r = subprocess.run(
    ["ssh", "xbabe3", shlex.join(cmd)],
    input=b"".join(n.encode() + bytes([0]) for n in names),
    capture_output=True,
    timeout=60,
    check=True,
)
archive = Path("runs/v0.0.11/h28-owned-bytes.tar")
assert not archive.exists()
archive.write_bytes(r.stdout)
out = Path("runs/v0.0.11/h28-safe-evidence")
out.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
    members = t.getmembers()
    assert (
        len(members) == len(names)
        and {m.name for m in members} == set(names)
        and all(m.isfile() for m in members)
    )
    t.extractall(out, filter="data")
for name, d in meta["files"].items():
    raw = (out / name).read_bytes()
    assert len(raw) == d["bytes"] and hashlib.sha256(raw).hexdigest() == d["sha256"], (
        name
    )
report = {
    "status": "verified_all_owned_payloads",
    "files": len(names),
    "bytes": sum(v["bytes"] for v in meta["files"].values()),
    "archive_sha256": hashlib.sha256(r.stdout).hexdigest(),
    "original_symlinks": meta["links"],
    "representation": "Only dereferenced regular owned bytes extracted with data filter; exact original symlink metadata retained separately.",
    "scope": "Post-task publication/transfer; no extra simulation or original continuous compute-envelope claim.",
}
curated = Path("artifacts/v0.0.11/lean-worker-source-failed-01")
curated.mkdir(exist_ok=False)
(curated / "transfer.json").write_text(json.dumps(report, indent=2) + "\n")
raw = (json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n").encode()
blob = gzip.compress(raw, compresslevel=9, mtime=0)
(curated / "raw-manifest.json.gz").write_bytes(blob)
(curated / "raw-manifest-encoding.json").write_text(
    json.dumps(
        {
            "decoded_bytes": len(raw),
            "decoded_sha256": hashlib.sha256(raw).hexdigest(),
            "encoded_bytes": len(blob),
            "encoded_sha256": hashlib.sha256(blob).hexdigest(),
            "scope": "Exact small text manifest only; no checkpoint/model/tape payload in Git.",
        },
        indent=2,
    )
    + "\n"
)
print(
    json.dumps({k: report[k] for k in ("status", "files", "bytes", "archive_sha256")})
)
