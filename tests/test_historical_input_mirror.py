import base64
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from historical_input_mirror import materialize


def source(tmp_path):
    root = tmp_path / "fresh"
    root.mkdir()
    raw = b'{"actual":"old receipt bytes"}\n'
    name = "/home/ubuntu/joxing/historical-input/runs/result.json"
    pack = {"files": {name: {"data_base64": base64.b64encode(raw).decode()}}}
    archive = (json.dumps(pack) + "\n").encode()
    blob = gzip.compress(archive, mtime=0)
    (root / "pack.gz").write_bytes(blob)
    marker = {"scope": "INPUT MIRROR, NOT EXECUTION"}
    config = {
        "aliases": ["historical-input"],
        "archive": "pack.gz",
        "encoded_sha256": hashlib.sha256(blob).hexdigest(),
        "decoded_archive_bytes": len(archive),
        "decoded_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}},
        "marker": marker,
    }
    charge = len(raw) + len(
        (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    return root, config, charge, raw


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "existing",
        "dangling",
        "charge",
        "archive",
        "payload",
        "foreign",
        "repeat",
    ],
)
def test_owned_exact_readonly_inputs_and_no_historical_overwrite(tmp_path, fault):
    root, cfg, charge, raw = source(tmp_path)
    alias = tmp_path / "historical-input"
    if fault == "existing":
        alias.mkdir()
        (alias / "original").write_text("keep")
    elif fault == "dangling":
        alias.symlink_to(tmp_path / "absent", target_is_directory=True)
    elif fault == "charge":
        charge -= 1
    elif fault == "archive":
        (root / "pack.gz").write_bytes(b"corrupt")
    elif fault == "payload":
        next(iter(cfg["files"].values()))["sha256"] = "0" * 64
    elif fault == "foreign":
        cfg["aliases"] = ["../outside"]
    elif fault == "repeat":
        materialize(root, cfg, charge, base=tmp_path)
    if fault == "none":
        result = materialize(root, cfg, charge, base=tmp_path)
        path = alias / "runs/result.json"
        assert path.read_bytes() == raw
        assert alias.is_symlink() and path.stat().st_mode & 0o222 == 0
        assert result["charged_fixture_bytes"] == charge
        assert (
            json.loads(
                (
                    Path(result["destination"]) / "INPUT_MIRROR_NOT_EXECUTION.json"
                ).read_text()
            )
            == cfg["marker"]
        )
    else:
        with pytest.raises((ValueError, FileExistsError)):
            materialize(root, cfg, charge, base=tmp_path)
        if fault == "existing":
            assert (alias / "original").read_text() == "keep"
        if fault not in ("existing", "dangling", "repeat"):
            assert not alias.exists()
