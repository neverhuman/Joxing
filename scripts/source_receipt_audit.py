"""Complete source-gate audit for raw failed and packed successful JUnit."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from receipt_encoding import decode_gzip
from source_shard_support import check_junit


def audit_junit(folder, nodes, returncode, *, started=True):
    folder = Path(folder)
    rawfile = folder / "controls.xml"
    packed = folder / "controls.xml.gz"
    value = {
        "qualified": False,
        "complete": False,
        "producer_returncode": returncode,
        "rows": [
            {"node": n, "status": "unconfirmed" if started else "not_run"}
            for n in nodes
        ],
    }
    if not rawfile.exists() and not packed.exists():
        value["error"] = (
            "Missing JUnit after source start" if started else "Source not_run"
        )
        return value
    try:
        raw = (
            rawfile.read_bytes()
            if rawfile.exists()
            else decode_gzip(packed.read_bytes(), limit=1024 * 1024)
        )
        if len(raw) > 1024 * 1024:
            raise ValueError("JUnit byte first limit")
        if (
            rawfile.exists()
            and packed.exists()
            and decode_gzip(packed.read_bytes(), limit=1024 * 1024) != raw
        ):
            raise ValueError("Raw/packed JUnit disagreement")
        value.update(
            raw_bytes=len(raw),
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            representation="raw" if rawfile.exists() else "gzip",
        )
        checked = check_junit(raw.decode(), nodes, [])
        cases = list(ET.fromstring(raw).iter("testcase"))
        for row, case in zip(checked["rows"], cases, strict=True):
            if case.find("error") is not None:
                row["failure_kind"] = "error"
            elif case.find("failure") is not None:
                row["failure_kind"] = "failure"
        value.update(
            checked, complete=True, qualified=checked["qualified"] and returncode == 0
        )
        if checked["qualified"] and returncode != 0:
            value["error"] = "Passing XML cannot override nonzero producer"
    except Exception as error:  # noqa: BLE001 -- retain incomplete/unknown denominator, never false pass
        value.update(
            qualified=False,
            complete=False,
            error=type(error).__name__ + ": " + str(error),
        )
    return value
