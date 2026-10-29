"""Dependency-free source-gate accounting and summary checks."""

import re

KNOWN_XFAIL = (
    "grokbot/tests/test_referee_fuzz.py::test_restore_does_not_alias_the_snapshot"
)


def partition_usage(root, fixture):
    scratch = receipts = 0
    for f in root.rglob("*"):
        try:
            if f.is_file():
                amount = f.stat().st_size
                if f.is_relative_to(fixture):
                    scratch += amount
                else:
                    receipts += amount
        except FileNotFoundError:
            pass  # Atomic replacement/deletion by a test is expected.
    return scratch, receipts


def validate_summary(packet, text):
    nodes = packet["expected_xfail_nodes"]
    expected = [] if packet["mode"] == "isolation_diagnostic" else [KNOWN_XFAIL]
    if (
        nodes != expected
        or re.findall(r"(\d+) passed", text) != [str(packet["expected_tests"])]
        or re.findall(r"(?m)^XFAIL ([^ ]+)", text) != nodes
        or re.findall(r"(\d+) xfailed", text) != ([str(len(nodes))] if nodes else [])
    ):
        raise ValueError("Exact source-gate summary/known-xfail set")
