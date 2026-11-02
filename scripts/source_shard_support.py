"""Exact module partition and JUnit identity checks; standard library only."""

import xml.etree.ElementTree as ET


def partition(nodes, xfails):
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or len(xfails) != len(set(xfails))
        or not set(xfails) <= set(nodes)
    ):
        raise ValueError("Source shard unique complete denominator")
    modules = {}
    for node in nodes:
        file, separator, _ = node.partition("::")
        if (
            not separator
            or not file.endswith(".py")
            or not file.startswith(("tests/", "grokbot/tests/"))
            or ".." in file.split("/")
        ):
            raise ValueError("Source shard test identity/path")
        modules.setdefault(file, []).append(node)
    return [
        {
            "file": file,
            "node_ids": selected,
            "expected_xfails": [n for n in selected if n in xfails],
        }
        for file, selected in modules.items()
    ]


def check_junit(xml, nodes, xfails):
    partition(nodes, xfails)
    mapping = {}
    for node in nodes:
        parts = node.split("::")
        cls = parts[0][:-3].replace("/", ".") + "".join("." + x for x in parts[1:-1])
        key = cls, parts[-1]
        if key in mapping:
            raise ValueError("Ambiguous source JUnit identity")
        mapping[key] = node
    rows, seen = [], set()
    for case in ET.fromstring(xml).iter("testcase"):
        key = case.get("classname"), case.get("name")
        if key not in mapping or key in seen:
            raise ValueError("Unknown/duplicate source testcase")
        seen.add(key)
        node = mapping[key]
        skip = case.find("skipped")
        failed = case.find("failure") is not None or case.find("error") is not None
        state = (
            "failed"
            if failed
            else "xfail"
            if skip is not None and skip.get("type") == "pytest.xfail"
            else "skipped"
            if skip is not None
            else "passed"
        )
        rows.append({"node": node, "status": state})
    if seen != set(mapping):
        raise ValueError("Missing source testcase")
    counts = {
        state: sum(r["status"] == state for r in rows)
        for state in ("passed", "xfail", "failed", "skipped")
    }
    qualified = all(
        r["status"] == ("xfail" if r["node"] in xfails else "passed") for r in rows
    )
    return {"qualified": qualified, "counts": counts, "rows": rows}
