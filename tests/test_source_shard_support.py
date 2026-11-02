import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "source_shards",
    Path(__file__).resolve().parents[1] / "scripts/source_shard_support.py",
)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)

NODES = [
    "tests/test_a.py::test_one",
    "grokbot/tests/test_b.py::test_two[x]",
    "tests/test_a.py::test_three",
]


def test_partition_preserves_every_identity_once_across_fresh_module_processes():
    plan = support.partition(NODES, [NODES[1]])
    assert [s["file"] for s in plan] == ["tests/test_a.py", "grokbot/tests/test_b.py"]
    assert plan[0]["node_ids"] == [NODES[0], NODES[2]]
    assert plan[1]["node_ids"] == [NODES[1]]
    assert sorted(n for s in plan for n in s["node_ids"]) == sorted(NODES)


@pytest.mark.parametrize(
    "nodes,xfails",
    [
        ([], []),
        (NODES + [NODES[0]], []),
        (["../tests/test_a.py::test_one"], []),
        (NODES, ["tests/test_missing.py::test_absent"]),
    ],
)
def test_unsafe_incomplete_or_duplicate_registry_rejected(nodes, xfails):
    with pytest.raises(ValueError):
        support.partition(nodes, xfails)


def test_actual_junit_identities_and_exact_known_xfail_are_required():
    xml = (
        '<testsuite><testcase classname="tests.test_a" name="test_one"/>'
        '<testcase classname="grokbot.tests.test_b" name="test_two[x]"><skipped type="pytest.xfail"/></testcase>'
        '<testcase classname="tests.test_a" name="test_three"/></testsuite>'
    )
    result = support.check_junit(xml, NODES, [NODES[1]])
    assert result["qualified"] and result["counts"] == {
        "passed": 2,
        "xfail": 1,
        "failed": 0,
        "skipped": 0,
    }
    assert not support.check_junit(
        xml.replace('<skipped type="pytest.xfail"/>', ""), NODES, [NODES[1]]
    )["qualified"]
    assert not support.check_junit(
        xml.replace("pytest.xfail", "pytest.skip"), NODES, [NODES[1]]
    )["qualified"]
    with pytest.raises(ValueError):
        support.check_junit(xml.replace("test_three", "test_one"), NODES, [NODES[1]])
    with pytest.raises(ValueError):
        support.check_junit(
            xml.replace("test_three", "test_unknown"), NODES, [NODES[1]]
        )


def test_failed_case_remains_in_complete_denominator():
    xml = '<testsuite><testcase classname="tests.test_a" name="test_one"><failure message="fault"/></testcase></testsuite>'
    result = support.check_junit(xml, [NODES[0]], [])
    assert not result["qualified"] and result["rows"] == [
        {"node": NODES[0], "status": "failed"}
    ]
