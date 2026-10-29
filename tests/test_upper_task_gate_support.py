import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "scripts/upper_task_gate_support.py"
spec = importlib.util.spec_from_file_location("gate_support", FILE)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def test_changing_fixture_sizes_never_become_receipt_bytes(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (tmp_path / "receipt.json").write_bytes(b"r" * 123)
    stop = threading.Event()

    def writer():
        f = fixtures / "growing.bin"
        while not stop.is_set():
            f.write_bytes(b"x" * (1024 * 1024))
            f.write_bytes(b"x" * 17)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(100):
            _, receipts = support.partition_usage(tmp_path, fixtures)
            assert receipts == 123
    finally:
        stop.set()
        thread.join()


def test_summary_validator_loads_in_isolated_python_without_project_packages():
    code = f"""
import importlib.util
s=importlib.util.spec_from_file_location('helper',{str(FILE)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p={{'mode':'full_source_gate','expected_tests':1241,'expected_xfail_nodes':[m.KNOWN_XFAIL]}}
t='XFAIL '+m.KNOWN_XFAIL+' - existing strict alias failure\\n1241 passed, 1 xfailed in 1.0s'
m.validate_summary(p,t)
for bad in (t.replace('1241 passed','1240 passed'), t.replace('XFAIL ','XPASS ')):
 try:m.validate_summary(p,bad)
 except ValueError:pass
 else:raise AssertionError('Unexpected source result accepted')
print('stdlib-only summary validator passed')
"""
    r = subprocess.run(
        [sys.executable, "-I", "-c", code], text=True, capture_output=True, check=True
    )
    assert "stdlib-only summary validator passed" in r.stdout
