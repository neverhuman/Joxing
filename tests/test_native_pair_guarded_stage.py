import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bounded_stage_guards_python_and_cpp_torch_writes_without_physics(tmp_path):
    script = tmp_path / "payload.py"
    script.write_text("""import json,torch
from pathlib import Path
root=Path(__file__).parent
(root/'small.json').write_text(json.dumps({'ok':True}))
try:
 (root/'large.json').write_text('x'*30000)
except TimeoutError:pass
else:raise AssertionError('TEXT credit not enforced')
torch.save({'x':torch.zeros(64)},root/'small.pt')
assert (root/'large.json').stat().st_size==0
print('guarded')
""")
    env = os.environ | {
        n: "1"
        for n in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/native_pair_guarded_stage.py"),
            "--root",
            str(tmp_path),
            "--raw",
            "200000",
            "--text",
            "90000",
            "--file-max",
            "64000",
            "--",
            str(script),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "guarded"
    assert json.loads((tmp_path / "small.json").read_text()) == {"ok": True}
