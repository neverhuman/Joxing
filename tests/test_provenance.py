from pathlib import Path
import subprocess

from joxing.provenance import revision


def test_plain_directory_inside_git_is_not_an_independent_revision(tmp_path: Path):
    subprocess.run(["git","init","-q",str(tmp_path)],check=True)
    nested=tmp_path/"copied-submodule"; nested.mkdir()
    assert revision(nested)=="uncommitted"
