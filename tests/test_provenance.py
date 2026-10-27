import subprocess
from pathlib import Path

from joxing import provenance
from joxing.provenance import revision


def test_plain_directory_inside_git_is_not_an_independent_revision(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    nested = tmp_path / "copied-submodule"
    nested.mkdir()
    assert revision(nested) == "uncommitted"


def test_source_identity_includes_nested_body_adapter(monkeypatch, tmp_path):
    source = tmp_path / "src/joxing"
    (source / "bodies").mkdir(parents=True)
    (source / "env.py").write_text("# same environment\n")
    adapter = source / "bodies/native.py"
    adapter.write_text("# native adapter v1\n")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    initial = provenance.source_hash()
    adapter.write_text("# different native adapter\n")
    assert provenance.source_hash() != initial
