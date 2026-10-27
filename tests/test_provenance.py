from joxing import provenance


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
