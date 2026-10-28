from joxing.native_strike_wave import chain_stages


def test_zero_exit_does_not_hide_a_halted_scoring_supervisor():
    entered = []

    def execute(row):
        entered.append(row["name"])
        return {"returncode": 0, "supervisor_status": "halted"}

    rows = chain_stages(
        [{"name": "scoring"}, {"name": "native"}], execute, lambda rows: None
    )
    assert entered == ["scoring"]
    assert [r["status"] for r in rows] == ["failed", "not_run"]


def test_native_stage_activates_only_after_completed_prerequisite():
    entered = []

    def execute(row):
        entered.append(row["name"])
        return {"returncode": 0, "supervisor_status": "completed"}

    rows = chain_stages(
        [{"name": "scoring"}, {"name": "native"}], execute, lambda rows: None
    )
    assert entered == ["scoring", "native"]
    assert all(r["status"] == "completed" for r in rows)
