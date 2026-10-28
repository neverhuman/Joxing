import copy
from dataclasses import asdict

import pytest

from joxing.referee import GloveInterval
from joxing.rules import HitTracker, Rules, Touch
from joxing.score_adapter_fixture import run_fixture
from joxing.score_diagnostic import audit_saved_trace, explain_interval, new_tracker


def touch(
    zone="head", force=100.0, speed=2.0, active=2.0, face=1.0, sample="c", striking=True
):
    return Touch(sample, zone, force, speed, active, face, striking)


def run_contacts(schedule, count=100):
    rules = Rules()
    production = HitTracker("test", "A_glove_r", rules)
    independent = new_tracker("test", "A_glove_r", rules)
    explanations = []
    for tick in range(count):
        contacts, clear, extension = schedule(tick)
        row = GloveInterval(tuple(contacts), clear, extension)
        before, measured = copy.deepcopy(independent), asdict(row)
        result = explain_interval(row, independent, tick, rules)
        assert independent == before and asdict(row) == measured
        expected = production.update(
            tick, contacts, clearance_m=clear, extension_m=extension
        )
        assert result["events"] == expected
        assert result["after"] == production.snapshot()
        independent = result["after"]
        explanations.append(result["explanation"])
    return production, explanations


@pytest.mark.parametrize(
    "zone,award", [("head", 2), ("torso", 1), ("guard", 0), ("native_non_scoring", 0)]
)
def test_physical_impulse_ownership_and_single_encounter(zone, award):
    tracker, _ = run_contacts(lambda k: ([touch(zone)], 0.0, 0.4))
    assert len(tracker.ledger) == 1
    assert next(iter(tracker.ledger.values()))["points"] == award


@pytest.mark.parametrize("force,award", [(250.0, 2), (249.999, 0)])
def test_exact_qualifying_impulse_boundary(force, award):
    tracker, _ = run_contacts(
        lambda k: ([touch(force=force)] if k == 0 else [], 0.0, 0.4)
    )
    assert next(iter(tracker.ledger.values()))["points"] == award


def test_eligible_onset_is_not_rejected_by_later_deceleration():
    tracker, explanations = run_contacts(
        lambda k: (
            [touch(speed=2.0 if k == 0 else 0.0, active=2.0 if k == 0 else 0.0)],
            0.0,
            0.4,
        )
    )
    assert next(iter(tracker.ledger.values()))["points"] == 2
    later = explanations[1]["samples"][0]
    assert not later["instantaneous_predicates"]["approach"]
    assert not later["instantaneous_predicates"]["active_speed"]
    assert later["latched_onset_eligible"] and later["effective_rejections"] == []


@pytest.mark.parametrize(
    "change", [{"speed": 0.0}, {"active": 0.0}, {"face": 0.0}, {"striking": False}]
)
def test_bad_onset_stays_ineligible_and_reports_current_predicates(change):
    tracker, explanations = run_contacts(
        lambda k: ([touch(**change)] if k == 0 else [touch()], 0.0, 0.4)
    )
    assert next(iter(tracker.ledger.values()))["points"] == 0
    assert "onset_ineligible" in explanations[1]["samples"][0]["effective_rejections"]
    assert not explanations[1]["samples"][0]["onset_cause_available"]


def test_guard_does_not_lend_impulse_or_onset_and_duplicates_do_not_pay():
    def schedule(k):
        if k == 0:
            return [touch("guard", force=10000.0)], 0.0, 0.4
        if k < 4:
            return [touch("torso"), touch("torso")], 0.0, 0.4
        return [], 0.0, 0.4

    tracker, explanations = run_contacts(schedule)
    event = next(iter(tracker.ledger.values()))
    assert event["points"] == 1 and event["direct_impulses_Ns"][
        "torso"
    ] == pytest.approx(0.3)
    assert event["guard_window_impulse_Ns"] == 10.0
    assert explanations[1]["duplicate_samples"] == 1


def test_multizone_tie_prefers_lower_points_and_cannot_pool_subthreshold_impulse():
    tracker, _ = run_contacts(
        lambda k: (
            [
                touch("head", force=200, sample="h"),
                touch("torso", force=200, sample="t"),
            ]
            if k == 0
            else [],
            0.0,
            0.4,
        )
    )
    event = next(iter(tracker.ledger.values()))
    assert event["zone"] == "torso" and event["points"] == 0


def test_geometric_clearance_and_retraction_are_both_required_to_rearm():
    def schedule(k):
        if k < 50 or k >= 131:
            return [touch()], 0.0, 0.4
        return [], 0.03, 0.4 if k < 100 else 0.35

    tracker, explanations = run_contacts(schedule, 181)
    assert len(tracker.ledger) == 2
    assert not explanations[99]["recovered"]
    assert explanations[100]["recovered"]


def test_bell_cannot_create_impulse_and_does_not_mutate_snapshot():
    rules = Rules()
    state = new_tracker("test", "A_glove_r", rules)
    state = explain_interval(
        GloveInterval((touch(force=200),), 0.0, 0.4), state, 0, rules
    )["after"]
    before = copy.deepcopy(state)
    result = explain_interval(
        GloveInterval((touch(force=10000),), 0.0, 0.4), state, 1, rules, bell=True
    )
    assert result["events"][0]["points"] == 0
    assert result["events"][0]["direct_impulses_Ns"]["head"] == 0.2
    assert state == before
    retry = explain_interval(GloveInterval((), 0.0, 0.4), result["after"], 1, rules)
    assert retry["events"] == [] and retry["explanation"]["duplicate_tick"]


@pytest.mark.parametrize("fault", ["gap", "reverse", "nan", "conflicting", "rules"])
def test_invalid_evidence_is_rejected_after_a_passing_baseline(fault):
    rules = Rules()
    state = new_tracker("test", "A_glove_r", rules)
    row = asdict(GloveInterval((touch(),), 0.0, 0.4))
    state = explain_interval(row, state, 0, rules)["after"]
    assert explain_interval(row, state, 1, rules)["after"]["last_tick"] == 1
    tick = 1
    if fault == "gap":
        tick = 2
    elif fault == "reverse":
        tick = -1
    elif fault == "nan":
        row["touches"][0]["active_speed_mps"] = float("nan")
    elif fault == "conflicting":
        row["touches"] += ({**row["touches"][0], "normal_force_N": 1.0},)
    else:
        state["rules"]["head_points"] += 1
    with pytest.raises(ValueError):
        explain_interval(row, state, tick, rules)


@pytest.mark.parametrize("zone", ["head", "torso"])
@pytest.mark.parametrize("reverse", [False, True])
def test_real_adapter_scores_and_independent_audit_rejects_changed_measurements(
    zone, reverse, monkeypatch
):
    result = run_fixture(zone, reverse)
    assert result["points"] == [2 if zone == "head" else 1, 0]
    assert result["physics_steps"] == 250 and result["native_physics_steps"] == 0
    monkeypatch.setattr(
        HitTracker,
        "update",
        lambda *a, **kw: pytest.fail("Production scorer called by independent audit"),
    )
    assert (
        audit_saved_trace(
            result["initial_referee"], result["final_referee"], result["intervals"], 250
        )["status"]
        == "passed"
    )
    changed = copy.deepcopy(result["intervals"])
    first = next(g for r in changed for g in r["gloves"].values() if g["touches"])
    first["touches"][0]["active_speed_mps"] = 0.0
    with pytest.raises(ValueError, match="events differ|snapshot differs"):
        audit_saved_trace(
            result["initial_referee"], result["final_referee"], changed, 250
        )


@pytest.mark.parametrize("failed_stage", ["saved", "fixture"])
def test_qualification_stops_and_retains_every_attempt_after_a_fault(
    tmp_path, monkeypatch, failed_stage
):
    import json

    from joxing import score_qualification as q

    root = tmp_path / "root"
    summary = root / "grokbot/redteam/data/cycle-0029-mirror-opponent-gate-summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "primary_vs_v0019": {"games_detail": [{} for _ in range(48)]},
                "referee_points_check": {"referee_points": [0, 0]},
            }
        )
    )

    def saved(*args):
        if failed_stage == "saved":
            raise ValueError("missing saved onset evidence")
        return {"intervals": 1, "final_sha256": "synthetic", "rejections": {}}

    def fixture(*args, **kwargs):
        raise RuntimeError("physical fixture unavailable")

    monkeypatch.setattr(q, "peak_rss", lambda: 0)  # synthetic lifecycle test only
    monkeypatch.setattr(q, "saved", saved)
    monkeypatch.setattr(q, "run_fixture", fixture)
    p = {
        "raw_bytes": 4 * 1024**2,
        "maximum_rss_bytes": 1024**3,
        "stage_wall_limits_s": {"proof": 10},
        "fixture_physics_limit": 1000,
    }
    out = tmp_path / "result"
    result = q.qualify(p, root, out, "synthetic-source", "synthetic-protocol")
    failed_index = 0 if failed_stage == "saved" else 3
    assert result["status"] == "halted"
    assert [r["id"] for r in result["rows"]] == q.ROW_IDS
    assert result["rows"][failed_index]["status"] == "failed"
    assert all(r["status"] == "not_run" for r in result["rows"][failed_index + 1 :])
    assert result["fixture_entered"] == result["fixture_completed"] == 0
    assert (out / "fault.json").exists()
