"""CI guard (owner 02:22): a league whose evaluation yields 0 valid games must
fail, not quietly produce an unratable report. Real physics, tiny runs."""

import pytest

from grokbot import selfplay

TINY = dict(
    generations=1,
    iterations=1,
    pairs=1,
    sigma=0.1,
    lr=0.05,
    eval_seeds=[0],
    workers=2,
    horizon_ticks=2000,
    seed=0,
    panel_eval=False,
    log=lambda row: None,
)


def test_a_league_with_zero_valid_games_fails_fast():
    """Anti-stalling at the native spawn: every bout is a no-contest."""
    with pytest.raises(selfplay.DegenerateLeagueError, match="0 valid games"):
        selfplay.run_league(anti_stalling=True, **TINY)


def test_the_same_league_with_scoreable_endings_passes_the_guard():
    report = selfplay.run_league(anti_stalling=False, **TINY)
    assert report["generations"][0]["eval_valid_share"] > 0
