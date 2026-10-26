"""The authorized extension must not silently mutate the original experiment."""

import copy
import json

import pytest

from joxing.provenance import ROOT, digest
from joxing.wave import validate_protocol
from joxing.wave_extension import extension_config


def test_extension_budgets_splits_and_parent_are_strict():
    original = json.loads((ROOT / "experiments/v0.0.5/protocol.json").read_text())
    p = json.loads((ROOT / "experiments/v0.0.5/extension-protocol.json").read_text())
    before = digest(original)
    config = extension_config(p)
    assert config["learner"]["steps"] == 160000
    assert config["caps"]["run_wall_s"] == 2700
    assert config["learner"]["workers"] == 4
    assert set(p["screen_ids"]).isdisjoint(p["confirmation_ids"])
    assert set(p["screen_ids"] + p["confirmation_ids"]).isdisjoint(
        [i for bank in original["splits"].values() for i in bank]
    )
    assert digest(original) == before
    assert validate_protocol(original) == original
    for field in (
        "steps",
        "parallel_trials",
        "workers_per_trial",
        "parent_protocol_hash",
        "screen_ids",
        "selection",
        "confirmation_seeds",
        "experiments",
    ):
        bad = copy.deepcopy(p)
        bad[field] = None if field != "experiments" else []
        with pytest.raises((ValueError, TypeError)):
            extension_config(bad)
