"""Data-driven arm reach primitive and a punch-capable policy (grokbot, v0.0.13).

v0.0.12 showed that rules and proximity shaping cannot create punching: the
agents leaned their bodies in instead. This module supplies a motor primitive.

* ``reach_effect`` measures, in the real env, how far each hand moves forward
  and up relative to the actor's own trunk when one DynSyn muscle group
  (msgym ``locomotionFull.json``, 138 groups over all 700 muscles) is driven
  for ``ticks``. ``discover_reach`` ranks the arm groups by forward reach. In
  the native model, groups 58 and 81 (a mirror pair) move the hand about
  0.20 m forward and 0.18 m up in 0.2 s.
* ``PunchPolicy`` is the ``selfplay.LinearPolicy`` posture controller plus, per
  arm, a learned punch: onset, duration and amplitude on that arm's reach
  groups. That adds 6 parameters. All 700 muscles stay driven; the punch only
  overrides its own groups during its window.
* ``active_extension`` is the hand's forward displacement relative to the own
  trunk. It is the shaping quantity that a whole-body lean cannot fake: a lean
  moves hand and trunk together.
"""

from __future__ import annotations

import json

import numpy as np

from grokbot import selfplay
from joxing.provenance import ROOT

GROUPS = [
    np.array(g)
    for g in json.loads(
        (ROOT / "third_party/msgym/DynSyn-SAC/configs/locomotionFull.json").read_text()
    )["agent_kwargs"]["policy_kwargs"]["dynsyn"]
]
ARM = {i for region in ("arm_r", "arm_l", "arm_c") for i in selfplay.REGIONS[region]}
ARM_GROUPS = [k for k, g in enumerate(GROUPS) if set(g.tolist()) <= ARM]
# Ego positions of own hand_r, hand_l and thoracic12 (KEY_BODIES 3, 4, 2).
HAND = {"r": 858 + 9, "l": 858 + 12}
TRUNK = 858 + 6
REACH_GROUPS = {"r": (57, 58), "l": (80, 81)}  # from discover_reach; see tests
PUNCH_PARAMS = 6
MAX_ONSET_TICKS = 600
MAX_DURATION_TICKS = 600


def hand_offsets(values):
    """Each hand's ego position relative to the own trunk."""
    trunk = values[TRUNK : TRUNK + 3]
    return {s: values[i : i + 3] - trunk for s, i in HAND.items()}


def active_extension(start_values, values):
    """Largest forward (ego +x) hand displacement relative to the own trunk."""
    a, b = hand_offsets(start_values), hand_offsets(values)
    return float(max(b[s][0] - a[s][0] for s in HAND))


def reach_effect(env, groups, *, ticks=200, seed=0, amplitude=1.0):
    """(right, left) hand displacement vs trunk relative to a zero command."""

    def run(command):
        observations, _ = env.reset(seed=seed)
        start = observations["A"].values.copy()
        for _ in range(ticks // env.control_ticks):
            observations = env.step({"A": command, "B": np.zeros(700, np.float32)})[
                "observations"
            ]
        before, after = hand_offsets(start), hand_offsets(observations["A"].values)
        return {s: after[s] - before[s] for s in HAND}

    base = run(np.zeros(700, np.float32))
    command = np.zeros(700, np.float32)
    for k in groups:
        command[GROUPS[k]] = amplitude
    moved = run(command)
    return {s: moved[s] - base[s] for s in HAND}


def discover_reach(env, *, top=2, **kwargs):
    """Arm groups ranked by forward reach for each hand, best first."""
    effects = {k: reach_effect(env, [k], **kwargs) for k in ARM_GROUPS}
    return {
        s: sorted(effects, key=lambda k: -effects[k][s][0])[:top] for s in HAND
    }, effects


class PunchPolicy:
    """LinearPolicy posture control plus a learned punch per arm.

    ``params`` is the 140 LinearPolicy parameters followed by, per arm (r then
    l), raw (onset, duration, amplitude) squashed to [0, MAX_ONSET_TICKS],
    [0, MAX_DURATION_TICKS] and [-1, 1].
    """

    def __init__(self, params, noise=0.0):
        params = np.asarray(params, float)
        if params.shape != (selfplay.N_PARAMS + PUNCH_PARAMS,):
            raise ValueError(f"Expected {selfplay.N_PARAMS + PUNCH_PARAMS} parameters")
        self.posture = selfplay.LinearPolicy(params[: selfplay.N_PARAMS], noise)
        raw = params[selfplay.N_PARAMS :].reshape(2, 3)
        squash = 1 / (1 + np.exp(-raw))
        self.punches = {
            side: (
                squash[i, 0] * MAX_ONSET_TICKS,
                squash[i, 1] * MAX_DURATION_TICKS,
                2 * squash[i, 2] - 1,
            )
            for i, side in enumerate(("r", "l"))
        }

    def __call__(self, observation, history=None, rng=None):
        command = self.posture(observation, history, rng)
        t = observation.sampled_tick
        for side, (onset, duration, amplitude) in self.punches.items():
            if onset <= t < onset + duration:
                for k in REACH_GROUPS[side]:
                    command[GROUPS[k]] = np.float32(amplitude)
        return command


def punch_params(posture, *, onset=0.0, duration=0.0, amplitude=0.0):
    """Concatenate posture params with identical raw punch params for both arms."""
    raw = np.array([onset, duration, amplitude] * 2, float)
    return np.concatenate([np.asarray(posture, float), raw])
