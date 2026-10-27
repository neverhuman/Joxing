"""Paired side-swapped match runner for frozen controllers (grokbot).

Plays complete diagnostic ``BoxingEnv`` games and returns one plain record per
attempted game. Every attempt is returned, including invalid ones, so a league
can keep its denominators. Records carry ``first``, ``second``, ``first_side``,
``outcome`` (from first's view) and ``invalid_reason``, the fields of
``grokbot.league.Game``.

A controller is ``f(observation, history, rng) -> float32[700]`` in [-1, 1],
the ``BoxingEnv.act`` contract. Its output is checked before physics, so a
malformed command is charged to its corner instead of surfacing as an engine
fault. The scripted panel below is a plumbing and baseline panel. It is not
boxing competence and not a promotion anchor.

Common random numbers: each policy draws from its own stream keyed by
(seed, policy name), not from the env's corner-indexed stream, so both games of a
side-swapped pair feed a stochastic policy identical randomness. Arena games are
reproduced from their seed; an env snapshot alone does not carry these streams.

outcome is the grokbot.endings reference utility (a measured self-caused
fall or exit loses; otherwise points at the bell). bell_outcome keeps the
env's current points-only result for comparison.
"""

from __future__ import annotations

import hashlib

import numpy as np

from grokbot import endings
from joxing.env import PLAYERS
from joxing.observations import FEATURE_SPEC

_HEIGHT = next(b["start"] for b in FEATURE_SPEC if b["name"] == "own_pelvis_height")


def constant(value):
    command = np.full(700, value, np.float32)
    return lambda observation, history, rng: command.copy()


def uniform_noise(observation, history, rng):
    return rng.uniform(-1.0, 1.0, 700).astype(np.float32)


# Excitation sigmoid(5*(c-0.5)): -1 -> 0.0006, 0 -> 0.076, 1 -> 0.924.
SCRIPTED_PANEL = {
    "relaxed": constant(-1.0),
    "baseline": constant(0.0),
    "cocontract": constant(1.0),
    "noise": uniform_noise,
}


class ControllerFault(Exception):
    def __init__(self, player, reason):
        super().__init__(f"{player}: {reason}")
        self.player = player


def policy_stream(seed, name):
    """The policy's own random stream for one seed, independent of its corner."""
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng([seed, key])


def _checked(player, controller, rng):
    def act(observation, history, corner_rng):
        try:
            command = controller(observation, history, rng)
        except Exception as error:
            raise ControllerFault(player, f"{type(error).__name__}: {error}") from error
        if not isinstance(command, np.ndarray) or command.shape != (700,):
            raise ControllerFault(player, "command is not a (700,) array")
        if command.dtype != np.float32:
            raise ControllerFault(player, f"command dtype {command.dtype}, not float32")
        if not np.isfinite(command).all() or np.any(np.abs(command) > 1):
            raise ControllerFault(player, "command is nonfinite or outside [-1, 1]")
        return command

    return act


def corner_result(points_ab, first_side):
    """(first, second) points and first's outcome from referee (A, B) points."""
    a, b = points_ab
    own, other = (a, b) if first_side == "A" else (b, a)
    return [own, other], int(np.sign(own - other))


def _exited(env, bodies, half_size_m):
    xy = env.data.xpos[bodies, :2]
    return bool(np.all(np.abs(xy).max(axis=1) > half_size_m))


def play(env, panel, first, second, first_side, seed, *, low_pelvis_m=0.5):
    """One complete game; never raises for a game-level failure."""
    if first == second:
        raise ValueError("A game needs two distinct panel entries")
    if first_side not in PLAYERS:
        raise ValueError("first_side must be 'A' or 'B'")
    second_side = "B" if first_side == "A" else "A"
    name = {first_side: first, second_side: second}
    controllers = {
        a: _checked(a, panel[name[a]], policy_stream(seed, name[a])) for a in PLAYERS
    }
    observations, info = env.reset(seed=seed)
    bodies = {
        a: [
            b
            for b in range(env.model.nbody)
            if env.model.body(b).name.startswith(a + "_")
        ]
        for a in PLAYERS
    }
    exits = {}
    record = {
        "first": first,
        "second": second,
        "first_side": first_side,
        "seed": seed,
        "contract_hash": info["contract_hash"],
        "horizon_ticks": env.horizon_ticks,
        "ending_variant": endings.VARIANT,
        "outcome": None,
        "bell_outcome": None,
        "ending": None,
        "losses": [],
        "invalid_reason": None,
        "fault_player": None,
        "points": None,
        "ticks": 0,
        "events": 0,
        "pelvis_low_tick": {first: None, second: None},
    }
    try:
        while True:
            for a in PLAYERS:
                height = float(observations[a].values[_HEIGHT])
                if height < low_pelvis_m and record["pelvis_low_tick"][name[a]] is None:
                    record["pelvis_low_tick"][name[a]] = env.tick
                if a not in exits and _exited(env, bodies[a], 2.5):
                    exits[a] = env.tick
            if env.done:
                break
            result = env.act(controllers)
            observations = result["observations"]
            record["events"] += len(result["info"]["events"])
    except ControllerFault as fault:
        record["invalid_reason"] = f"controller fault {fault}"
        record["fault_player"] = name[fault.player]
    except Exception as error:  # engine/referee faults quarantine the game
        record["invalid_reason"] = env.invalid_reason or (
            f"{type(error).__name__}: {error}"
        )
    record["ticks"] = env.tick
    if record["invalid_reason"] is not None:
        return record
    record["points"], record["bell_outcome"] = corner_result(
        env.referee.points, first_side
    )
    registry = env.referee.registry
    touches = endings.contact_ticks(env.evidence, registry)
    losses = endings.support_losses(env.evidence, registry) + [
        endings.LossEvent(a, t, "exit", endings.attribute(touches, t, t))
        for a, t in exits.items()
    ]
    ending = endings.decide(
        env.referee.points,
        losses,
        bell=env.done and not env.truncated,
        bell_tick=env.tick - 1,
    )
    record["losses"] = [
        {
            "policy": name[e.player],
            "tick": e.tick,
            "kind": e.kind,
            "attribution": e.attribution,
        }
        for e in sorted(losses, key=lambda e: (e.tick, e.player))
    ]
    record["ending"] = {"reason": ending.reason, "tick": ending.tick}
    if ending.utility is None:
        record["invalid_reason"] = ending.reason  # blocked or not a sporting end
    else:
        own = ending.utility[0] if first_side == "A" else ending.utility[1]
        record["outcome"] = own
    return record


def paired(env, panel, first, second, seeds, **kwargs):
    """Both corner assignments for every seed, first as A then as B."""
    return [
        play(env, panel, first, second, side, seed, **kwargs)
        for seed in seeds
        for side in PLAYERS
    ]
