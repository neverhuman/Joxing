"""Solo motor gate: can one athlete stand and reach at boxing range? (grokbot)

Four league versions (v0.0.11-v0.0.14) failed for one reason: the athletes
fall backward within about 1.5 s and their punch reaches about 0.2 m, so no
glove ever meets a target at speed. A league cannot fix that, so this measures
the missing capability directly, one athlete at a time, before any further
league is preregistered.

A trial runs the unchanged diagnostic env at a named range reset for a full
horizon. Unlike the reference-ending env it does not stop at the first fall.
The candidate plays corner A against a frozen opponent in corner B, and the
trial reports:

* ``stand_ticks``: ticks until the candidate's first persistent non-foot floor
  support (``endings.BoutMonitor``), or the horizon if it never touched down;
* ``strike_ticks``: ticks on which the candidate's glove was striking
  (``ContactClassifier.strikes_by``: approach and referee active speed
  >= 0.5 m/s on head/torso/guard/glove);
* ``scoring_strike_ticks``: the subset on a head or torso target (a hit, not a
  blocked punch; ``ContactClassifier.scoring_by``);
* ``invalid``: an engine fault, which quarantines the trial and never scores.

``train`` is antithetic ES with common random numbers on a continuous fitness:
stand fraction plus a strike indicator. That gives a gradient from the first
iteration, which the league's sparse no-contest outcomes never did.
"""

from __future__ import annotations

import multiprocessing
import os

import numpy as np

from grokbot import endings
from grokbot.arena import policy_stream
from grokbot.scenarios import RangeBoxingEnv
from grokbot.selfplay import _controller, centred_ranks
from joxing.scene import build_scene

_ENV = None


def _init(separation_m, horizon_ticks):
    global _ENV
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    _ENV = RangeBoxingEnv(
        build_scene(), separation_m=separation_m, horizon_ticks=horizon_ticks
    )


def pool(workers, separation_m, horizon_ticks):
    if not 1 <= workers <= 4:
        raise ValueError("Use 1-4 workers on shared hosts")
    return multiprocessing.get_context("fork").Pool(
        workers, initializer=_init, initargs=(separation_m, horizon_ticks)
    )


def trial(task):
    """(candidate spec, opponent spec, seed) -> one solo-gate measurement."""
    candidate, opponent, seed = task
    env = _ENV
    registry = {g["id"]: g for g in env.scene.report["physical_geoms"]}
    monitor = endings.BoutMonitor(registry)
    controllers = {"A": _controller(candidate), "B": _controller(opponent)}
    streams = {
        "A": policy_stream(seed, "candidate"),
        "B": policy_stream(seed, "opponent"),
    }
    observations, _ = env.reset(seed=seed)
    seen = 0
    try:
        while not env.done:
            commands = {
                p: controllers[p](observations[p], None, streams[p]) for p in "AB"
            }
            observations = env.step(commands)["observations"]
            monitor.update(env.evidence[seen:])
            seen = len(env.evidence)
    except Exception as error:  # engine faults quarantine the trial
        return {
            "invalid": f"{type(error).__name__}: {error}",
            "stand_ticks": 0,
            "strike_ticks": 0,
            "scoring_strike_ticks": 0,
            "first_scoring_tick": None,
            "opponent_fall_tick": None,
            "opponent_fall_attribution": None,
            "ticks": env.tick,
        }
    falls = {e.player: e.tick for e in monitor.losses}
    attribution = {e.player: e.attribution for e in monitor.losses}
    return {
        "invalid": None,
        "stand_ticks": falls.get("A", env.horizon_ticks),
        "strike_ticks": len(monitor.contacts.strikes_by["A"]),
        "scoring_strike_ticks": len(monitor.contacts.scoring_by["A"]),
        "first_scoring_tick": min(monitor.contacts.scoring_by["A"], default=None),
        "opponent_fall_tick": falls.get("B"),
        "opponent_fall_attribution": attribution.get("B"),
        "ticks": env.tick,
    }


def hit_and_stayed_up(result, stay_up_ticks):
    """A head/torso hit, and still standing ``stay_up_ticks`` after the first."""
    first = result["first_scoring_tick"]
    return first is not None and result["stand_ticks"] >= first + stay_up_ticks


def knocked_down(result):
    """A head/torso hit, then the opponent falls before the athlete does.

    Ending v2 decides by the first fall and jekko's referee scored 0 points for
    every hit so far, so this is the only decisive result available (v0.0.22).
    """
    first, fall = result["first_scoring_tick"], result["opponent_fall_tick"]
    return (
        first is not None and fall is not None and first < fall < result["stand_ticks"]
    )


def clean_knocked_down(result):
    """``knocked_down``, and the ending attributes the opponent's fall to a
    strike (``knockdown``), not ``unresolved`` after body contact (v0.0.23).

    v0.0.22 lost 0 of 168 valid games but missed its gates on blocked games:
    falls the ending cannot attribute, which its bonus still paid for.
    """
    return knocked_down(result) and result["opponent_fall_attribution"] == "knockdown"


def valid_knocked_down(result):
    """``knocked_down``, and the fall is not left ``unresolved`` by body contact.

    v0.0.23's ``clean_knocked_down`` never fired: the opponent falls 1-2 s
    after the hit, outside the 0.5 s window, so ending v2 attributes the fall
    to the opponent itself (``self``) - a valid win. Only ``unresolved`` falls
    are blocked (invalid), so this pays every valid knockdown (v0.0.24).
    """
    return knocked_down(result) and result["opponent_fall_attribution"] != "unresolved"


def fitness(
    result,
    horizon_ticks,
    strike_bonus,
    strike_key="strike_ticks",
    stay_up_ticks=None,
    knockdown=False,
    clean_knockdown=False,
    valid_knockdown=False,
):
    """Stand fraction plus a strike bonus.

    With ``stay_up_ticks`` (v0.0.19) the bonus needs a head/torso hit after
    which the athlete stays up that long: v0.0.18 hit and then fell first.
    With ``knockdown`` (v0.0.22) it needs ``knocked_down``: v0.0.21's games
    against v0.0.19 were 42/48 bell draws because nobody fell.
    """
    if result["invalid"]:
        return None
    if valid_knockdown:
        earned = valid_knocked_down(result)
    elif clean_knockdown:
        earned = clean_knocked_down(result)
    elif knockdown:
        earned = knocked_down(result)
    elif stay_up_ticks is None:
        earned = result[strike_key] > 0
    else:
        earned = hit_and_stayed_up(result, stay_up_ticks)
    return result["stand_ticks"] / horizon_ticks + strike_bonus * earned


def train(
    workers_pool,
    params,
    opponent,
    rng,
    *,
    kind,
    iterations,
    pairs,
    sigma,
    lr,
    horizon_ticks,
    strike_bonus,
    strike_key="strike_ticks",
    stay_up_ticks=None,
    knockdown=False,
    clean_knockdown=False,
    valid_knockdown=False,
    log=print,
):
    """Antithetic ES on the solo fitness; quarantined pairs are dropped."""
    history = []
    for iteration in range(iterations):
        noise = rng.standard_normal((pairs, len(params)))
        seeds = rng.integers(2**31, size=pairs)
        tasks = [
            ((kind, params + sign * sigma * noise[k]), opponent, int(seeds[k]))
            for k in range(pairs)
            for sign in (1, -1)
        ]
        results = workers_pool.map(trial, tasks)
        scores = [
            fitness(
                r,
                horizon_ticks,
                strike_bonus,
                strike_key,
                stay_up_ticks,
                knockdown,
                clean_knockdown,
                valid_knockdown,
            )
            for r in results
        ]
        valid = np.array(
            [
                scores[2 * k] is not None and scores[2 * k + 1] is not None
                for k in range(pairs)
            ]
        )
        if valid.any():
            flat = np.array([s for s in scores if s is not None], float)
            paired = np.array(
                [[scores[2 * k], scores[2 * k + 1]] for k in range(pairs) if valid[k]]
            )
            ranks = centred_ranks(paired.ravel()).reshape(-1, 2)
            gradient = ((ranks[:, 0] - ranks[:, 1])[:, None] * noise[valid]).sum(0) / (
                2 * valid.sum() * sigma
            )
            params = params + lr * gradient
        row = {
            "iteration": iteration,
            "mean_stand_s": float(
                np.mean([r["stand_ticks"] for r in results if not r["invalid"]]) / 1000
            )
            if valid.any()
            else None,
            "strike_share": float(np.mean([r["strike_ticks"] > 0 for r in results])),
            "scoring_share": float(
                np.mean([r["scoring_strike_ticks"] > 0 for r in results])
            ),
            "hit_and_up_share": float(
                np.mean(
                    [
                        not r["invalid"] and hit_and_stayed_up(r, stay_up_ticks or 0)
                        for r in results
                    ]
                )
            ),
            "invalid": int(sum(r["invalid"] is not None for r in results)),
            "mean_fitness": float(flat.mean()) if valid.any() else None,
        }
        if knockdown or clean_knockdown or valid_knockdown:  # earlier rows unchanged
            row["knockdown_share"] = float(
                np.mean([not r["invalid"] and knocked_down(r) for r in results])
            )
        if valid_knockdown:
            row["valid_knockdown_share"] = float(
                np.mean([not r["invalid"] and valid_knocked_down(r) for r in results])
            )
        if clean_knockdown:
            row["clean_knockdown_share"] = float(
                np.mean([not r["invalid"] and clean_knocked_down(r) for r in results])
            )
        history.append(row)
        log(row)
    return params, history


def gate(workers_pool, spec, opponent, seeds, horizon_ticks):
    """Stance and reach pass rates over noisy solo trials (faults count as fails)."""
    results = workers_pool.map(trial, [(spec, opponent, s) for s in seeds])
    stood = [
        r["invalid"] is None and r["stand_ticks"] >= horizon_ticks for r in results
    ]
    reached = [r["invalid"] is None and r["strike_ticks"] > 0 for r in results]
    scored = [r["invalid"] is None and r["scoring_strike_ticks"] > 0 for r in results]
    return {
        "trials": len(results),
        "stance_pass_rate": float(np.mean(stood)),
        "reach_pass_rate": float(np.mean(reached)),
        "scoring_reach_rate": float(np.mean(scored)),
        "median_stand_s": float(np.median([r["stand_ticks"] for r in results]) / 1000),
        "invalid": int(sum(r["invalid"] is not None for r in results)),
        "results": results,
    }
