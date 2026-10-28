"""Two-player self-play league on the reference ending (grokbot; DIAGNOSTIC).

This is a real self-play learning loop on the shared 1,400-muscle world. It is
NOT a qualified boxing result: the AGENTS.md contact, rules, stable-motor and
replay gates are unmet, there is no registered boxing-range reset, and from
the native keyframe (1.6 m apart) the athletes cannot reach each other. Under
the reference ending the learnable objective is therefore "outlast the
opponent", which is balance, not boxing. Reports must say so.

* Policy: deterministic, linear. Each of ``len(REGIONS)`` muscle regions gets
  ``tone + gains . (features - features_at_reset)``. The 13 features are
  egocentric pelvis/trunk tilt, pelvis velocity and height from the actor's
  own observation, never opponent state. The result is clipped to [-1, 1] and
  broadcast to the region's muscles (all 700 stay driven).
* Learner: antithetic evolution strategies (OpenAI-ES style, centred ranks).
  Each antithetic pair shares opponent, seed and corner, which gives common
  random numbers. Fitness is the reference-ending utility of the game
  (``grokbot.reference_env``). Blocked bouts score 0 and are counted.
* League: opponents come from ``grokbot.opponents``. The latest checkpoint is
  self-play, archived checkpoints are weighted by PFSP from the learner's
  record, and the scripted panel serves as anchors. Every generation is
  frozen and archived, including weaker ones.
* Evaluation: frozen side-swapped pairs against the previous checkpoint and
  every scripted panel member, played to the reference ending, feed
  ``grokbot.league.Ledger`` Elo (anchored at the scripted ``baseline``) and the
  ``grokbot.promotion`` gate.

Workers are separate processes (``fork``), each with one thread and its own
scene. The caller sets ``nice``.
"""

from __future__ import annotations

import multiprocessing
import os

import mujoco
import numpy as np

from grokbot import opponents
from grokbot.arena import SCRIPTED_PANEL, policy_stream
from grokbot.reference_env import ReferenceEndingEnv
from grokbot.scenarios import RangeBoxingEnv
from joxing.env import PLAYERS, BoxingEnv
from joxing.scene import SOURCE, build_scene

# Own hands (KEY_BODIES 3, 4) to opponent head and trunk (1, 2), both in the
# actor's own ego frame: training-only approach shaping, never evaluated.
OWN_HANDS = [858 + 3 * k for k in (3, 4)]
OPPONENT_TARGETS = [1056 + 3 * k for k in (1, 2)]
FEATURES = np.array(
    [930, 931, 932, 948, 949, 950, 891, 892, 893, 1023, 1024, 1025, 1257]
)  # pelvis/trunk z-row of rotation (tilt), pelvis lin/ang velocity, height
LEG = ("femur", "tibia", "talus", "calcn", "toes", "patella", "fibula")
ARM = ("humerus", "ulna", "radius", "hand", "scapula", "clavicle")


def _regions():
    """Native actuator indices grouped by attachment bodies and side."""
    model = mujoco.MjModel.from_xml_path(str(SOURCE))
    groups = {}
    for i in range(model.nu):
        tendon = model.actuator_trnid[i, 0]
        bodies = set()
        for k in range(
            model.tendon_adr[tendon],
            model.tendon_adr[tendon] + model.tendon_num[tendon],
        ):
            kind, obj = model.wrap_type[k], model.wrap_objid[k]
            if kind == mujoco.mjtWrap.mjWRAP_SITE:
                bodies.add(model.body(model.site_bodyid[obj]).name)
            elif kind in (mujoco.mjtWrap.mjWRAP_SPHERE, mujoco.mjtWrap.mjWRAP_CYLINDER):
                bodies.add(model.body(model.geom_bodyid[obj]).name)
        names = " ".join(sorted(bodies)).lower()
        name = model.actuator(i).name.lower()
        side = "_r" if name.endswith("_r") else "_l" if name.endswith("_l") else "_c"
        part = (
            "leg"
            if any(k in names for k in LEG)
            else "arm"
            if any(k in names for k in ARM)
            else "neck"
            if any(k in names for k in ("head", "cervical"))
            else "trunk"
        )
        groups.setdefault(part + side, []).append(i)
    return {k: np.array(v) for k, v in sorted(groups.items())}


REGIONS = _regions()
N_PARAMS = len(REGIONS) * (1 + len(FEATURES))


class LinearPolicy:
    """Linear controller: params -> muscle commands; obeys env.act().

    Deterministic unless ``noise`` > 0, which adds seeded Gaussian noise to every
    regional command from the game's own per-policy generator. Evaluation uses
    it: a deterministic policy in a deterministic world plays the same game for
    every seed, so repeated seeds would not be independent samples.
    """

    def __init__(self, params, noise=0.0):
        params = np.asarray(params, float)
        if params.shape != (N_PARAMS,) or not np.isfinite(params).all():
            raise ValueError(f"Expected {N_PARAMS} finite parameters")
        table = params.reshape(len(REGIONS), 1 + len(FEATURES))
        self.tone, self.gains = table[:, 0], table[:, 1:]
        self.params = params
        self.noise = float(noise)
        self.origin = None

    def __call__(self, observation, history=None, rng=None):
        features = observation.values[FEATURES].astype(float)
        if self.origin is None or observation.sampled_tick == 0:
            self.origin = features
        regional = self.tone + self.gains @ (features - self.origin)
        if self.noise:
            if rng is None:
                raise ValueError("A noisy policy needs the game's generator")
            regional = regional + rng.normal(0.0, self.noise, len(regional))
        regional = np.clip(regional, -1, 1)
        command = np.empty(700, np.float32)
        for value, ids in zip(regional, REGIONS.values(), strict=True):
            command[ids] = value
        return command


def random_params(rng, tone=0.5):
    """Random tones in [-tone, tone], zero feedback: the untrained policy."""
    table = np.zeros((len(REGIONS), 1 + len(FEATURES)))
    table[:, 0] = rng.uniform(-tone, tone, len(REGIONS))
    return table.ravel()


# ---------------------------------------------------------------- workers --
_ENV = None


def _init(horizon_ticks, separation_m=None, anti_stalling=False):
    global _ENV
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    scene = build_scene()
    env = (
        BoxingEnv(scene, horizon_ticks=horizon_ticks)
        if separation_m is None
        else RangeBoxingEnv(
            scene, separation_m=separation_m, horizon_ticks=horizon_ticks
        )
    )
    _ENV = ReferenceEndingEnv(env, anti_stalling=anti_stalling)


def reach(values):
    """Closest own hand to the opponent's head or trunk, in metres."""
    hands = [values[i : i + 3] for i in OWN_HANDS]
    targets = [values[i : i + 3] for i in OPPONENT_TARGETS]
    return float(min(np.linalg.norm(h - t) for h in hands for t in targets))


def _controller(spec):
    kind, value, *noise = spec
    if kind == "params":
        return LinearPolicy(value, noise[0] if noise else 0.0)
    if kind == "punch":
        from grokbot.motor_primitives import PunchPolicy

        return PunchPolicy(value, noise[0] if noise else 0.0)
    return SCRIPTED_PANEL[value]


def play_game(task):
    """(learner spec, opponent spec, learner corner, seed) -> learner utility."""
    learner, opponent, side, seed = task
    env = _ENV
    other = "B" if side == "A" else "A"
    controllers = {side: _controller(learner), other: _controller(opponent)}
    # Per-policy streams (common random numbers); linear policies ignore them.
    label = opponent[1] if opponent[0] == "scripted" else "opponent"
    from grokbot.motor_primitives import active_extension

    streams = {side: policy_stream(seed, "learner"), other: policy_stream(seed, label)}
    observations, _ = env.reset(seed=seed)
    start = closest = reach(observations[side].values)
    first = observations[side].values.copy()
    extension = 0.0
    while True:
        commands = {
            p: controllers[p](observations[p], None, streams[p]) for p in PLAYERS
        }
        try:
            result = env.step(commands)
        except Exception as error:  # engine faults quarantine the game
            return {
                "utility": 0.0,
                "invalid": f"{type(error).__name__}: {error}",
                "blocked": False,
                "no_contest": False,
                "tick": env.env.tick,
                "contact_ticks": 0,
                "strike_ticks": 0,
                "opponent_strike_ticks": 0,
                "points": None,
                "approach": 0.0,
                "extension": 0.0,
            }
        observations = result["observations"]
        closest = min(closest, reach(observations[side].values))
        extension = max(extension, active_extension(first, observations[side].values))
        if result["terminated"]:
            ending = result["info"]["ending"]
            no_result = ending["utility"] is None
            contacts = env.monitor.contacts
            return {
                "utility": 0.0
                if no_result
                else float(ending["utility"][PLAYERS.index(side)]),
                "invalid": None,
                "blocked": result["info"]["blocked"],
                "no_contest": result["info"]["no_contest"],
                "tick": env.env.tick,
                "contact_ticks": len(contacts.strikes) + len(contacts.blocking),
                "strike_ticks": len(contacts.strikes_by[side]),
                "opponent_strike_ticks": len(contacts.strikes_by[other]),
                "points": list(env.env.referee.points),
                "approach": (start - closest) / start if start > 0 else 0.0,
                "extension": extension,
            }


def _pool(workers, horizon_ticks, separation_m=None, anti_stalling=False):
    return multiprocessing.get_context("fork").Pool(
        workers,
        initializer=_init,
        initargs=(horizon_ticks, separation_m, anti_stalling),
    )


# ---------------------------------------------------------------- learner --
def centred_ranks(values):
    values = np.asarray(values, float)
    ranks = np.empty(len(values))
    ranks[np.argsort(values, kind="stable")] = np.arange(len(values))
    # Ties share their mean rank so a flat fitness gives a zero update.
    for v in np.unique(values):
        ranks[values == v] = ranks[values == v].mean()
    return ranks / max(len(values) - 1, 1) - 0.5


def train_generation(
    pool,
    params,
    archive,
    records,
    rng,
    *,
    iterations,
    pairs,
    sigma,
    lr,
    eps=0.1,
    shaping_weight=0.0,
    shaping_key="approach",
    strike_bonus=0.0,
    kind="params",
    frozen=None,
):
    """ES iterations against the P4 league; returns new params and stats.

    ``frozen`` maps extra anchor names to ``(kind, params)`` opponents (e.g. a
    previous version's champion) sampled in the anchors bucket.
    """
    frozen = frozen or {}
    stats = {"games": 0, "blocked": 0, "mean_fitness": []}
    for _ in range(iterations):
        history = opponents.history_probabilities(records, eps=eps) if records else {}
        mix = opponents.mixture(
            latest=["latest"],
            history=history,
            anchors=list(SCRIPTED_PANEL) + list(frozen),
        )
        noise = rng.standard_normal((pairs, len(params)))
        tasks, meta = [], []
        for k in range(pairs):
            name = opponents.sample(mix, rng)
            spec = (
                (kind, params)
                if name == "latest"
                else (kind, archive[name])
                if name in archive
                else frozen[name]
                if name in frozen
                else ("scripted", name)
            )
            side = "A" if k % 2 == 0 else "B"
            seed = int(rng.integers(2**31))
            for sign in (1, -1):
                tasks.append(
                    ((kind, params + sign * sigma * noise[k]), spec, side, seed)
                )
            meta.append(name)
        results = pool.map(play_game, tasks)
        # Training-only approach shaping: reaching toward the opponent's head or
        # trunk scores up to ``shaping_weight``; standing off scores nothing.
        # Training-only terms: shaping on ``shaping_key`` (proximity "approach"
        # or trunk-relative hand "extension") and a bonus for landing a strike.
        fitness = np.array(
            [
                r["utility"]
                + shaping_weight * max(r[shaping_key], 0.0)
                + strike_bonus * (r["strike_ticks"] > 0)
                for r in results
            ]
        )
        stats["games"] += len(results)
        stats["invalid"] = stats.get("invalid", 0) + sum(
            r["invalid"] is not None for r in results
        )
        stats["blocked"] += sum(r["blocked"] for r in results)
        stats["contact_ticks"] = stats.get("contact_ticks", 0) + sum(
            r["contact_ticks"] for r in results
        )
        stats.setdefault("end_ticks", []).extend(r["tick"] for r in results)
        stats["mean_approach"] = float(np.mean([r["approach"] for r in results]))
        stats["no_contest"] = stats.get("no_contest", 0) + sum(
            r["no_contest"] for r in results
        )
        stats["mean_fitness"].append(float(fitness.mean()))
        for k, name in enumerate(meta):
            if name in archive:
                for u in fitness[2 * k : 2 * k + 2]:
                    wins, games = records.get(name, (0.0, 0.0))
                    records[name] = (wins + (u + 1) / 2, games + 1)
        # A pair with a quarantined (engine-fault) game carries no comparison:
        # drop it from the update instead of scoring the fault as a result.
        valid = np.array(
            [
                results[2 * k]["invalid"] is None
                and results[2 * k + 1]["invalid"] is None
                for k in range(pairs)
            ]
        )
        if valid.any():
            ranks = centred_ranks(fitness.reshape(pairs, 2)[valid].ravel())
            shaped = ranks.reshape(-1, 2)
            gradient = ((shaped[:, 0] - shaped[:, 1])[:, None] * noise[valid]).sum(
                0
            ) / (2 * valid.sum() * sigma)
            params = params + lr * gradient
    return params, stats


def finite_or_none(value):
    return float(value) if value is not None and np.isfinite(value) else None


def safe_ratings(ledger, anchor):
    """Elo by name, or {} when there is no valid game or no common scale.

    A league whose bouts are all no-contests has no ratings at all; that is a
    result to report, not a crash (v0.0.12 attempt 01).
    """
    try:
        ratings = ledger.ratings()  # centred on the rated players
    except ValueError:
        return {}
    if np.isfinite(ratings[ledger.index[anchor]]):
        ratings = ratings - ratings[ledger.index[anchor]]
    return {
        n: round(float(ratings[ledger.index[n]]), 1)
        for n in ledger.players
        if np.isfinite(ratings[ledger.index[n]])
    }


def evaluation_tasks(candidate, archive, opponent_names, seeds, noise, kind="params"):
    """Side-swapped pairs: candidate checkpoint vs each named opponent.

    ``archive`` values are parameter vectors of policy ``kind``, or
    ``(kind, params)`` tuples for opponents of another kind.
    """

    def spec_of(name):
        value = archive[name]
        if isinstance(value, tuple):
            return (value[0], value[1], noise)
        return (kind, value, noise)

    spec = spec_of(candidate)
    tasks = []
    for name in opponent_names:
        other = spec_of(name) if name in archive else ("scripted", name)
        for seed in seeds:
            for side in PLAYERS:
                tasks.append(((candidate, name, side, seed), (spec, other, side, seed)))
    return tasks


def run_league(
    *,
    generations,
    iterations,
    pairs,
    sigma,
    lr,
    eval_seeds,
    workers,
    horizon_ticks,
    seed,
    panel_eval=True,
    eval_noise=0.05,
    separation_m=None,
    anti_stalling=False,
    shaping_weight=0.0,
    min_engaged_share=0.0,
    init_params=None,
    policy_kind="params",
    shaping_key="approach",
    strike_bonus=0.0,
    frozen=None,
    log=print,
):
    """Train ``generations`` frozen checkpoints; evaluate each one as it lands."""
    from grokbot.league import Game, Ledger
    from grokbot.promotion import GateConfig, _utilities, decide, hoeffding_lower

    if not 1 <= workers <= 4:
        raise ValueError("Use 1-4 workers on shared hosts")
    rng = np.random.default_rng(seed)
    names = [f"gen{g}" for g in range(generations + 1)]
    archive = {
        "gen0": random_params(rng)
        if init_params is None
        else np.asarray(init_params, float).copy()
    }
    records, games, report = {}, [], {"generations": []}
    params = archive["gen0"].copy()
    frozen = frozen or {}
    panel = (list(SCRIPTED_PANEL) if panel_eval else []) + list(frozen)
    lookup = dict(archive)

    def evaluate(pool, candidate, opponent_names):
        lookup.update(archive)
        lookup.update(frozen)
        tasks = evaluation_tasks(
            candidate, lookup, opponent_names, eval_seeds, eval_noise, policy_kind
        )
        outcomes = pool.map(play_game, [t for _, t in tasks])
        for ((cand, opp, side, s), _), out in zip(tasks, outcomes, strict=True):
            games.append(
                {
                    "first": cand,
                    "second": opp,
                    "first_side": side,
                    "seed": s,
                    "outcome": None
                    if out["blocked"] or out["no_contest"] or out["invalid"]
                    else int(out["utility"]),
                    "invalid_reason": out["invalid"]
                    if out["invalid"]
                    else "blocked: unresolved loss attribution"
                    if out["blocked"]
                    else "no contest: no engagement before the ending"
                    if out["no_contest"]
                    else None,
                    "fault_player": None,
                    "losses": [],
                    "ending": None,
                    "end_tick": out["tick"],
                    "contact_ticks": out["contact_ticks"],
                    "no_contest": out["no_contest"],
                    "engaged": {
                        cand: out["strike_ticks"] > 0,
                        opp: out["opponent_strike_ticks"] > 0,
                    },
                }
            )

    champion = "gen0"
    with _pool(workers, horizon_ticks, separation_m, anti_stalling) as pool:
        if panel:
            evaluate(pool, "gen0", panel)
        for g in range(1, generations + 1):
            params, stats = train_generation(
                pool,
                params,
                {k: v for k, v in archive.items()},
                records,
                rng,
                iterations=iterations,
                pairs=pairs,
                sigma=sigma,
                lr=lr,
                shaping_weight=shaping_weight,
                shaping_key=shaping_key,
                strike_bonus=strike_bonus,
                kind=policy_kind,
                frozen=frozen,
            )
            archive[f"gen{g}"] = params.copy()
            # Every earlier frozen checkpoint, not just the previous one: beating
            # the latest while losing to an older snapshot is cycling. Promotion
            # is decided against the champion, one of these.
            earlier = [f"gen{k}" for k in range(g)]
            evaluate(pool, f"gen{g}", [*earlier, *panel])
            ledger = Ledger(names[: g + 1] + panel)
            for r in games:
                ledger.add(
                    Game(
                        r["first"],
                        r["second"],
                        r["first_side"],
                        r["outcome"],
                        r["invalid_reason"],
                    )
                )
            mine = [r for r in games if r["first"] == f"gen{g}"]
            this_gen = Ledger(names[: g + 1] + panel)
            for r in mine:
                this_gen.add(
                    Game(
                        r["first"],
                        r["second"],
                        r["first_side"],
                        r["outcome"],
                        r["invalid_reason"],
                    )
                )

            def mean_vs(name, rows=mine):
                vals = [
                    r["outcome"]
                    for r in rows
                    if r["second"] == name and r["outcome"] is not None
                ]
                return float(np.mean(vals)) if vals else None

            elo = safe_ratings(ledger, "baseline" if panel else "gen0")
            gate = decide(
                games,
                f"gen{g}",
                champion,
                [champion, *panel],
                GateConfig(
                    min_valid=len(eval_seeds), min_engaged_share=min_engaged_share
                ),
            )
            previous_champion = champion
            if gate.promote:
                champion = f"gen{g}"
            row = {
                "generation": g,
                "train_games": stats["games"],
                "train_blocked": stats["blocked"],
                "train_invalid": stats.get("invalid", 0),
                "train_mean_fitness": stats["mean_fitness"],
                "vs_previous": mean_vs(f"gen{g - 1}"),
                "vs_earlier": {n: mean_vs(n) for n in earlier},
                "beats_all_earlier": all((mean_vs(n) or 0) > 0 for n in earlier),
                "champion_before": previous_champion,
                "vs_champion": mean_vs(previous_champion),
                "vs_scripted": {n: mean_vs(n) for n in panel},
                "vs_frozen": {
                    n: {
                        "pairs": len(samples),
                        "mean": float(np.mean(samples)) if samples else None,
                        "lower95": hoeffding_lower(
                            float(np.mean(samples)) if samples else 0.0,
                            len(samples),
                            0.95,
                        ),
                    }
                    for n in frozen
                    for samples in [_utilities(games, f"gen{g}", n)]
                },
                "train_contact_ticks": stats["contact_ticks"],
                "train_median_end_tick": float(np.median(stats["end_ticks"])),
                "elo": elo,
                "side_balance": finite_or_none(ledger.side_balance()),
                "side_balance_this_gen": finite_or_none(this_gen.side_balance()),
                "eval_contact_ticks": sum(r["contact_ticks"] for r in mine),
                "eval_engaged_share": float(
                    np.mean([r["engaged"][f"gen{g}"] for r in mine])
                ),
                "eval_no_contest_share": float(
                    np.mean([r["no_contest"] for r in mine])
                ),
                "train_mean_approach": stats.get("mean_approach"),
                "promoted": gate.promote,
                "champion_after": champion,
                "gate_reasons": list(gate.reasons),
                "params_norm": float(np.linalg.norm(params)),
            }
            report["generations"].append(row)
            log(row)
    report["champion"] = champion
    report["archive"] = {k: v.tolist() for k, v in archive.items()}
    report["evaluation_games"] = games
    return report
