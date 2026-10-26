"""v0.0.2 assisted muscle-to-contact experiment, never a boxing environment."""

from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from .geometry import StrikeAdapter
from .motor import GROUPS
from .physics import step_contacts
from .provenance import ROOT, digest, manifest, write_json
from .rules import HitTracker, Rules
from .scene import SOURCE


@dataclass(frozen=True)
class StrikeConfig:
    dt_s: float = 0.001
    episode_s: float = 0.8
    phase_ends_s: tuple[float, ...] = (0.20, 0.35, 0.50, 0.8)
    target: tuple[float, float, float] = (0.38, -0.20, 1.10)
    target_jitter_m: float = 0.01
    target_radius_m: float = 0.08
    target_zone: str = "torso"
    glove_radius_m: float = 0.062
    glove_mass_kg: float = 0.283

    def __post_init__(self):
        positive = (
            self.dt_s,
            self.episode_s,
            self.target_radius_m,
            self.glove_radius_m,
            self.glove_mass_kg,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Strike dimensions/timing must be finite and positive")
        if (
            len(self.target) != 3
            or not np.isfinite(self.target).all()
            or not 0 <= self.target_jitter_m <= 0.03
        ):
            raise ValueError("Invalid fixture target or randomization")
        if (
            len(self.phase_ends_s) != 4
            or not np.isfinite(self.phase_ends_s).all()
            or any(
                a >= b
                for a, b in zip((0.0, *self.phase_ends_s[:-1]), self.phase_ends_s)
            )
        ):
            raise ValueError("Expected four strictly increasing pulse phases")
        if abs(self.phase_ends_s[-1] - self.episode_s) > 1e-9:
            raise ValueError("Last pulse phase must end at episode boundary")
        if self.target_zone not in ("head", "torso", "guard"):
            raise ValueError("Unknown target zone")
        if any(
            abs(t / self.dt_s - round(t / self.dt_s)) > 1e-8
            for t in (*self.phase_ends_s, self.episode_s)
        ):
            raise ValueError("Pulse phases must contain integer physics ticks")
        Rules(dt_s=self.dt_s)


class StrikeRig:
    """One fixed-pelvis human, added glove and fixed target; all 700 muscles retained."""

    def __init__(self, config: StrikeConfig):
        self.config = config
        spec = mujoco.MjSpec.from_file(str(SOURCE))
        for key in list(spec.keys):
            spec.delete(key)
        for joint in list(spec.joints):
            if joint.name.startswith("pelvis_"):
                spec.delete(joint)
        spec.option.gravity = [0, 0, 0]
        spec.option.timestep = config.dt_s
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        glove = spec.body("hand_r").add_body(name="strike_glove", pos=[0, -0.045, 0])
        glove.mass = config.glove_mass_kg
        glove.inertia = np.full(
            3, 0.4 * config.glove_mass_kg * config.glove_radius_m**2
        )
        glove.add_geom(
            name="strike_glove",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[config.glove_radius_m, 0, 0],
            mass=0,
            contype=8,
            conaffinity=16,
            rgba=[0.9, 0.16, 0.1, 1.0],
            solref=[0.015, 1.0],
            friction=[0.5, 0.005, 0.0001],
        )
        target = spec.worldbody.add_body(
            name="strike_target", mocap=True, pos=config.target
        )
        target.add_geom(
            name="strike_target",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[config.target_radius_m, 0, 0],
            mass=0,
            contype=16,
            conaffinity=8,
            rgba=[0.2, 0.8, 0.65, 1.0],
            solref=[0.015, 1.0],
            friction=[0.5, 0.005, 0.0001],
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.model_hash = digest(spec.to_xml())
        self.task_hash = digest(asdict(config))
        self.rules = Rules(dt_s=config.dt_s)
        self.groups = [
            np.array([self.model.actuator(name).id for name in group])
            for group in GROUPS
        ]
        self.glove = self.model.geom("strike_glove").id
        self.target = self.model.geom("strike_target").id
        self.trunk = self.model.body("sternum").id
        self.adapter = StrikeAdapter(
            self.model, self.glove, self.target, self.trunk, config.target_zone
        )
        self.elbow = self.model.joint("elbow_flexion_r").qposadr[0]
        self.signature = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_size = mujoco.mj_stateSize(self.model, self.signature)
        self.n_steps = round(config.episode_s / config.dt_s)
        self.last_valid_engine = np.zeros(self.state_size)
        self.reset(0)

    def reset(self, seed: int, params=None):
        self.params = (
            np.full((4, len(GROUPS)), -2.0)
            if params is None
            else np.asarray(params, dtype=float).copy()
        )
        if self.params.shape != (4, len(GROUPS)) or not np.isfinite(self.params).all():
            raise ValueError("Expected four finite phases of six muscle-group logits")
        self.excitations = 1 / (1 + np.exp(-np.clip(self.params, -12, 12)))
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[0] = np.asarray(self.config.target) + rng.uniform(
            -self.config.target_jitter_m, self.config.target_jitter_m, 3
        )
        self.data.qpos[self.elbow] = 0.05 + rng.uniform(-0.01, 0.01)
        self.data.qvel[:] = rng.uniform(-0.005, 0.005, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.tick = 0
        self.ended = False
        self.scenario_seed = seed
        self.tracker = HitTracker(f"strike-{seed}", "A/right", self.rules)
        evidence = self.adapter.measure(self.data, [], self.data.qvel.copy())
        self.initial_extension = evidence.extension_m
        self.stats = {
            "min_clearance_m": evidence.clearance_m,
            "final_clearance_m": evidence.clearance_m,
            "peak_extension_m": evidence.extension_m,
            "final_extension_m": evidence.extension_m,
            "peak_force_N": 0.0,
            "normal_impulse_Ns": 0.0,
            "max_penetration_m": 0.0,
            "contact_steps": 0,
        }

    def advance(self):
        if self.ended or self.tick >= self.n_steps:
            raise RuntimeError("Cannot advance a finished motor episode")
        phase = min(
            3,
            int(
                np.searchsorted(
                    self.config.phase_ends_s, self.tick * self.config.dt_s, side="right"
                )
            ),
        )
        self.data.ctrl[:] = 0.005
        for i, ids in enumerate(self.groups):
            self.data.ctrl[ids] = self.excitations[phase, i]
        pre_qvel = self.data.qvel.copy()
        mujoco.mj_getState(
            self.model, self.data, self.last_valid_engine, self.signature
        )
        contacts = step_contacts(self.model, self.data, {self.glove})
        evidence = self.adapter.measure(self.data, contacts, pre_qvel)
        self.tracker.update(
            self.tick,
            evidence.touches,
            clearance_m=evidence.clearance_m,
            extension_m=evidence.extension_m,
        )
        s = self.stats
        s["min_clearance_m"] = min(s["min_clearance_m"], evidence.clearance_m)
        s["final_clearance_m"] = evidence.clearance_m
        s["peak_extension_m"] = max(s["peak_extension_m"], evidence.extension_m)
        s["final_extension_m"] = evidence.extension_m
        s["peak_force_N"] = max(s["peak_force_N"], evidence.normal_force_N)
        s["normal_impulse_Ns"] += evidence.normal_force_N * self.config.dt_s
        s["max_penetration_m"] = max(s["max_penetration_m"], evidence.max_penetration_m)
        s["contact_steps"] += int(evidence.normal_force_N > 0.0)
        points = sum(e["points"] for e in self.tracker.ledger.values())
        row = [
            self.tick * self.config.dt_s,
            *evidence.glove_position.tolist(),
            evidence.normal_force_N,
            evidence.clearance_m,
            evidence.extension_m,
            points,
        ]
        self.tick += 1
        return row

    def finish(self):
        if self.tick != self.n_steps:
            raise ValueError("Motor episode must finish at its physical horizon")
        if not self.ended:
            # Administrative bell adds no new physical sample or impulse interval.
            self.tracker.update(
                self.tick,
                [],
                clearance_m=self.stats["final_clearance_m"],
                extension_m=self.stats["final_extension_m"],
                bell=True,
            )
            self.ended = True
        events = list(self.tracker.ledger.values())
        hits = sum(e["points"] > 0 for e in events)
        retraction = self.stats["peak_extension_m"] - self.stats["final_extension_m"]
        recovered = (
            self.tracker.current is None
            and self.stats["final_clearance_m"] >= self.rules.clearance_m
            and retraction >= self.rules.retraction_m
        )
        proximity = np.exp(-((max(0.0, self.stats["min_clearance_m"]) / 0.08) ** 2))
        withdrawal = min(1.0, max(0.0, retraction) / 0.15)
        final_extension_cost = max(
            0.0, self.stats["final_extension_m"] - self.initial_extension - 0.03
        )
        # Motor-curriculum objective; not canonical terminal match utility.
        reward = (
            4 * min(hits, 1)
            + 2 * proximity
            + withdrawal
            + 2 * int(hits == 1 and recovered)
            - 3 * final_extension_cost
            - 0.03 * float(np.mean(self.excitations**2))
            - 20 * max(0.0, self.stats["max_penetration_m"] - 0.02)
        )
        return {
            "scenario_seed": self.scenario_seed,
            "valid": True,
            "reward": float(reward),
            "legal_hits": hits,
            "points": sum(e["points"] for e in events),
            "hit_success": hits == 1,
            "strike_withdrawal_success": bool(hits == 1 and recovered),
            "recovered": bool(recovered),
            "retraction_m": float(retraction),
            "events": events,
            **self.stats,
        }

    def snapshot(self, last_valid=False):
        state = np.zeros(self.state_size)
        if last_valid:
            state[:] = self.last_valid_engine
        else:
            mujoco.mj_getState(self.model, self.data, state, self.signature)
        return {
            "schema": "joxing-strike-snapshot-v0.0.2",
            "model_hash": self.model_hash,
            "task_hash": self.task_hash,
            "rules_hash": self.rules.hash,
            "state_signature": int(self.signature),
            "state_size": self.state_size,
            "state": state.tolist(),
            "tick": self.tick,
            "ended": self.ended,
            "scenario_seed": self.scenario_seed,
            "params": self.params.tolist(),
            "initial_extension": self.initial_extension,
            "stats": copy.deepcopy(self.stats),
            "tracker": copy.deepcopy(self.tracker.snapshot()),
        }

    def restore(self, snapshot):
        if (
            snapshot["model_hash"] != self.model_hash
            or snapshot["task_hash"] != self.task_hash
            or snapshot["rules_hash"] != self.rules.hash
            or snapshot["state_signature"] != int(self.signature)
            or snapshot["state_size"] != self.state_size
        ):
            raise ValueError("Incompatible motor snapshot")
        state = np.asarray(snapshot["state"], dtype=float)
        if state.shape != (self.state_size,) or not np.isfinite(state).all():
            raise ValueError("Invalid engine snapshot")
        mujoco.mj_setState(self.model, self.data, state, self.signature)
        self.params = np.asarray(snapshot["params"], dtype=float)
        self.excitations = 1 / (1 + np.exp(-np.clip(self.params, -12, 12)))
        for key in ("tick", "ended", "scenario_seed", "initial_extension", "stats"):
            setattr(self, key, copy.deepcopy(snapshot[key]))
        self.tracker = HitTracker.restore(copy.deepcopy(snapshot["tracker"]))

    def rollout(self, params, seed: int, record=False):
        self.reset(seed, params)
        trace, states, snapshot = [], [], None
        while self.tick < self.n_steps:
            row = self.advance()
            if record:
                trace.append(row)
                states.append(self.data.qpos.copy())
                if snapshot is None and row[4] > 0:
                    snapshot = self.snapshot()
        result = self.finish()
        if record:
            result |= {
                "trace": trace,
                "qpos": np.asarray(states),
                "contact_snapshot": snapshot,
                "target": self.data.mocap_pos[0].tolist(),
            }
        return result


_rig = None


def _init_worker(config):
    global _rig
    _rig = StrikeRig(config)


def _episode(job):
    params, seed = job
    try:
        return _rig.rollout(params, seed)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {
            "valid": False,
            "scenario_seed": seed,
            "failure": str(exc),
            "params": np.asarray(params).tolist(),
            "snapshot": _rig.snapshot(last_valid=True),
        }


def _checked(results, out: Path):
    failed = [r for r in results if not r["valid"]]
    if failed:
        write_json(
            out / "fault.json",
            manifest()
            | {
                "attempts_in_batch": len(results),
                "invalid_count": len(failed),
                "failures": failed,
            },
        )
        raise RuntimeError(
            "Invalid strike rollout; fault receipt saved; training halted"
        )
    return results


def initial_strike_policy():
    """Explicitly convert the preceding reach experiment into four pulse phases."""
    path = ROOT / "artifacts/receipts/reach-checkpoint.npz"
    with np.load(path, allow_pickle=False) as checkpoint:
        reach = checkpoint["mean"].copy()
    if reach.shape != (2, len(GROUPS)) or not np.isfinite(reach).all():
        raise ValueError("Invalid prior reach warm-start artifact")
    return np.array(
        [reach[0], reach[1], np.full(len(GROUPS), -2.0), np.full(len(GROUPS), -2.0)]
    )


def save_checkpoint(path: Path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def _replay_contact(config, params, seed, out):
    rig = StrikeRig(config)
    uninterrupted = rig.rollout(params, seed, record=True)
    snapshot = uninterrupted["contact_snapshot"]
    if snapshot is None:
        return {
            "passed": False,
            "reason": "No contact in preregistered replay scenario",
        }
    final_state = np.zeros(rig.state_size)
    mujoco.mj_getState(rig.model, rig.data, final_state, rig.signature)
    restored = StrikeRig(config)
    restored.restore(json.loads(json.dumps(snapshot)))
    while restored.tick < restored.n_steps:
        restored.advance()
    replay = restored.finish()
    after = np.zeros(restored.state_size)
    mujoco.mj_getState(restored.model, restored.data, after, restored.signature)
    error = float(np.max(np.abs(final_state - after)))
    ledger_equal = uninterrupted["events"] == replay["events"]
    write_json(out / "contact-snapshot.json", snapshot)
    return {
        "passed": bool(error <= 1e-9 and ledger_equal),
        "snapshot_tick": snapshot["tick"],
        "final_state_max_error": error,
        "ledger_equal": ledger_equal,
    }


def train_strike_suite(
    out: Path,
    *,
    workers=4,
    generations=12,
    population=24,
    seeds=(7, 19, 41),
    eval_count=64,
    config=None,
    pilot=False,
):
    config = config or StrikeConfig()
    if (
        workers < 1
        or workers > 4
        or generations < 1
        or population < 8
        or eval_count < 1
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Invalid bounded strike training protocol")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise ValueError(
            "Run directory already contains a protocol; choose a new immutable run directory"
        )
    protocol = manifest() | {
        "version": "0.0.2",
        "experiment": "assisted-muscle-strike-withdrawal",
        "config": asdict(config),
        "workers": workers,
        "generations": generations,
        "population": population,
        "training_seeds": seeds,
        "eval_seeds": list(
            range(10000 if pilot else 20000, (10000 if pilot else 20000) + eval_count)
        ),
        "pilot": pilot,
        "acceptance": {
            "hit_success_per_seed": 0.8,
            "strike_withdrawal_per_seed": 0.75,
            "invalid_episodes": 0,
            "replay_state_error": 1e-9,
        },
        "assistance": ["fixed pelvis", "zero gravity"],
        "policy": "four-phase open-loop muscle-group excitation pulses",
        "competitive_result": False,
        "selection": "final fitted mean; no held-out policy selection",
        "diagnostic_scenario_seed": 10000 if pilot else 20000,
        "initial_policy": {
            "source": "artifacts/receipts/reach-checkpoint.npz",
            "conversion": "reach phases 1/2 followed by two low-excitation withdrawal phases",
            "params": initial_strike_policy().tolist(),
            "policy_hash": digest(initial_strike_policy().tolist()),
        },
    }
    protocol["protocol_hash"] = digest(protocol)
    write_json(out / "protocol.json", protocol)
    started = time.perf_counter()
    summaries = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(config,)
    ) as pool:
        for seed in seeds:
            seed_out = out / f"seed-{seed}"
            seed_out.mkdir()
            rng = np.random.default_rng(seed)
            mean, std = initial_strike_policy(), np.full((4, len(GROUPS)), 1.0)
            initial = mean.copy()
            history = []
            for generation in range(generations):
                candidates = rng.normal(mean, std, (population, *mean.shape))
                candidates[0] = mean
                results = _checked(
                    list(
                        pool.map(_episode, [(p, 100 + generation) for p in candidates])
                    ),
                    seed_out,
                )
                rewards = np.array([r["reward"] for r in results])
                elite = candidates[np.argsort(rewards)[-max(2, population // 4) :]]
                mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
                std = np.maximum(0.15, 0.2 * std + 0.8 * elite.std(axis=0))
                row = {
                    "generation": generation,
                    "episodes": (generation + 1) * population,
                    "mean_reward": float(rewards.mean()),
                    "mean_hit_success": float(
                        np.mean([r["hit_success"] for r in results])
                    ),
                    "mean_strike_withdrawal_success": float(
                        np.mean([r["strike_withdrawal_success"] for r in results])
                    ),
                    "best_reward": float(rewards.max()),
                }
                history.append(row)
                save_checkpoint(
                    seed_out / f"generation-{generation:03d}.npz",
                    candidates=candidates,
                    rewards=rewards,
                    scenario_seed=100 + generation,
                    mean=mean,
                    std=std,
                )
                write_json(seed_out / "progress.json", history)
                save_checkpoint(
                    seed_out / "checkpoint.npz", mean=mean, std=std, initial=initial
                )
                write_json(
                    out / "progress.json",
                    {
                        "training_seed": seed,
                        **row,
                        "wall_s": time.perf_counter() - started,
                    },
                )
                print(
                    f"seed={seed} generation={generation} hit_rate={row['mean_hit_success']:.3f} withdraw_rate={row['mean_strike_withdrawal_success']:.3f} reward={row['mean_reward']:.3f}",
                    flush=True,
                )
            evaluation = {}
            for name, params in (("initial", initial), ("trained", mean)):
                rows = _checked(
                    list(
                        pool.map(
                            _episode, [(params, s) for s in protocol["eval_seeds"]]
                        )
                    ),
                    seed_out,
                )
                write_json(seed_out / f"{name}-evaluations.json", rows)
                evaluation[name] = {
                    "attempts": len(rows),
                    "invalid": 0,
                    "hit_success_rate": float(
                        np.mean([r["hit_success"] for r in rows])
                    ),
                    "strike_withdrawal_success_rate": float(
                        np.mean([r["strike_withdrawal_success"] for r in rows])
                    ),
                    "mean_peak_force_N": float(
                        np.mean([r["peak_force_N"] for r in rows])
                    ),
                    "mean_normal_impulse_Ns": float(
                        np.mean([r["normal_impulse_Ns"] for r in rows])
                    ),
                }
            diagnostic_seed = protocol["diagnostic_scenario_seed"]
            replay = _replay_contact(config, mean, diagnostic_seed, seed_out)
            rig = StrikeRig(config)
            for name, params in (("initial", initial), ("trained", mean)):
                episode = rig.rollout(params, diagnostic_seed, record=True)
                np.savez_compressed(
                    seed_out / f"{name}-replay.npz",
                    qpos=episode.pop("qpos"),
                    trace=np.array(episode.pop("trace")),
                    params=params,
                    target=episode["target"],
                )
                episode.pop("contact_snapshot")
                write_json(seed_out / f"{name}-replay.json", episode)
            role_controls = {}
            for zone, expected in (("head", 2), ("guard", 0)):
                control = StrikeRig(replace(config, target_zone=zone)).rollout(
                    mean, diagnostic_seed
                )
                role_controls[zone] = {
                    "points": control["points"],
                    "expected_points": expected,
                    "physical_impulse_Ns": control["normal_impulse_Ns"],
                    "passed": control["points"] == expected,
                }
            passive = rig.rollout(np.full((4, len(GROUPS)), -12.0), diagnostic_seed)
            role_controls["passive"] = {
                "points": passive["points"],
                "expected_points": 0,
                "passed": passive["points"] == 0,
            }
            write_json(seed_out / "role-controls.json", role_controls)
            result = {
                "training_seed": seed,
                "policy_hash": digest(mean.tolist()),
                "model_hash": rig.model_hash,
                "task_hash": rig.task_hash,
                "rules_hash": rig.rules.hash,
                "evaluation": evaluation,
                "replay": replay,
                "role_controls": role_controls,
                "history": history,
                "paired_task_success_improvement": evaluation["trained"][
                    "strike_withdrawal_success_rate"
                ]
                - evaluation["initial"]["strike_withdrawal_success_rate"],
                "passed": bool(
                    evaluation["trained"]["hit_success_rate"] >= 0.8
                    and evaluation["trained"]["strike_withdrawal_success_rate"] >= 0.75
                    and replay["passed"]
                    and all(c["passed"] for c in role_controls.values())
                ),
            }
            write_json(seed_out / "result.json", result)
            summaries.append(result)
    result = manifest() | {
        "version": "0.0.2",
        "protocol_hash": protocol["protocol_hash"],
        "pilot": pilot,
        "passed": bool(
            all(r["passed"] for r in summaries)
            and np.mean([r["paired_task_success_improvement"] for r in summaries]) > 0
        ),
        "seeds": summaries,
        "wall_s": time.perf_counter() - started,
        "training_episodes": generations * population * len(seeds),
        "evaluation_attempts": 2 * eval_count * len(seeds),
        "diagnostic_attempts": 7 * len(seeds),
        "invalid_episodes": 0,
        "limitations": [
            "fixed pelvis and zero gravity",
            "open-loop pulses",
            "one spherical target neighborhood",
            "simulation force predictions, no empirical calibration",
            "not unassisted boxing or self-play",
        ],
    }
    write_json(out / "result.json", result)
    return result
