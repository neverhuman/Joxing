"""Named arm-only learning task over an intact native paired walking world.

No reference-clock warp, external force or evaluated pose write. The new policy
owns46arm-group residuals; the frozen actor still owns all700muscle channels.
Legacy138action/reward/observation and closed proof code remain unchanged.
"""

import copy
from dataclasses import asdict, dataclass

import numpy as np

from .motor_residual import action
from .native_pair_motor_rl import PairMotorConfig, actor_goal, exact_f32, yaw_frame
from .native_reset_placement import (
    MODES,
    PlacedNativePairMotorEnv,
    ResetPlacementFactory,
)
from .provenance import ROOT, digest
from .shared_motor_inputs import FrozenPairTeacher
from .shared_motor_task import NativeReferenceDriver
from .upper_residual import HeldUpperCommand, UpperResidualSpec
from .upper_residual_task import UpperPairTeacher

SCHEMA = "joxing-native-arm-guard-v1"
REWARD = "native-arm-guard-linear-distance-v1"
PLAYERS = ("A", "B")


def episode_nonce(run_id, logical_seed, serial):
    if (
        type(run_id) is not str
        or not 1 <= len(run_id) <= 64
        or not run_id.isascii()
        or type(logical_seed) is not int
        or logical_seed < 0
        or type(serial) is not int
        or serial < 1
    ):
        raise ValueError("Explicit guard run/logical epoch identity required")
    return digest({"run_id": run_id, "logical_seed": logical_seed, "serial": serial})


class LogicalEpisodeNativeReferenceDriver(NativeReferenceDriver):
    def __init__(self, task, seed, nonce):
        if (
            type(nonce) is not str
            or len(nonce) != 64
            or any(c not in "0123456789abcdef" for c in nonce)
        ):
            raise ValueError("Guard reference nonce")
        self.nonce = nonce
        super().__init__(task, seed)

    def snapshot(self):
        return super().snapshot() | {"logical_episode_nonce": self.nonce}


def guard_actor_observations(base):
    # The published actor history ends in nine current-goal values. Keep every
    # causal body/age/valid field; replace only its old reach/return schedule.
    vectors = base.observations()
    sample_tick = base.core.tick - base.core.tick % 20
    for p in PLAYERS:
        values = base.histories[p].at(sample_tick).values[-1]
        vectors[p][-9:] = actor_goal(values, 0).astype(np.float32)
    return vectors


def guard_measure(base, initial_head_z):
    model, data = base.core.model, base.scratch
    result = {}
    for p in PLAYERS:
        pelvis = model.body(p + "_pelvis").id
        head = data.xpos[model.body(p + "_head_neck").id]
        rotation = data.xmat[pelvis].reshape(3, 3)
        frame = yaw_frame(rotation)
        targets = np.array(
            [
                head
                + 0.12 * frame[0]
                + side * 0.15 * frame[1]
                - np.array([0.0, 0.0, 0.12])
                for side in (-1, 1)
            ]
        )
        gloves = [model.body(p + "_glove_" + side).id for side in ("r", "l")]
        distance = np.linalg.norm(data.xpos[gloves] - targets, axis=1)
        up = float(rotation[2, 1])
        result[p] = {
            "distances_m": distance.tolist(),
            "face_cos": float(
                np.clip(
                    np.dot(-data.xmat[gloves[0]].reshape(3, 3)[:, 1], frame[0]), 0, 1
                )
            ),
            "up_world_z": up,
            "tilt_rad": float(np.arccos(np.clip(up, -1, 1))),
            "head_z_m": float(head[2]),
            "head_drop_m": float(initial_head_z[p] - head[2]),
        }
    if not all(
        np.isfinite(
            [*m["distances_m"], *[v for k, v in m.items() if k != "distances_m"]]
        ).all()
        for m in result.values()
    ):
        raise FloatingPointError("Nonfinite guard measurement")
    return result


def guard_good(m):
    return bool(
        max(m["distances_m"]) <= 0.15
        and m["face_cos"] >= 0.5
        and m["tilt_rad"] <= np.deg2rad(20)
        and m["head_drop_m"] <= 0.10
    )


def guard_reward(m, applied, failure_onset):
    a = action(applied, 46)
    e = float(np.mean(m["distances_m"]))
    h = float(np.clip(1 - m["head_drop_m"] / 0.10, 0, 1))
    progress = float(np.clip(1 - e / 1.5, 0, 1)) * h
    return float(
        0.001
        * (
            0.2
            + 0.5 * np.clip(m["up_world_z"], 0, 1)
            + 0.3 * progress
            + 0.1 * m["face_cos"]
            - 0.01 * np.mean(a**2)
        )
        - float(failure_onset)
    )


@dataclass
class GuardCounter:
    ticks: int = 0
    current_hold: int = 0
    maximum_hold: int = 0
    distance_sum: float = 0.0

    def observe(self, m):
        self.ticks += 1
        self.current_hold = self.current_hold + 1 if guard_good(m) else 0
        self.maximum_hold = max(self.maximum_hold, self.current_hold)
        self.distance_sum += float(np.mean(m["distances_m"]))

    def summary(self):
        return {
            "mean_target_error_m": self.distance_sum / self.ticks
            if self.ticks
            else None,
            "maximum_guard_hold_ticks": self.maximum_hold,
            "qualified_hold": self.maximum_hold >= 40,
            "measured_ticks": self.ticks,
        }

    @classmethod
    def restore(cls, value, tick):
        if set(value) != {"ticks", "current_hold", "maximum_hold", "distance_sum"}:
            raise ValueError("Guard counter fields")
        c = cls(**value)
        if (
            type(c.ticks) is not int
            or c.ticks != tick
            or type(c.current_hold) is not int
            or type(c.maximum_hold) is not int
            or not 0 <= c.current_hold <= c.maximum_hold <= tick
            or not np.isfinite(c.distance_sum)
            or c.distance_sum < 0
        ):
            raise ValueError("Guard counter clock/bounds")
        return c


class GuardMotorWorld:
    actor_dimension = 7901
    competitive_ready = False
    render_mode = None

    def __init__(
        self,
        scene,
        donor,
        frozen_teacher,
        groups,
        bundle,
        spec,
        *,
        run_id,
        horizon_ticks=3000,
        guard=None,
    ):
        episode_nonce(run_id, 0, 1)
        spec.check()
        if (
            len(spec.selected_ids),
            len(spec.protected_ids),
            len(spec.active_groups),
            len(spec.mixed_groups),
        ) != (122, 578, 46, 0):
            raise ValueError("Qualified native arm partition required")
        if scene.report["reset_placement"]["mode"] != MODES[1]:
            raise ValueError("Guard learning uses corrected separated lanes")
        (
            self.scene,
            self.donor,
            self.frozen_teacher,
            self.groups,
            self.bundle,
            self.spec,
        ) = scene, donor, frozen_teacher, groups, copy.deepcopy(bundle), spec
        self.native_seed = scene.report["reset_placement"]["seed"]
        self.config = PairMotorConfig(placement=MODES[1], horizon_ticks=horizon_ticks)
        self.run_id, self._guard_callback = run_id, guard
        self.contract = {
            "schema": SCHEMA,
            "scene_hash": scene.report["scene_hash"],
            "native_seed": self.native_seed,
            "run_id": run_id,
            "teacher_bundle": self.bundle,
            "upper_spec_sha256": spec.sha256,
            "config": asdict(self.config),
            "actor": "causal4frame7901; current phase0guard9; zero delay",
            "reward": REWARD,
            "ownership": "fixed native seed; nonce(run,logical_seed,serial)",
            "action": "raw/clipped46; upper compose .1; zero138base; excitation once",
            "competition": False,
        }
        self.contract_hash = digest(self.contract)
        self.episode_serial = self.completed_episodes = self.physics_total = 0
        self.base = self.adapter = None
        self.closed = self.invalid = False
        self.rng = np.random.default_rng(0)

    @property
    def core(self):
        return None if self.base is None else self.base.core

    @property
    def histories(self):
        return self.base.histories

    @property
    def done(self):
        return self.base.done

    @property
    def failures(self):
        return self.base.failures

    def _credit(self, n):
        if self._guard_callback is not None:
            self._guard_callback(n)
        self.physics_total += n

    def _new_base(self, nonce):
        adapter = UpperPairTeacher(self.frozen_teacher, self.spec)

        def drivers(seed):
            if type(seed) is not int or seed != self.native_seed:
                raise ValueError("Guard physical reset phase changed")
            return {
                p: LogicalEpisodeNativeReferenceDriver(self.donor, seed, nonce)
                for p in PLAYERS
            }

        base = PlacedNativePairMotorEnv(
            self.scene,
            drivers,
            adapter,
            self.groups,
            self.bundle,
            config=self.config,
            guard=self._credit,
        )
        return base, adapter

    def _check(self):
        if self.closed or self.invalid or digest(self.contract) != self.contract_hash:
            raise ValueError("Closed/invalid/drifted guard world")
        self.spec.check()
        if self.base is not None:
            self.base._check()
            if self.adapter.pending is not None or self.adapter.prepared is not None:
                raise ValueError("Uncommitted guard adapter transaction")

    def reset(self, *, seed):
        self._check()
        if type(seed) is not int or seed < 0:
            raise ValueError("Explicit logical guard seed")
        serial = self.episode_serial + 1
        nonce = episode_nonce(self.run_id, seed, serial)
        if self.base is not None:
            self.base.close()
        self.base, self.adapter = self._new_base(nonce)
        self.base.reset(seed=self.native_seed)
        self.episode_seed, self.episode_serial, self.nonce = seed, serial, nonce
        self.rng = np.random.default_rng(seed)
        self.initial_head_z = {
            p: float(
                self.base.scratch.xpos[self.scene.model.body(p + "_head_neck").id, 2]
            )
            for p in PLAYERS
        }
        self.counters = {p: GuardCounter() for p in PLAYERS}
        self.returns = {p: 0.0 for p in PLAYERS}
        self.control_rewards = dict(self.returns)
        self.held = None
        self.last_metrics = {}
        self.last_transition = None
        self.cached = guard_actor_observations(self.base)
        return self.observations(), self.info()

    def observations(self):
        self._check()
        return {p: a.copy() for p, a in self.cached.items()}

    def begin_control(self, actions):
        self._check()
        if self.done or self.core.remaining or set(actions) != set(PLAYERS):
            raise ValueError("Both46guard actions at an idle live control boundary")
        pair = {p: action(actions[p], 46) for p in PLAYERS}
        self.adapter.stage(pair, game_id=self.core.game_id, tick=self.core.tick)
        try:
            commands = self.base.begin_control(
                {p: np.zeros(138, np.float32) for p in PLAYERS}
            )
            held = self.adapter.prepared
            if held is None or set(held) != set(PLAYERS):
                raise ValueError("Missing paired upper command records")
            for p in PLAYERS:
                if held[p].command_bytes != commands[p].tobytes():
                    raise ValueError("Guard double-composition fault")
                raw = np.frombuffer(held[p].teacher_bytes, np.float32)
                protected = list(self.spec.protected_ids)
                if commands[p][protected].tobytes() != raw[protected].tobytes():
                    raise ValueError("Protected teacher channel drift")
            self.held = held
        except BaseException:
            self.invalid = True
            raise
        finally:
            self.adapter.clear()
        self.control_rewards = {p: 0.0 for p in PLAYERS}
        return {p: c.copy() for p, c in commands.items()}

    def advance_substep(self):
        self._check()
        if self.held is None or not self.core.remaining or self.done:
            raise RuntimeError("No held guard command")
        for p in PLAYERS:
            command = self.held[p].command_at(
                actor=p, game_id=self.core.game_id, engine_tick=self.core.tick
            )
            if command.tobytes() != self.core.commands[p].tobytes():
                raise ValueError("Held guard command drift")
        prior = copy.deepcopy(self.failures)
        try:
            contacts = self.base.advance_substep()
            self.last_metrics = guard_measure(self.base, self.initial_head_z)
            for p in PLAYERS:
                r = guard_reward(
                    self.last_metrics[p],
                    np.frombuffer(self.held[p].residual_bytes, np.float32),
                    prior[p] is None and self.failures[p] is not None,
                )
                self.returns[p] += r
                self.control_rewards[p] += r
                self.counters[p].observe(self.last_metrics[p])
            if not self.core.remaining:
                self.cached = guard_actor_observations(self.base)
                if self.done:
                    self.completed_episodes += 1
                self.last_transition = {
                    "terminated": any(self.failures.values()),
                    "truncated": self.core.done and not any(self.failures.values()),
                    "rewards": dict(self.control_rewards),
                    "info": self.info(),
                }
            return contacts
        except BaseException:
            self.invalid = True
            raise

    def step(self, actions):
        self.begin_control(actions)
        while self.core.remaining:
            self.advance_substep()
        t = self.last_transition
        return (
            self.observations(),
            dict(t["rewards"]),
            t["terminated"],
            t["truncated"],
            copy.deepcopy(t["info"]),
        )

    def info(self):
        return {
            "schema": SCHEMA,
            "episode_id": self.core.game_id,
            "episode_serial": self.episode_serial,
            "logical_seed": self.episode_seed,
            "native_seed": self.native_seed,
            "tick": self.core.tick,
            "physics_total": self.physics_total,
            "motor_failures": copy.deepcopy(self.failures),
            "guard_metrics": {p: c.summary() for p, c in self.counters.items()},
            "guard_returns": dict(self.returns),
            "candidate_points": self.base.referee.points,
            "inner_motor_returns_diagnostic_only": dict(self.base.returns),
            "sporting_qualified": False,
            "competition": False,
            "joint_episode_count": self.completed_episodes,
        }

    def snapshot(self):
        self._check()
        value = {
            "schema": SCHEMA,
            "contract_hash": self.contract_hash,
            "episode_seed": self.episode_seed,
            "episode_serial": self.episode_serial,
            "completed_episodes": self.completed_episodes,
            "nonce": self.nonce,
            "base": self.base.snapshot(),
            "held": None
            if self.held is None
            else {p: h.snapshot() for p, h in self.held.items()},
            "initial_head_z": dict(self.initial_head_z),
            "counters": {p: asdict(c) for p, c in self.counters.items()},
            "returns": dict(self.returns),
            "control_rewards": dict(self.control_rewards),
            "last_metrics": copy.deepcopy(self.last_metrics),
            "guard_goals": {p: a[-9:].tolist() for p, a in self.cached.items()},
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }
        return value | {"sha256": digest(value)}

    def restore(self, snapshot, *, restore_epoch=False):
        self._check()
        s = copy.deepcopy(snapshot)
        sha = s.pop("sha256")
        expected = {
            "schema",
            "contract_hash",
            "episode_seed",
            "episode_serial",
            "completed_episodes",
            "nonce",
            "base",
            "held",
            "initial_head_z",
            "counters",
            "returns",
            "control_rewards",
            "last_metrics",
            "guard_goals",
            "rng",
        }
        if (
            set(s) != expected
            or digest(s) != sha
            or s["schema"] != SCHEMA
            or s["contract_hash"] != self.contract_hash
            or s["nonce"]
            != episode_nonce(self.run_id, s["episode_seed"], s["episode_serial"])
            or type(s["completed_episodes"]) is not int
            or s["completed_episodes"] < 0
            or (
                not restore_epoch
                and (s["episode_seed"], s["episode_serial"])
                != (self.episode_seed, self.episode_serial)
            )
        ):
            raise ValueError("Foreign guard snapshot/epoch")
        tick = s["base"]["core"]["tick"]
        if type(tick) is not int or not 0 <= tick <= self.config.horizon_ticks:
            raise ValueError("Guard replay clock")
        counters = {p: GuardCounter.restore(s["counters"][p], tick) for p in PLAYERS}
        rng = np.random.default_rng()
        rng.bit_generator.state = s["rng"]
        for key in (
            "initial_head_z",
            "returns",
            "control_rewards",
            "guard_goals",
            "counters",
        ):
            if set(s[key]) != set(PLAYERS):
                raise ValueError("Guard snapshot owners")
        if not np.isfinite(
            [
                *s["initial_head_z"].values(),
                *s["returns"].values(),
                *s["control_rewards"].values(),
            ]
        ).all():
            raise ValueError("Guard snapshot finite scalars")
        candidate, adapter = self._new_base(s["nonce"])
        try:
            candidate.reset(seed=self.native_seed)
            initial = {
                p: float(
                    candidate.scratch.xpos[
                        self.scene.model.body(p + "_head_neck").id, 2
                    ]
                )
                for p in PLAYERS
            }
            if digest(initial) != digest(s["initial_head_z"]):
                raise ValueError("Guard reset head anchor")
            candidate.restore(s["base"])
            cached = guard_actor_observations(candidate)
            for p in PLAYERS:
                if (
                    cached[p][-9:].tobytes()
                    != exact_f32(s["guard_goals"][p], (9,)).tobytes()
                ):
                    raise ValueError("Guard replay cached goal")
            held = None
            if s["held"] is None:
                if tick or candidate.core.remaining:
                    raise ValueError("Missing guard hold")
            else:
                if set(s["held"]) != set(PLAYERS):
                    raise ValueError("Guard held owners")
                sample_tick = (
                    tick - tick % 20 if candidate.core.remaining else tick - 20
                )
                held = {
                    p: HeldUpperCommand.restore(
                        s["held"][p],
                        self.spec,
                        actor=p,
                        game_id=candidate.core.game_id,
                        sampled_tick=sample_tick,
                    )
                    for p in PLAYERS
                }
                for p in PLAYERS:
                    if (
                        held[p].command_bytes != candidate.core.commands[p].tobytes()
                        or candidate.decoder.record["teacher"][p]
                        != np.frombuffer(held[p].command_bytes, np.float32).tolist()
                        or np.any(candidate.decoder.record["actions"][p])
                    ):
                        raise ValueError("Guard inner-zero/upper record mismatch")
            candidate._scratch()
            if tick:
                measured = guard_measure(candidate, initial)
                if digest(measured) != digest(s["last_metrics"]):
                    raise ValueError("Guard current metric mismatch")
            elif s["last_metrics"]:
                raise ValueError("Guard initial metrics")
        except BaseException:
            candidate.close()
            raise
        old = self.base
        self.base, self.adapter, self.cached, self.held = (
            candidate,
            adapter,
            cached,
            held,
        )
        self.counters, self.rng = counters, rng
        for key in (
            "episode_seed",
            "episode_serial",
            "completed_episodes",
            "nonce",
            "initial_head_z",
            "returns",
            "control_rewards",
            "last_metrics",
        ):
            setattr(self, key, s[key])
        self.last_transition = None
        if old is not None:
            old.close()

    def close(self):
        if self.base is not None:
            self.base.close()
        self.closed = True


class GuardMotorFactory:
    def __init__(self, inputs):
        self.reference = ResetPlacementFactory(inputs)
        self.spec = UpperResidualSpec.from_native_model(
            self.reference.donor.model,
            self.reference.groups,
            ROOT / "third_party/ms-human-700",
        )

    @property
    def bundle(self):
        return self.reference.bundle

    def make(self, *, native_seed=71000, run_id, horizon=3000, guard=None):
        # Reuse the qualified scene construction without retaining its evaluated data.
        original = self.reference.make(seed=native_seed, mode=MODES[1], horizon=horizon)
        scene = original.scene
        original.close()
        return GuardMotorWorld(
            scene,
            self.reference.donor,
            FrozenPairTeacher(self.reference.model, self.reference.normalizer),
            self.reference.groups,
            self.reference.bundle,
            self.spec,
            run_id=run_id,
            horizon_ticks=horizon,
            guard=guard,
        )

    def close(self):
        self.reference.close()
