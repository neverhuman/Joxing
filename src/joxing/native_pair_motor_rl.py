"""Native paired muscle motor learning. Dense motor reward, never competition.

One intact1400-muscle world, frozen scalar walking teacher, external residuals.
Only reset writes poses. Actor readouts use scratch state; live motor caches stay
unchanged. A falling valid episode is training data; solver/parity faults are not.
"""

import copy
from dataclasses import asdict, dataclass, fields

import mujoco
import numpy as np

from .bodies.native import NativeBodyAdapter
from .env import INTEGRATION
from .history import TimedHistory, TimedSample
from .motor_residual import action, decode_command, validate_groups
from .observations import DIMENSION, FEATURE_SPEC, SCHEMA_HASH, PublicState, observe
from .provenance import digest
from .referee import MatchReferee
from .scene import build_scene
from .shared_motor import NativeMotorMap
from .shared_motor_inputs import validate_pair
from .shared_motor_task import PLAYERS, SharedMotorTask
from .support import SupportMonitor, floor_loads

SCHEMA = "joxing-native-paired-motor-rl-v1"
RING = np.array([[-2.5, -2.5, 0], [2.5, -2.5, 0], [2.5, 2.5, 0], [-2.5, 2.5, 0]], float)


def residual_command(teacher, residual, groups):
    # Arithmetic +0 must not erase a teacher's signed-zero command bits.
    return (
        np.array(teacher, copy=True)
        if not np.any(residual)
        else decode_command(teacher, residual, groups)
    )


class ResidualPairTeacher:
    def __init__(self, teacher, groups):
        self.teacher, self.groups = teacher, validate_groups(groups)
        self.pending = self.record = None
        self.decisions = 0

    def stage(self, actions, *, game_id, tick):
        if self.pending is not None or set(actions) != set(PLAYERS):
            raise ValueError("Atomic paired actions required")
        staged = {p: action(actions[p], 138) for p in PLAYERS}
        self.pending = (game_id, tick, staged)

    def infer_pair(self, inputs, *, game_id, tick):
        validate_pair(inputs, game_id, tick)
        if self.pending is None or self.pending[:2] != (game_id, tick):
            raise ValueError("Missing/stale paired residual")
        _, _, actions = self.pending
        self.pending = None
        raw = self.teacher.infer_pair(inputs, game_id=game_id, tick=tick)
        self.decisions += 2
        commands = {
            p: residual_command(raw[p], actions[p], self.groups) for p in PLAYERS
        }
        self.record = {
            "game_id": game_id,
            "sampled_tick": tick,
            "actions": {p: v.tolist() for p, v in actions.items()},
            "teacher": {p: v.tolist() for p, v in raw.items()},
            "input_sha256": {p: digest(inputs[p].snapshot()) for p in PLAYERS},
        }
        return commands


class NativePairMotorCore(SharedMotorTask):
    """Named parallel/opposed constructor; unchanged base step/hold/replay methods."""

    def __init__(
        self, scene, drivers, teacher, *, horizon_ticks, bundle_hash, placement
    ):
        expected = {
            "parallel": ("pretrained-parallel-walk-1ms-v1", {"A": 0.0, "B": 0.0}),
            "opposed": (
                "composition-smoke-not-competition",
                {"A": 0.0, "B": float(np.pi)},
            ),
        }
        if (
            placement not in expected
            or scene.report["profile"] != expected[placement][0]
            or set(drivers) != set(PLAYERS)
        ):
            raise ValueError("Explicit native pair placement required")
        if (
            (scene.model.nq, scene.model.nv, scene.model.nu, scene.model.na)
            != (170, 170, 1400, 1400)
            or scene.model.opt.timestep != 0.001
            or scene.model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            or type(horizon_ticks) is not int
            or not 20 <= horizon_ticks <= 20000
            or horizon_ticks % 20
            or not bundle_hash
        ):
            raise ValueError("Intact native pair/cadence/horizon required")
        self.scene, self.model, self.drivers, self.teacher = (
            scene,
            scene.model,
            drivers,
            teacher,
        )
        self.horizon_ticks, self.bundle_hash = horizon_ticks, bundle_hash
        self.maps = {
            p: NativeMotorMap(
                self.model,
                drivers[p].task.model,
                p,
                spawn_yaw_rad=expected[placement][1][p],
            )
            for p in PLAYERS
        }
        self.data = mujoco.MjData(self.model)
        self.tick = self.remaining = self.physics_total = 0
        self.done = self.invalid = False
        self.commands = None
        self.game_id = (
            "motor-" + digest({p: drivers[p].snapshot() for p in PLAYERS})[:32]
        )
        mujoco.mj_resetData(self.model, self.data)
        for p in PLAYERS:
            m, initial = self.maps[p], drivers[p].initial
            self.data.qpos[m.qpos_ids] = initial["qpos"]
            self.data.qvel[m.qvel_ids] = initial["qvel"]
            self.data.act[m.activation_ids] = initial["activation"]
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._check()
        self.inputs = self._capture()
        self.contract = {
            "schema": "joxing-native-pair-motor-core-v1",
            "placement": placement,
            "scene_hash": scene.report["scene_hash"],
            "horizon_ticks": horizon_ticks,
            "bundle_hash": bundle_hash,
            "initial_references": {p: drivers[p].snapshot() for p in PLAYERS},
            "yaw_rad": expected[placement][1],
            "transform": "normalized700->sigmoid once float32",
            "competition": False,
        }
        self.contract_hash = digest(self.contract)


@dataclass(frozen=True)
class PairMotorConfig:
    placement: str = "opposed"
    horizon_ticks: int = 3000
    history: int = 4
    reward_id: str = "native700-paired-balance-reach-return-v1"
    reach_m: float = 0.45
    error_scale_m: float = 0.25
    alive_weight: float = 0.2
    upright_weight: float = 0.5
    reach_weight: float = 0.3
    face_weight: float = 0.1
    effort_weight: float = 0.01
    fall_penalty: float = 1.0

    def __post_init__(self):
        if (
            self.placement not in ("parallel", "opposed")
            or type(self.horizon_ticks) is not int
            or not 20 <= self.horizon_ticks <= 20000
            or self.horizon_ticks % 20
        ):
            raise ValueError("Named placement/20tick horizon required")
        fixed = asdict(self)
        fixed.pop("placement")
        fixed.pop("horizon_ticks")
        default = {
            f.name: f.default
            for f in fields(type(self))
            if f.name not in ("placement", "horizon_ticks")
        }
        if fixed != default:
            raise ValueError("New motor objective needs registered variant")


def feature_scale():
    scale = np.ones(DIMENSION, np.float32)
    for block in FEATURE_SPEC:
        name = block["name"]
        v = (
            5.0
            if "velocity" in name
            else 2.5
            if "position" in name or "vertices" in name
            else 3.0
            if name == "round_remaining"
            else 1.0
        )
        scale[block["start"] : block["stop"]] = v
    return scale


def yaw_frame(rotation):
    forward = rotation[:, 0]
    length = np.linalg.norm(forward[:2])
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError("Undefined actor yaw")
    x, y = forward[:2] / length
    return np.array([[x, y, 0], [-y, x, 0], [0, 0, 1.0]])


def actor_goal(values, tick):
    def block(name, shape):
        row = next(b for b in FEATURE_SPEC if b["name"] == name)
        return values[row["start"] : row["stop"]].reshape(shape)

    own = block("own_body_position_ego", (11, 3))
    other = block("opponent_body_position_ego", (11, 3))
    rot = block("own_body_rotation_ego", (11, 3, 3))
    phase = 0 if tick % 1000 < 300 else 1 if tick % 1000 < 600 else 2
    targets = np.stack(
        [own[1] + np.array([0.12, side * 0.15, -0.12], np.float32) for side in (-1, 1)]
    )
    delta = other[2] - own[2]
    length = np.linalg.norm(delta)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Degenerate actor motor target")
    if phase == 1:
        targets[0] = own[2] + delta / length * min(0.45, float(length))
    centers = own[[3, 4]] + np.einsum(
        "nij,j->ni", rot[[3, 4]], np.array([0, -0.045, 0], np.float32)
    )
    return np.r_[(targets - centers).ravel(), np.eye(3, dtype=np.float32)[phase]]


def actor_vector(history, tick, scale):
    h = history.at(tick)
    if not h.valid[-1] or h.sampled_ticks[-1] != tick:
        raise ValueError("Zero-delay paired sample missing")
    return np.r_[
        np.tanh(h.values / scale).ravel(),
        h.valid.astype(np.float32),
        h.sample_age_ticks * 0.001,
        h.receipt_age_ticks * 0.001,
        actor_goal(h.values[-1], tick),
    ].astype(np.float32)


def exact_f32(value, shape):
    raw = np.asarray(value)
    a = np.asarray(value, np.float32)
    if (
        raw.dtype.kind not in "fiu"
        or a.shape != shape
        or not np.isfinite(a).all()
        or not np.array_equal(raw, a.astype(np.float64))
        or not np.array_equal(np.signbit(raw), np.signbit(a))
    ):
        raise ValueError("Malformed/rounded float32 snapshot")
    return a


class NativePairMotorEnv:
    """External paired actions; model/reference factories are reused across resets.

    No sporting reward, no native model defaults changed. Points are candidate
    telemetry only. Zero-delay, four-frame causal history is an explicit profile.
    """

    competitive_ready = False
    render_mode = None

    def __init__(
        self,
        scene,
        driver_factory,
        teacher,
        groups,
        teacher_bundle,
        *,
        config=None,
        guard=None,
    ):
        self.config = config or PairMotorConfig()
        self.scene, self.driver_factory, self.teacher = scene, driver_factory, teacher
        self.groups = validate_groups(groups)
        if not teacher_bundle or any(
            type(v) is not str or len(v) != 64 for v in teacher_bundle.values()
        ):
            raise ValueError("Frozen complete teacher bundle required")
        self.bundle = copy.deepcopy(teacher_bundle)
        self.decoder = ResidualPairTeacher(teacher, self.groups)
        self.contract = {
            "schema": SCHEMA,
            "scene_hash": scene.report["scene_hash"],
            "teacher_bundle": self.bundle,
            "groups": self.groups,
            "config": asdict(self.config),
            "actor_features": SCHEMA_HASH,
            "normalization": "fixed tanh scales plus valid/sample/receipt age and current goal",
            "timing": "zero-delay;both before either action;20held1ms;four samples",
            "competition": False,
        }
        self.contract_hash = digest(self.contract)
        self.scale = feature_scale()
        self.actor_dimension = 4 * DIMENSION + 12 + 9
        self._guard_callback = guard
        self.physics_total = self.episode_serial = self.completed_episodes = 0
        self.rng = np.random.default_rng(0)
        self.core = None
        self.closed = False
        self.bodies = {p: NativeBodyAdapter(scene, p) for p in PLAYERS}
        self.scratch = mujoco.MjData(scene.model)
        self.roles = {}
        self.weights = {}
        for p in PLAYERS:
            self.roles[p] = {
                i: (
                    "right"
                    if scene.model.body(scene.model.geom_bodyid[i]).name
                    == p + "_calcn_r"
                    or scene.model.body(scene.model.geom_bodyid[i]).name
                    == p + "_toes_r"
                    else "left"
                    if scene.model.body(scene.model.geom_bodyid[i]).name
                    == p + "_calcn_l"
                    or scene.model.body(scene.model.geom_bodyid[i]).name
                    == p + "_toes_l"
                    else "nonfoot"
                )
                for i in range(scene.model.ngeom)
                if scene.model.geom(i).name.startswith(p + "_")
            }
            self.weights[p] = 9.81 * sum(
                scene.model.body_mass[i]
                for i in range(scene.model.nbody)
                if scene.model.body(i).name.startswith(p + "_")
            )
        self.floor = scene.model.geom("floor").id

    @classmethod
    def from_inputs(cls, inputs, *, config=None, guard=None):
        import json

        from .motor_residual import reference_hash
        from .native_transfer import load_controller
        from .provenance import ROOT
        from .shared_motor_inputs import FrozenPairTeacher
        from .shared_motor_task import NativeReferenceDriver
        from .upstream_motor import NativeTask
        from .wave import canonical_actor_hash

        config = config or PairMotorConfig()
        model, normalizer = load_controller(inputs)
        model.actor.eval()
        model.actor.requires_grad_(False)
        donor = NativeTask(
            json.loads((ROOT / "experiments/v0.0.6/protocol.json").read_text())[
                "environment"
            ]
        )

        def drivers(seed):
            return {p: NativeReferenceDriver(donor, seed) for p in PLAYERS}

        scene = build_scene(
            spawn_variant="parallel-walk-v1" if config.placement == "parallel" else None
        )
        bundle = {
            "actor": canonical_actor_hash(model.actor),
            "normalizer": normalizer.fingerprint,
            "reference": reference_hash(donor.env.trajectory),
        }
        env = cls(
            scene,
            drivers,
            FrozenPairTeacher(model, normalizer),
            model.actor.dynsyn_layer.muscle_groups,
            bundle,
            config=config,
            guard=guard,
        )
        env.donor = donor
        return env

    def _check(self):
        if self.closed or digest(self.contract) != self.contract_hash:
            raise ValueError("Closed/drifted native pair environment")
        if self.core is not None:
            self.core._check()

    def _credit(self, n):
        if self._guard_callback is not None:
            self._guard_callback(n)
        self.physics_total += n

    def _scratch(self, *, full=False):
        mujoco.mj_setState(
            self.core.model, self.scratch, self.core.state(), INTEGRATION
        )
        if full:
            mujoco.mj_forward(self.core.model, self.scratch)
        else:
            mujoco.mj_kinematics(self.core.model, self.scratch)

    def goals(self):
        phase = (
            0
            if self.core.tick % 1000 < 300
            else 1
            if self.core.tick % 1000 < 600
            else 2
        )
        result = {}
        for p in PLAYERS:
            other = "B" if p == "A" else "A"
            m = self.core.model
            head = self.scratch.xpos[m.body(p + "_head_neck").id]
            chest = self.scratch.xpos[m.body(p + "_thoracic12").id]
            target = self.scratch.xpos[m.body(other + "_thoracic12").id]
            rotation = self.scratch.xmat[m.body(p + "_pelvis").id].reshape(3, 3)
            frame = yaw_frame(rotation)
            guard = np.array(
                [
                    head
                    + 0.12 * frame[0]
                    - 0.12 * np.array([0.0, 0, 1.0])
                    + side * 0.15 * frame[1]
                    for side in (-1, 1)
                ]
            )
            delta = target - chest
            length = np.linalg.norm(delta)
            if length < 1e-9:
                raise ValueError("Degenerate motor target direction")
            reach = chest + delta / length * min(self.config.reach_m, length)
            goal = guard.copy()
            if phase == 1:
                goal[0] = reach
            result[p] = (goal, phase, frame, delta / length)
        return result

    def _capture(self):
        self._scratch(full=True)
        measurements = {p: b.measure(self.scratch) for p, b in self.bodies.items()}
        public = PublicState(
            self.core.game_id,
            self.core.tick,
            (self.core.horizon_ticks - self.core.tick) * 0.001,
            self.referee.points,
            RING,
        )
        commands = self.core.commands or {p: np.zeros(700, np.float32) for p in PLAYERS}
        self.current_goals = self.goals()
        vectors = {}
        for p in PLAYERS:
            o = observe(
                measurements[p],
                measurements["B" if p == "A" else "A"],
                public,
                commands[p],
            )
            self.histories[p].receive(
                TimedSample(
                    o.game_id, p, SCHEMA_HASH, o.sampled_tick, o.sampled_tick, o.values
                ),
                now_tick=o.sampled_tick,
            )
            vectors[p] = actor_vector(self.histories[p], self.core.tick, self.scale)
            if (
                vectors[p].shape != (self.actor_dimension,)
                or not np.isfinite(vectors[p]).all()
            ):
                raise ValueError("Malformed causal paired actor observation")
        self.cached = {
            p: np.frombuffer(v.tobytes(), np.float32) for p, v in vectors.items()
        }
        return self.observations()

    def reset(self, *, seed):
        self._check()
        if type(seed) is not int or seed < 0:
            raise ValueError("Explicit nonnegative paired seed")
        self.rng = np.random.default_rng(seed)
        self.episode_seed = seed
        self.episode_serial += 1
        self.decoder.pending = self.decoder.record = None
        self.core = NativePairMotorCore(
            self.scene,
            self.driver_factory(seed),
            self.decoder,
            horizon_ticks=self.config.horizon_ticks,
            bundle_hash=self.contract_hash,
            placement=self.config.placement,
        )
        self.referee = MatchReferee(self.scene, self.core.game_id)
        self.histories = {
            p: TimedHistory(self.core.game_id, p, SCHEMA_HASH, DIMENSION, 4)
            for p in PLAYERS
        }
        self.monitors = {p: SupportMonitor() for p in PLAYERS}
        self.failures = {p: None for p in PLAYERS}
        self._scratch()
        self.initial_height = {
            p: float(self.scratch.xpos[self.core.model.body(p + "_pelvis").id, 2])
            for p in PLAYERS
        }
        self.done = False
        self.returns = {p: 0.0 for p in PLAYERS}
        self.control_rewards = {p: 0.0 for p in PLAYERS}
        self.last_metrics = {}
        self.last_transition = None
        return self._capture(), self.info()

    def observations(self):
        return {p: v.copy() for p, v in self.cached.items()}

    def begin_control(self, actions):
        self._check()
        if self.done or self.core.remaining:
            raise RuntimeError("Paired control inside hold or ending")
        self.decoder.stage(actions, game_id=self.core.game_id, tick=self.core.tick)
        try:
            commands = self.core.begin_control()
        except Exception:
            self.core.invalid = True
            raise
        self.control_rewards = {p: 0.0 for p in PLAYERS}
        self.control_start_tick = self.core.tick
        return commands

    def advance_substep(self):
        self._check()
        if self.done or not self.core.remaining:
            raise RuntimeError("No owned paired motor hold")
        pre = self.core.data.qvel.copy()
        tick = self.core.tick
        contacts = self.core.advance_substep(self._credit)
        intervals = self.referee.stage(self.core.data, contacts, pre)
        self.referee.commit(intervals, tick, bell=self.core.done)
        self._scratch()
        goals = self.goals()
        m = self.core.model
        for p in PLAYERS:
            pelvis = m.body(p + "_pelvis").id
            up = float(self.scratch.xmat[pelvis].reshape(3, 3)[2, 1])
            tilt = float(np.arccos(np.clip(up, -1, 1)))
            drop = self.initial_height[p] - float(self.scratch.xpos[pelvis, 2])
            owned_contacts = [
                c
                for c in contacts
                if (c.geom1 == self.floor and c.geom2 in self.roles[p])
                or (c.geom2 == self.floor and c.geom1 in self.roles[p])
            ]
            loads = floor_loads(owned_contacts, self.floor, self.roles[p])
            reason = self.monitors[p].update(loads, self.weights[p], tilt, drop, 0.001)
            onset = reason is not None and self.failures[p] is None
            if onset:
                self.failures[p] = {"reason": reason, "tick": self.core.tick}
            hand_ids = [m.body(p + "_glove_" + side).id for side in ("r", "l")]
            distance = np.linalg.norm(self.scratch.xpos[hand_ids] - goals[p][0], axis=1)
            reach = float(
                np.mean(np.exp(-((distance / self.config.error_scale_m) ** 2)))
            )
            face = float(
                np.clip(
                    np.dot(
                        -self.scratch.xmat[hand_ids[0]].reshape(3, 3)[:, 1], goals[p][3]
                    ),
                    0,
                    1,
                )
            )
            effort = float(
                np.mean(np.asarray(self.decoder.record["actions"][p], np.float32) ** 2)
            )
            reward = (
                0.001
                * (
                    self.config.alive_weight
                    + self.config.upright_weight * np.clip(up, 0, 1)
                    + self.config.reach_weight * reach
                    + self.config.face_weight * face
                    - self.config.effort_weight * effort
                )
                - self.config.fall_penalty * onset
            )
            self.control_rewards[p] += float(reward)
            self.returns[p] += float(reward)
            self.last_metrics[p] = {
                "tilt_rad": tilt,
                "drop_m": drop,
                "up_world_z": up,
                "floor_loads_N": loads,
                "hand_target_distance_m": distance.tolist(),
                "reach": reach,
                "face_cos_positive": face,
                "residual_effort": effort,
                "phase": goals[p][1],
            }
        if not self.core.remaining:
            self.done = self.core.done or any(self.failures.values())
            if self.done:
                self.completed_episodes += 1
            obs = self._capture()
            terminated = any(self.failures.values())
            truncated = self.core.done and not terminated
            self.last_transition = {
                "sample_tick": self.control_start_tick,
                "end_tick": self.core.tick,
                "observations": obs,
                "rewards": dict(self.control_rewards),
                "terminated": terminated,
                "truncated": truncated,
                "info": self.info(),
            }
        return contacts

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
            "competition": False,
            "placement": self.config.placement,
            "episode_id": self.core.game_id,
            "episode_serial": self.episode_serial,
            "tick": self.core.tick,
            "physics_total": self.physics_total,
            "motor_failures": copy.deepcopy(self.failures),
            "motor_metrics": copy.deepcopy(self.last_metrics),
            "motor_returns": dict(self.returns),
            "candidate_points": self.referee.points,
            "sporting_qualified": False,
            "joint_episode_count": self.completed_episodes,
            "teacher_decisions_per_control": 2,
        }

    def snapshot(self):
        self._check()
        if self.decoder.pending is not None:
            raise ValueError("No snapshot during uncommitted paired inference")
        value = {
            "schema": SCHEMA,
            "contract_hash": self.contract_hash,
            "core": self.core.snapshot(),
            "decoder_record": copy.deepcopy(self.decoder.record),
            "history": {p: h.snapshot() for p, h in self.histories.items()},
            "cached": {p: v.tolist() for p, v in self.cached.items()},
            "referee": self.referee.snapshot(),
            "monitors": {p: m.snapshot() for p, m in self.monitors.items()},
            "failures": copy.deepcopy(self.failures),
            "initial_height": dict(self.initial_height),
            "done": self.done,
            "returns": dict(self.returns),
            "control_rewards": dict(self.control_rewards),
            "last_metrics": copy.deepcopy(self.last_metrics),
            "control_start_tick": getattr(self, "control_start_tick", 0),
            "episode_seed": self.episode_seed,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }
        return value | {"sha256": digest(value)}

    def restore(self, snapshot):
        s = copy.deepcopy(snapshot)
        sha = s.pop("sha256")
        if (
            s["schema"] != SCHEMA
            or s["contract_hash"] != self.contract_hash
            or digest(s) != sha
            or self.decoder.pending is not None
            or s["episode_seed"] != self.episode_seed
        ):
            raise ValueError("Foreign paired motor snapshot")
        histories = {
            p: TimedHistory.restore(
                s["history"][p],
                game_id=self.core.game_id,
                player=p,
                schema_hash=SCHEMA_HASH,
            )
            for p in PLAYERS
        }
        cached = {
            p: exact_f32(s["cached"][p], (self.actor_dimension,)) for p in PLAYERS
        }
        if any(
            v.shape != (self.actor_dimension,) or not np.isfinite(v).all()
            for v in cached.values()
        ):
            raise ValueError("Malformed saved actor history")
        sample_tick = s["core"]["tick"] - s["core"]["tick"] % 20
        for p in PLAYERS:
            if (
                max(histories[p]._samples) != sample_tick
                or cached[p].tobytes()
                != actor_vector(histories[p], sample_tick, self.scale).tobytes()
            ):
                raise ValueError("Saved actor/history/goal mismatch")
        record = s["decoder_record"]
        commands = s["core"]["commands"]
        if record is None:
            if s["core"]["tick"] or s["core"]["remaining"] or commands is not None:
                raise ValueError("Missing held residual record")
        else:
            tick = s["core"]["tick"]
            remaining = s["core"]["remaining"]
            expected_tick = tick - tick % 20 if remaining else tick - 20
            if (
                record["game_id"] != self.core.game_id
                or record["sampled_tick"] != expected_tick
            ):
                raise ValueError("Held residual clock/owner")
            for p in PLAYERS:
                a = action(record["actions"][p], 138)
                raw = action(record["teacher"][p], 700)
                expected = residual_command(raw, a, self.groups)
                if expected.tobytes() != np.asarray(commands[p], np.float32).tobytes():
                    raise ValueError("Held residual reconstruction")
        monitors = {p: SupportMonitor(**s["monitors"][p]) for p in PLAYERS}
        if any(
            not np.isfinite(list(asdict(v).values())).all()
            or min(asdict(v).values()) < 0
            for v in monitors.values()
        ):
            raise ValueError("Malformed support durations")
        if (
            set(s["failures"]) != set(PLAYERS)
            or set(s["returns"]) != set(PLAYERS)
            or set(s["control_rewards"]) != set(PLAYERS)
        ):
            raise ValueError("Saved motor owners")
        for fail in s["failures"].values():
            if fail is not None and (
                set(fail) != {"reason", "tick"}
                or fail["reason"]
                not in ("nonfoot_support", "excess_tilt", "pelvis_drop")
                or type(fail["tick"]) is not int
                or not 0 < fail["tick"] <= s["core"]["tick"]
            ):
                raise ValueError("Saved physical failure")
        if not np.isfinite(
            [
                *s["returns"].values(),
                *s["control_rewards"].values(),
                *s["initial_height"].values(),
            ]
        ).all() or s["done"] != bool(
            not s["core"]["remaining"]
            and (s["core"]["done"] or any(s["failures"].values()))
        ):
            raise ValueError("Saved return/end state")
        referee = MatchReferee.restore(self.scene, s["referee"])
        rng = np.random.default_rng()
        rng.bit_generator.state = s["rng"]
        if referee.next_tick != s["core"]["tick"] or type(s["done"]) is not bool:
            raise ValueError("Referee/end clock")
        # Base validates/commits engine/reference/700commands atomically; no inference.
        self.core.restore(s["core"])
        self.histories, self.cached, self.monitors, self.referee, self.rng = (
            histories,
            cached,
            monitors,
            referee,
            rng,
        )
        self.decoder.record = record
        for name in (
            "failures",
            "initial_height",
            "done",
            "returns",
            "control_rewards",
            "last_metrics",
            "control_start_tick",
        ):
            setattr(self, name, s[name])
        self.last_transition = None

    def close(self):
        if hasattr(self, "donor"):
            self.donor.close()
        self.core = None
        self.scratch = None
        self.closed = True
