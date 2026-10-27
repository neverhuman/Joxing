"""Diagnostic simultaneous dual-body loop. Native competition is disabled."""

import copy
from dataclasses import asdict

import mujoco
import numpy as np

from .bodies import NativeBodyAdapter
from .history import TimedHistory, TimedSample
from .observations import DIMENSION, SCHEMA_HASH, PublicState, observe
from .physics import step_contacts
from .provenance import digest
from .referee import MatchReferee
from .rules import match_reward
from .scene import build_scene
from .timing import ActionQueue, CommandPacket

INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION
PLAYERS = ("A", "B")


def excitation(command):
    return 1 / (1 + np.exp(-5 * (command - 0.5)))


class BoxingEnv:
    competitive_ready = False

    def __init__(
        self,
        scene=None,
        *,
        horizon_ticks=60000,
        bundles=None,
        scenario="native-keyframe-v1",
    ):
        self.scene = build_scene() if scene is None else scene
        self.model = self.scene.model
        if (
            self.model.opt.timestep != 0.001
            or self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        ):
            raise ValueError("Diagnostic scene requires named1ms implicitfast contract")
        if type(horizon_ticks) is not int or not 1 <= horizon_ticks <= 60000:
            raise ValueError("Invalid diagnostic bell time")
        self.horizon_ticks, self.control_ticks = horizon_ticks, 20
        if scenario not in {
            "native-keyframe-v1",
            "close-overlap-collision-probe-v1",
            "close-overlap-collision-probe-v2",
        }:
            raise ValueError("Unregistered diagnostic reset scenario")
        self.scenario = scenario
        self.bundles = dict(
            bundles or {a: "diagnostic-stateless-" + a for a in PLAYERS}
        )
        if set(self.bundles) != set(PLAYERS) or not all(self.bundles.values()):
            raise ValueError("Both frozen controller identities required")
        self.bodies = {a: NativeBodyAdapter(self.scene, a) for a in PLAYERS}
        ids = np.concatenate([b.actuator_ids for b in self.bodies.values()])
        if len(set(ids)) != 1400 or self.model.nu != 1400 or self.model.na != 1400:
            raise ValueError("All1400native muscles required")
        self.contract = {
            "schema": "joxing-boxing-env-diagnostic-v1",
            "scene_hash": self.scene.report["scene_hash"],
            "observation_hash": SCHEMA_HASH,
            "rules_hash": MatchReferee(self.scene, "contract").rules.hash,
            "horizon_ticks": horizon_ticks,
            "reset_scenario": scenario,
            "reset_local_root_shift_m": {
                "native-keyframe-v1": 0.0,
                "close-overlap-collision-probe-v1": 0.7,
                "close-overlap-collision-probe-v2": 0.727,
            }[scenario],
            "control_ticks": 20,
            "bundles": self.bundles,
            "transform": "sigmoid(5*(normalized700-.5)) once in float32",
            "queue_capacity": 64,
            "max_delay_ticks": 40,
            "history_capacity": 8,
            "competition": False,
            "profile": self.scene.report["profile"],
        }
        self.contract_hash = digest(self.contract)
        self.data = mujoco.MjData(self.model)
        self.episode_serial = 0
        self.physics_total = 0
        self.initialized = False

    def assert_competitive(self):
        raise RuntimeError("Native contact/rules/motor/replay gates not qualified")

    def state(self, data=None):
        value = np.empty(mujoco.mj_stateSize(self.model, INTEGRATION))
        mujoco.mj_getState(
            self.model, self.data if data is None else data, value, INTEGRATION
        )
        return value

    def reset(self, *, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("Explicit nonnegative seed required")
        self.episode_serial += 1
        self.game_id = f"diagnostic-{seed}-{self.episode_serial}"
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.scene.report["initial_qpos"]
        if self.scenario != "native-keyframe-v1":
            # Reset-only collision stress input, not a boxing start or strike.
            # Geometry, masses, muscle parameters and evaluated poses stay native.
            for a in PLAYERS:
                address = self.model.joint(a + "_pelvis_tx").qposadr[0]
                self.data.qpos[address] += self.contract["reset_local_root_shift_m"]
        self.commands = {a: np.zeros(700, np.float32) for a in PLAYERS}
        for a, b in self.bodies.items():
            self.data.ctrl[b.actuator_ids] = excitation(self.commands[a])
        self.tick = 0
        self.remaining_control = 0
        self.control_start = 0
        self.done = False
        self.truncated = False
        self.invalid_reason = None
        self.end_reason = None
        self.referee = MatchReferee(self.scene, self.game_id)
        self.queues = {
            a: ActionQueue(self.game_id, a, self.bundles[a]) for a in PLAYERS
        }
        self.histories = {
            a: TimedHistory(self.game_id, a, SCHEMA_HASH, DIMENSION) for a in PLAYERS
        }
        seeds = np.random.SeedSequence(seed).spawn(2)
        self.rngs = {
            a: np.random.default_rng(s) for a, s in zip(PLAYERS, seeds, strict=True)
        }
        self.policy_states = {a: {} for a in PLAYERS}
        self.evidence = []
        self.last_events = []
        self.last_rewards = (0.0, 0.0)
        self.initialized = True
        self._cache_observations()
        return self.observations(), self.info()

    def _make_observations(self, data, tick, points, commands):
        measured = {a: b.measure(data) for a, b in self.bodies.items()}
        public = PublicState(
            self.game_id,
            tick,
            (self.horizon_ticks - tick) * 0.001,
            tuple(points),
            np.array(
                [[-2.5, -2.5, 0], [2.5, -2.5, 0], [2.5, 2.5, 0], [-2.5, 2.5, 0]], float
            ),
        )
        return {
            a: observe(
                measured[a], measured["B" if a == "A" else "A"], public, commands[a]
            )
            for a in PLAYERS
        }

    def _cache_observations(self):
        warnings = np.array([w.number for w in self.data.warning])
        mujoco.mj_forward(self.model, self.data)
        if np.any(np.array([w.number for w in self.data.warning]) > warnings):
            raise RuntimeError("Diagnostic observation forward emitted engine warning")
        self.observation_tick = self.tick
        self.commands_at_observation = {a: c.copy() for a, c in self.commands.items()}
        self.points_at_observation = self.referee.points
        self.cached_observations = self._make_observations(
            self.data, self.tick, self.referee.points, self.commands
        )
        self.decision_state = self.state()
        for a, o in self.cached_observations.items():
            self.histories[a].receive(
                TimedSample(
                    self.game_id, a, SCHEMA_HASH, self.tick, self.tick, o.values
                ),
                now_tick=self.tick,
            )

    def observations(self):
        if not self.initialized:
            raise RuntimeError("Reset before observing")
        return dict(self.cached_observations)

    def decision_frame(self):
        if not self.initialized:
            raise RuntimeError("Reset before requesting a decision")
        if self.remaining_control or self.done or self.invalid_reason:
            raise RuntimeError("No new decision inside a held command or after ending")
        return {
            a: (self.cached_observations[a], self.histories[a].at(self.tick))
            for a in PLAYERS
        }

    def act(self, controllers):
        if set(controllers) != set(PLAYERS):
            raise ValueError("Both controllers required")
        frame = self.decision_frame()
        actions = {a: controllers[a](*frame[a], self.rngs[a]) for a in PLAYERS}
        return self.step(actions)

    def begin_control(self, commands, *, delays=None):
        if digest(self.contract) != self.contract_hash:
            raise ValueError("Diagnostic contract drift")
        self.decision_frame()
        if set(commands) != set(PLAYERS):
            raise ValueError("Both simultaneous commands required")
        delays = {a: 0 for a in PLAYERS} if delays is None else delays
        if set(delays) != set(PLAYERS):
            raise ValueError("Both explicit corner delays required")
        staged = {}
        for a in PLAYERS:
            if type(delays[a]) is not int or not 0 <= delays[a] <= 40:
                raise ValueError("Invalid diagnostic command delay")
            # Validate both complete queues before changing any live actor state.
            q = ActionQueue.restore(
                self.queues[a].snapshot(),
                game_id=self.game_id,
                player=a,
                bundle_hash=self.bundles[a],
            )
            q.enqueue(
                CommandPacket(
                    self.game_id,
                    a,
                    self.tick,
                    self.tick + delays[a],
                    commands[a],
                    self.bundles[a],
                ),
                now_tick=self.tick,
            )
            staged[a] = q
        self.queues = staged
        self.remaining_control = self.control_ticks
        self.control_start = self.tick
        self.last_events = []
        self.last_rewards = (0.0, 0.0)

    def advance_substep(self, guard=None):
        if not self.remaining_control or self.done or self.invalid_reason:
            raise RuntimeError("No owned pending substep")
        if guard is not None:
            guard(1)
        if len(self.evidence) >= 60000:
            raise TimeoutError("Retained diagnostic interval limit")
        try:
            if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
                raise ValueError("External assistance forbidden")
            for a, b in self.bodies.items():
                if self.queues[
                    a
                ].last_polled_tick != self.tick - 1 or not np.array_equal(
                    self.data.ctrl[b.actuator_ids], excitation(self.commands[a])
                ):
                    raise ValueError("Held control/queue ownership drift")
            due = {a: q.pop_due(self.tick) for a, q in self.queues.items()}
            for a, p in due.items():
                if p is not None:
                    self.commands[a] = p.command.copy()
                    self.data.ctrl[self.bodies[a].actuator_ids] = excitation(p.command)
            pre_qvel = self.data.qvel.copy()
            before = self.referee.points
            self.physics_total += 1
            contacts = step_contacts(self.model, self.data)
            if not np.isfinite(self.data.act).all():
                raise FloatingPointError("Nonfinite activation")
            intervals = self.referee.stage(self.data, contacts, pre_qvel)
            bell = self.tick + 1 == self.horizon_ticks
            events = self.referee.commit(intervals, self.tick, bell=bell)
            self.evidence.append(
                {
                    "tick": self.tick,
                    "contacts": [asdict(c) for c in contacts],
                    "gloves": {k: asdict(v) for k, v in intervals.items()},
                    "events": events,
                }
            )
            self.tick += 1
            self.remaining_control -= 1
            self.last_events.extend(events)
            after = self.referee.points
            outcome = int(np.sign(after[0] - after[1])) if bell else 0
            rewards = match_reward(
                before[0] - before[1],
                after[0] - after[1],
                terminal=bell,
                outcome=outcome,
            )
            self.last_rewards = tuple(
                a + b for a, b in zip(self.last_rewards, rewards, strict=True)
            )
            if bell:
                self.done = True
                self.end_reason = "diagnostic_bell"
                self.remaining_control = 0
            if not self.remaining_control:
                self._cache_observations()
        except Exception as error:
            self.invalid_reason = type(error).__name__ + ": " + str(error)
            self.fault_state = self.state()
            raise

    def step(self, commands, *, delays=None, guard=None):
        self.begin_control(commands, delays=delays)
        return self.resume_control(guard=guard)

    def resume_control(self, *, guard=None):
        while self.remaining_control:
            self.advance_substep(guard)
        return self.transition()

    def transition(self):
        if self.invalid_reason:
            raise RuntimeError("Invalid world cannot produce a match result")
        return {
            "observations": self.observations(),
            "rewards": dict(zip(PLAYERS, self.last_rewards, strict=True)),
            "terminated": {a: self.done and not self.truncated for a in PLAYERS},
            "truncated": {a: self.truncated for a in PLAYERS},
            "bootstrap_mask": {
                a: float(not (self.done and not self.truncated)) for a in PLAYERS
            },
            "trace_mask": {a: float(not self.done) for a in PLAYERS},
            "info": self.info(),
        }

    def truncate(self, reason):
        if self.done or self.invalid_reason or not reason:
            raise ValueError("Invalid external truncation")
        partial_reward = self.last_rewards if self.remaining_control else (0.0, 0.0)
        self.done = True
        self.truncated = True
        self.end_reason = "external: " + reason
        self.remaining_control = 0
        self._cache_observations()
        self.last_rewards = partial_reward
        return self.transition()

    def info(self):
        return {
            "game_id": self.game_id,
            "tick": self.tick,
            "points": self.referee.points,
            "events": copy.deepcopy(self.last_events),
            "competitive_ready": False,
            "classification": "diagnostic candidate points, not qualified boxing",
            "gates_pending": list(self.scene.report["gates_pending"]),
            "end_reason": self.end_reason,
            "invalid_reason": self.invalid_reason,
            "contract_hash": self.contract_hash,
            "controller_bundles": dict(self.bundles),
        }

    def snapshot(self):
        from .replay import ReplaySnapshot

        return ReplaySnapshot.capture(self)

    def restore(self, snapshot):
        from .replay import ReplaySnapshot

        ReplaySnapshot.restore(self, snapshot)

    def collector_cut(self):
        return self.snapshot()  # Live state continues; never an episode truncation.
