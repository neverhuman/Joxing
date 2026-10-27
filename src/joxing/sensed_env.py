"""Opt-in sensor-timed diagnostic world; default BoxingEnv stays unchanged."""

import copy

import mujoco
import numpy as np

from .env import INTEGRATION, PLAYERS, BoxingEnv, excitation
from .observations import SCHEMA_HASH
from .provenance import digest
from .replay import ReplaySnapshot
from .sensing import SensorPacket, SensorQueue


class SensedBoxingEnv(BoxingEnv):
    def __init__(self, *args, sensor_schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        schedule = (
            {a: [0] for a in PLAYERS} if sensor_schedule is None else sensor_schedule
        )
        if set(schedule) != set(PLAYERS) or any(
            not isinstance(v, (list, tuple))
            or not 1 <= len(v) <= 16
            or any(
                x is not None and (type(x) is not int or not 0 <= x <= 40) for x in v
            )
            for v in schedule.values()
        ):
            raise ValueError("Both bounded deterministic sensor schedules required")
        self.sensor_schedule = {a: tuple(schedule[a]) for a in PLAYERS}
        self.contract = self.contract | {
            "schema": "joxing-sensed-boxing-env-v1",
            "sensor_schedule": {a: list(s) for a, s in self.sensor_schedule.items()},
            "sensor_poll": "every completed physics tick; sample every20ticks",
            "actor_metadata": "private game identity redacted",
            "sample_proof_capacity": 24,
        }
        self.contract_hash = digest(self.contract)
        self.sensors = None

    def _validate_sensor_contract(self):
        if (
            digest(self.contract) != self.contract_hash
            or {a: list(s) for a, s in self.sensor_schedule.items()}
            != self.contract["sensor_schedule"]
        ):
            raise ValueError("Sensor contract/schedule drift")

    def _cache_observations(self):
        self._validate_sensor_contract()
        super()._cache_observations()  # Private true-state cache, never actor input.
        if self.sensors is None or self.sensors["A"].game_id != self.game_id:
            self.sensors = {a: SensorQueue(self.game_id, a) for a in PLAYERS}
            self.sample_proofs = {}
            self.last_sensor_sample = -1
            self.sensor_events = []
        if self.tick % 20 == 0 and self.tick != self.last_sensor_sample:
            self.sample_proofs[self.tick] = {
                "engine": self.decision_state.tolist(),
                "points": list(self.points_at_observation),
                "commands": {
                    a: c.tolist() for a, c in self.commands_at_observation.items()
                },
            }
            for a in PLAYERS:
                schedule = self.sensor_schedule[a]
                delay = schedule[(self.tick // 20) % len(schedule)]
                packet = SensorPacket(
                    self.game_id,
                    a,
                    SCHEMA_HASH,
                    self.tick,
                    self.tick + (delay or 0),
                    self.cached_observations[a].values,
                )
                self.sensors[a].enqueue(packet, now_tick=self.tick, drop=delay is None)
                self.sensor_events.append(
                    {
                        "player": a,
                        "event": "sampled",
                        **copy.deepcopy(self.sensors[a].records[self.tick]),
                    }
                )
            self.last_sensor_sample = self.tick
        self._deliver_sensors()

    def _deliver_sensors(self):
        for a, q in self.sensors.items():
            for r in q.pop_due(self.tick):
                self.sensor_events.append({"player": a, "event": "received", **r})
        needed = set().union(*(q.needed_ticks() for q in self.sensors.values()))
        self.sample_proofs = {
            t: p for t, p in self.sample_proofs.items() if t in needed
        }
        if len(self.sample_proofs) > 24 or len(self.sensor_events) > 24004:
            raise TimeoutError("Sensor evidence allocation bound")

    def advance_substep(self, guard=None):
        self._validate_sensor_contract()
        super().advance_substep(guard)
        try:
            self._deliver_sensors()
        except Exception as error:
            self.invalid_reason = type(error).__name__ + ": " + str(error)
            self.fault_state = self.state()
            raise

    def observations(self):
        if not self.initialized:
            raise RuntimeError("Reset before observing")
        return {a: q.actor_input(self.tick) for a, q in self.sensors.items()}

    def decision_frame(self):
        self._validate_sensor_contract()
        super().decision_frame()  # Validate decision/hold/ending clock only.
        return self.observations()

    def act(self, controllers, *, delays=None, guard=None):
        if set(controllers) != set(PLAYERS):
            raise ValueError("Both sensor controllers required")
        frame = self.decision_frame()
        # The prescribed diagnostic cold-start fallback does not read state or
        # consume policy RNG. Actor functions receive only delivered inputs.
        commands = {
            a: controllers[a](frame[a], self.rngs[a])
            if frame[a].available
            else self.commands[a].copy()
            for a in PLAYERS
        }
        return self.step(commands, delays=delays, guard=guard)

    def snapshot(self):
        self._validate_sensor_contract()
        value = {
            "schema": "joxing-sensed-replay-v1",
            "base": ReplaySnapshot.capture(self),
            "sensors": {a: q.snapshot() for a, q in self.sensors.items()},
            "sample_proofs": {
                str(t): copy.deepcopy(p) for t, p in self.sample_proofs.items()
            },
            "last_sensor_sample": self.last_sensor_sample,
        }
        return value | {"sha256": digest(value)}

    def restore(self, snapshot):
        self._validate_sensor_contract()
        s = copy.deepcopy(snapshot)
        expected = s.pop("sha256")
        if (
            digest(s) != expected
            or s["schema"] != "joxing-sensed-replay-v1"
            or set(s)
            != {"schema", "base", "sensors", "sample_proofs", "last_sensor_sample"}
        ):
            raise ValueError("Foreign/corrupt sensor application snapshot")
        candidate = copy.copy(self)
        ReplaySnapshot.restore(candidate, s["base"])
        if set(s["sensors"]) != set(PLAYERS):
            raise ValueError("Both sensor owners required")
        candidate.sensors = {
            a: SensorQueue.restore(s["sensors"][a], game_id=candidate.game_id, player=a)
            for a in PLAYERS
        }
        if any(
            q.last_polled_tick != candidate.tick for q in candidate.sensors.values()
        ):
            raise ValueError("Sensor/engine clock differs")
        last = s["last_sensor_sample"]
        if type(last) is not int or last != candidate.tick // 20 * 20:
            raise ValueError("Sensor cadence differs")
        candidate.last_sensor_sample = last
        for a, q in candidate.sensors.items():
            if (
                q.capacity != 64
                or q.max_delay_ticks != 40
                or q.history.capacity != 8
                or not q.records
                or max(q.records) != last
            ):
                raise ValueError("Sensor capacity/sample receipt differs from contract")
            if any(p.delivery_tick <= candidate.tick for p in q.pending.values()):
                raise ValueError("Already-due sensor packet remains pending")
            for t, r in q.records.items():
                schedule = self.sensor_schedule[a]
                delay = schedule[(t // 20) % len(schedule)]
                if (
                    t % 20
                    or t > last
                    or r["delivery_tick"] != t + (delay or 0)
                    or r["dropped_at_source"] != (delay is None)
                    or (
                        r["received_tick"] is not None
                        and r["received_tick"] != r["delivery_tick"]
                    )
                ):
                    raise ValueError("Sensor receipt differs from declared schedule")
        proofs = {int(t): p for t, p in s["sample_proofs"].items()}
        needed = set().union(*(q.needed_ticks() for q in candidate.sensors.values()))
        if set(proofs) != needed or len(proofs) > 24:
            raise ValueError("Missing/unbounded actual sampled-engine proof")
        for t, p in proofs.items():
            raw = np.array(p["engine"], np.float64)
            if (
                t > last
                or t % 20
                or raw.shape != self.state().shape
                or not np.isfinite(raw).all()
                or abs(raw[0] - t * 0.001) > 1e-9
            ):
                raise ValueError("Malformed sampled engine clock/state")
            scratch = mujoco.MjData(self.model)
            mujoco.mj_setState(self.model, scratch, raw, INTEGRATION)
            if np.any(scratch.qfrc_applied) or np.any(scratch.xfrc_applied):
                raise ValueError("Sample proof contains assistance")
            mujoco.mj_forward(self.model, scratch)
            if any(w.number for w in scratch.warning):
                raise ValueError("Sample proof emits engine warning")
            if set(p["commands"]) != set(PLAYERS):
                raise ValueError("Both sampled own-command states required")
            for a in PLAYERS:
                command = np.asarray(p["commands"][a], np.float32)
                if (
                    command.shape != (700,)
                    or not np.isfinite(command).all()
                    or np.any(np.abs(command) > 1)
                    or not np.array_equal(
                        scratch.ctrl[self.bodies[a].actuator_ids], excitation(command)
                    )
                ):
                    raise ValueError("Sampled controls differ from engine state")
            observations = candidate._make_observations(
                scratch, t, p["points"], p["commands"]
            )
            for a, q in candidate.sensors.items():
                packet = q.pending.get(t)
                if packet is not None and not np.array_equal(
                    packet.values, observations[a].values
                ):
                    raise ValueError(
                        "Pending sensor differs from actual sampled engine"
                    )
                sample = q.history._samples.get(t)
                if sample is not None and not np.array_equal(
                    sample.values, observations[a].values
                ):
                    raise ValueError(
                        "Delivered history differs from actual sampled engine"
                    )
        candidate.sample_proofs = proofs
        candidate.sensor_events = []  # Observational tail starts here, not game state.
        self.__dict__.update(candidate.__dict__)
