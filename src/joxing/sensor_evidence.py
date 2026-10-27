"""Bounded received-input evidence layered over the unchanged D0 physical tape.

The driver MUST call capture after every completed physics tick. This is an
evidence collector, not a controller: privileged sampling proofs never enter
diagnostic_command. No physics, transport or policy implementation is replaced.
"""

import copy

import numpy as np

from .boxing_evidence import PLAYERS, BoxingTape, diagnostic_command
from .motor_storage import save_npz, write_json


class BaseTapeView:
    """Expose the unchanged base snapshot shape only to the D0 tape writer."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def snapshot(self):
        return self.env.snapshot()["base"]


class SensorTape(BoxingTape):
    def __init__(self, env, *, maximum_ticks=1000):
        if type(maximum_ticks) is not int or not 1 <= maximum_ticks <= 1000:
            raise ValueError("Sensor evidence tick bound")
        super().__init__(BaseTapeView(env), maximum_ticks=maximum_ticks)
        self.sensor_initial = env.snapshot()
        self.samples = []
        self.frames = []
        self.after_rngs = []
        self.sample_ages, self.receipt_ages = [], []
        self.sensor_event_start = len(env.sensor_events)
        self.sensor_events = []
        self.sample_ticks = set()
        self._sample(env)

    def _sample(self, env):
        # Private cache is valid even if both source packets are dropped and
        # live replay proof pruning has already removed this sampled engine.
        if env.tick % 20 or env.tick in self.sample_ticks:
            return
        if len(self.samples) >= 51:
            raise TimeoutError("Sensor true-sample evidence cap")
        self.samples.append(
            {
                "tick": env.tick,
                "engine": env.decision_state.copy(),
                "points": list(env.points_at_observation),
                "commands": np.stack(
                    [env.commands_at_observation[a] for a in PLAYERS]
                ).astype(np.float32),
                "values": np.stack(
                    [env.cached_observations[a].values for a in PLAYERS]
                ),
            }
        )
        self.sample_ticks.add(env.tick)

    def choose(self, env):
        if len(self.decisions) >= 50:
            raise TimeoutError("Sensor decision evidence cap")
        frame = env.decision_frame()  # Both actors frozen before either call.
        before = {a: copy.deepcopy(env.rngs[a].bit_generator.state) for a in PLAYERS}
        commands = {
            a: diagnostic_command(f.latest_observation, f.history, env.rngs[a])
            if (f := frame[a]).available
            else env.commands[a].copy()
            for a in PLAYERS
        }
        self.decision_states.append(env.decision_state.copy())
        self.observations.append(
            np.stack(
                [
                    frame[a].latest_observation.values
                    if frame[a].available
                    else np.zeros(1970, np.float32)
                    for a in PLAYERS
                ]
            )
        )
        self.histories.append(np.stack([frame[a].history.values for a in PLAYERS]))
        self.history_ticks.append(
            np.stack([frame[a].history.sampled_ticks for a in PLAYERS])
        )
        self.history_received.append(
            np.stack(
                [
                    np.where(
                        frame[a].history.valid,
                        env.tick - frame[a].history.receipt_age_ticks,
                        -1,
                    )
                    for a in PLAYERS
                ]
            )
        )
        self.history_valid.append(np.stack([frame[a].history.valid for a in PLAYERS]))
        self.sample_ages.append(
            np.stack([frame[a].history.sample_age_ticks for a in PLAYERS])
        )
        self.receipt_ages.append(
            np.stack([frame[a].history.receipt_age_ticks for a in PLAYERS])
        )
        self.chosen.append(np.stack([commands[a] for a in PLAYERS]))
        self.decision_ticks.append(env.tick)
        self.rngs.append(before)
        self.after_rngs.append(
            {a: copy.deepcopy(env.rngs[a].bit_generator.state) for a in PLAYERS}
        )
        self.decisions.append(
            {
                "tick": env.tick,
                "points": list(env.referee.points),
                "game_id": env.game_id,
            }
        )
        self.frames.append(
            {
                a: {
                    "decision_tick": frame[a].decision_tick,
                    "available": frame[a].available,
                    "actor_called": frame[a].available,
                    "sample_age_ticks": frame[a].sample_age_ticks,
                    "receipt_age_ticks": frame[a].receipt_age_ticks,
                    "observation_owner": None
                    if not frame[a].available
                    else {
                        "game_id": frame[a].latest_observation.game_id,
                        "player": frame[a].latest_observation.player,
                        "sampled_tick": frame[a].latest_observation.sampled_tick,
                        "schema_hash": frame[a].latest_observation.schema_hash,
                    },
                }
                for a in PLAYERS
            }
        )
        return commands

    def capture(self, env):
        super().capture(BaseTapeView(env))
        self._sample(env)
        events = env.sensor_events[self.sensor_event_start :]
        if len(self.sensor_events) + len(events) > 24004:
            raise TimeoutError("Sensor receipt evidence cap")
        self.sensor_events.extend(copy.deepcopy(events))
        self.sensor_event_start = len(env.sensor_events)

    def save(self, env, folder, *, quarantined=False):
        from pathlib import Path

        folder = Path(folder)
        result = super().save(BaseTapeView(env), folder, quarantined=quarantined)
        write_json(folder / "sensor-initial.json", self.sensor_initial)
        if not quarantined:
            write_json(folder / "sensor-final.json", env.snapshot())
        write_json(
            folder / "sensor-trace.json",
            {
                "schema": "joxing-sensor-evidence-v1",
                "frames": self.frames,
                "rngs_after": self.after_rngs,
                "events": self.sensor_events,
                "sample_points": [s["points"] for s in self.samples],
                "initial_events_in_snapshot": True,
            },
        )
        width = len(env.state())
        save_npz(
            folder / "sensor-tape.npz",
            sample_ticks=np.asarray([s["tick"] for s in self.samples], np.int64),
            sample_engines=np.asarray(
                [s["engine"] for s in self.samples], np.float64
            ).reshape(-1, width),
            sample_commands=np.asarray(
                [s["commands"] for s in self.samples], np.float32
            ).reshape(-1, 2, 700),
            sample_values=np.asarray(
                [s["values"] for s in self.samples], np.float32
            ).reshape(-1, 2, 1970),
            history_sample_ages=np.asarray(self.sample_ages, np.int64).reshape(
                -1, 2, 8
            ),
            history_receipt_ages=np.asarray(self.receipt_ages, np.int64).reshape(
                -1, 2, 8
            ),
        )
        return result | {"sensor_samples": len(self.samples)}
