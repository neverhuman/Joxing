"""Typed bounded diagnostic match tapes. No optimization or competition."""

import copy
import hashlib
from pathlib import Path

import numpy as np

from .motor_storage import save_npz, write_json
from .observations import FEATURE_SPEC
from .provenance import digest

PLAYERS = ("A", "B")
CONTACT_COLUMNS = {
    "point": (0, 3),
    "normal": (3, 6),
    "force_on_geom2": (6, 9),
    "velocity1_at_contact": (9, 12),
    "velocity2_at_contact": (12, 15),
    "normal_force_N": (15, 16),
    "approach_mps": (16, 17),
    "penetration_m": (17, 18),
}


def diagnostic_command(observation, history, rng):
    """Untrained stateless causal feedback fixture, not a boxing motor."""
    velocity = observation.values[79:158]
    prior = history.values[history.valid.astype(bool)]
    difference = (
        np.zeros(79) if len(prior) < 2 else prior[-1, 79:158] - prior[-2, 79:158]
    )
    feedback = np.tanh(velocity + 0.25 * difference)[np.arange(700) % 79]
    visible = next(
        b for b in FEATURE_SPEC if b["name"] == "opponent_body_linear_velocity_ego"
    )
    opponent = np.tanh(observation.values[visible["start"] : visible["stop"]])[
        np.arange(700) % 33
    ]
    return np.clip(
        -0.2 - 0.04 * feedback + 0.01 * opponent + rng.uniform(-0.01, 0.01, 700), -1, 1
    ).astype(np.float32)


def recording_cases():
    return [
        {
            "id": "source-90101",
            "seed": 90101,
            "scenario": "native-keyframe-v1",
            "horizon_ticks": 1000,
            "start_tick": 0,
            "end_tick": 1000,
            "physics": 1000,
        },
        {
            "id": "boundary-90101",
            "parent": "source-90101",
            "checkpoint": "boundary",
            "seed": 90101,
            "scenario": "native-keyframe-v1",
            "horizon_ticks": 1000,
            "start_tick": 200,
            "end_tick": 1000,
            "physics": 800,
        },
        {
            "id": "mid-90101",
            "parent": "source-90101",
            "checkpoint": "mid",
            "seed": 90101,
            "scenario": "native-keyframe-v1",
            "horizon_ticks": 1000,
            "start_tick": 203,
            "end_tick": 1000,
            "physics": 797,
        },
        {
            "id": "contact-source-90104",
            "seed": 90104,
            "scenario": "close-overlap-collision-probe-v2",
            "horizon_ticks": 20,
            "start_tick": 0,
            "end_tick": 20,
            "physics": 20,
        },
        {
            "id": "contact-replay-90104",
            "parent": "contact-source-90104",
            "checkpoint": "contact-mid",
            "seed": 90104,
            "scenario": "close-overlap-collision-probe-v2",
            "horizon_ticks": 20,
            "start_tick": 1,
            "end_tick": 20,
            "physics": 19,
        },
        {
            "id": "bell-90102",
            "seed": 90102,
            "scenario": "native-keyframe-v1",
            "horizon_ticks": 3,
            "start_tick": 0,
            "end_tick": 3,
            "physics": 3,
        },
        {
            "id": "truncation-90103",
            "seed": 90103,
            "scenario": "native-keyframe-v1",
            "horizon_ticks": 40,
            "start_tick": 0,
            "end_tick": 3,
            "physics": 3,
        },
    ]


def controller_bundles():
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        a: digest({"controller": "untrained-feedback-v1", "source": code, "corner": a})
        for a in PLAYERS
    }


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class BoxingTape:
    """At most 1000 intervals and 50 decisions; bounded before each append."""

    def __init__(self, env, *, maximum_ticks=1000):
        self.initial = env.snapshot()
        self.maximum_ticks = maximum_ticks
        self.decisions, self.intervals = [], []
        self.applied, self.excitations, self.packets, self.hashes = [], [], [], []
        self.offsets, self.contacts, self.geom_ids = [0], [], []
        self.decision_states, self.observations, self.histories = [], [], []
        self.history_ticks, self.history_received, self.history_valid = [], [], []
        self.chosen, self.decision_ticks, self.rngs = [], [], []

    def choose(self, env):
        if len(self.decisions) >= 50:
            raise TimeoutError("Diagnostic decision evidence cap")
        frame = env.decision_frame()  # Freeze both before either controller.
        before = {a: copy.deepcopy(env.rngs[a].bit_generator.state) for a in PLAYERS}
        commands = {a: diagnostic_command(*frame[a], env.rngs[a]) for a in PLAYERS}
        self.decision_states.append(env.decision_state.copy())
        self.observations.append(np.stack([frame[a][0].values for a in PLAYERS]))
        self.histories.append(np.stack([frame[a][1].values for a in PLAYERS]))
        self.history_ticks.append(
            np.stack([frame[a][1].sampled_ticks for a in PLAYERS])
        )
        self.history_received.append(
            np.stack(
                [
                    np.where(
                        frame[a][1].valid, env.tick - frame[a][1].receipt_age_ticks, -1
                    )
                    for a in PLAYERS
                ]
            )
        )
        self.history_valid.append(np.stack([frame[a][1].valid for a in PLAYERS]))
        self.chosen.append(np.stack([commands[a] for a in PLAYERS]))
        self.decision_ticks.append(env.tick)
        self.rngs.append(before)
        self.decisions.append(
            {
                "tick": env.tick,
                "points": list(env.referee.points),
                "game_id": env.game_id,
            }
        )
        return commands

    def capture(self, env):
        if len(self.intervals) >= self.maximum_ticks:
            raise TimeoutError("Diagnostic interval evidence cap")
        row = env.evidence[-1]
        if len(self.contacts) + len(row["contacts"]) > 150000:
            raise TimeoutError("Diagnostic contact evidence cap")
        if row["tick"] != self.initial["application"]["tick"] + len(self.intervals):
            raise ValueError("Missing/duplicate evidence interval")
        self.intervals.append(
            {k: copy.deepcopy(v) for k, v in row.items() if k != "contacts"}
        )
        self.applied.append(np.stack([env.commands[a] for a in PLAYERS]))
        self.excitations.append(
            np.stack([env.data.ctrl[env.bodies[a].actuator_ids] for a in PLAYERS])
        )
        self.packets.append(
            [
                [
                    env.queues[a].last_applied.decision_tick,
                    env.queues[a].last_applied.apply_tick,
                ]
                if env.queues[a].last_applied
                else [-1, -1]
                for a in PLAYERS
            ]
        )
        self.hashes.append(array_sha(env.state()))
        for c in row["contacts"]:
            self.geom_ids.append([c["geom1"], c["geom2"]])
            self.contacts.append(
                np.concatenate([np.asarray(c[k]).reshape(-1) for k in CONTACT_COLUMNS])
            )
        self.offsets.append(len(self.contacts))

    def save(self, env, folder, *, quarantined=False):
        folder = Path(folder)
        final = None if quarantined else env.snapshot()
        write_json(folder / "initial.json", self.initial)
        if final is not None:
            write_json(folder / "final.json", final)
        write_json(
            folder / "trace.json",
            {
                "schema": "joxing-shared-evidence-v1",
                "decisions": self.decisions,
                "rngs_before": self.rngs,
                "intervals": self.intervals,
                "state_sha256": self.hashes,
                "contact_columns": CONTACT_COLUMNS,
                "transition": {"invalid_reason": env.invalid_reason}
                if quarantined
                else {k: v for k, v in env.transition().items() if k != "observations"},
            },
        )
        width = len(env.state())
        save_npz(
            folder / "tape.npz",
            applied=np.asarray(self.applied, np.float32).reshape(-1, 2, 700),
            excitation=np.asarray(self.excitations, np.float64).reshape(-1, 2, 700),
            packets=np.asarray(self.packets, np.int64).reshape(-1, 2, 2),
            contact_offsets=np.asarray(self.offsets, np.int64),
            contact_geoms=np.asarray(self.geom_ids, np.int64).reshape(-1, 2),
            contacts=np.asarray(self.contacts, np.float64).reshape(-1, 18),
            decision_states=np.asarray(self.decision_states, np.float64).reshape(
                -1, width
            ),
            observations=np.asarray(self.observations, np.float32).reshape(-1, 2, 1970),
            histories=np.asarray(self.histories, np.float32).reshape(-1, 2, 8, 1970),
            history_ticks=np.asarray(self.history_ticks, np.int64).reshape(-1, 2, 8),
            history_received=np.asarray(self.history_received, np.int64).reshape(
                -1, 2, 8
            ),
            history_valid=np.asarray(self.history_valid, bool).reshape(-1, 2, 8),
            chosen=np.asarray(self.chosen, np.float32).reshape(-1, 2, 700),
            decision_ticks=np.asarray(self.decision_ticks, np.int64),
        )
        return {
            "physics": len(self.intervals),
            "final_sha256": None if final is None else final["sha256"],
            "positive_glove_touches": sum(
                t["normal_force_N"] > 0
                for row in self.intervals
                for g in row["gloves"].values()
                for t in g["touches"]
            ),
        }
