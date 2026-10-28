"""One fixed frozen-controller diagnostic, with actual held-command continuation.

Full native anatomy and candidate rules stay unchanged. A zero-point result is
retained and explained. No optimization, competitive promotion or seed search.
"""

import copy
from pathlib import Path

import numpy as np

from .boxing_evidence import CONTACT_COLUMNS, array_sha
from .evidence_storage import save_npz, write_json
from .provenance import digest


def encode_contacts(contacts):
    geoms = np.asarray([[c["geom1"], c["geom2"]] for c in contacts], np.int64).reshape(
        -1, 2
    )
    values = np.asarray(
        [
            np.concatenate([np.asarray(c[k]).reshape(-1) for k in CONTACT_COLUMNS])
            for c in contacts
        ],
        np.float64,
    ).reshape(-1, 18)
    return geoms, values


def choose(env, controllers):
    frame = env.decision_frame()  # Both observations frozen before either action.
    before_rng = {a: copy.deepcopy(env.rngs[a].bit_generator.state) for a in "AB"}
    before_policy = {a: controllers[a].snapshot() for a in "AB"}
    commands = {a: controllers[a](*frame[a], env.rngs[a]) for a in "AB"}
    env.policy_states = {a: controllers[a].snapshot() for a in "AB"}
    metadata = {
        "tick": env.tick,
        "rng_before": before_rng,
        "policy_before": before_policy,
        "history_sha256": {a: array_sha(frame[a][1].values) for a in "AB"},
        "history_ticks": {a: frame[a][1].sampled_ticks.tolist() for a in "AB"},
        "origin_after": copy.deepcopy(env.policy_states),
    }
    return commands, metadata, np.stack([frame[a][0].values for a in "AB"])


class NativeStrikeTape:
    def __init__(self, *, maximum_ticks=3000, maximum_contacts=150000):
        if not 1 <= maximum_ticks <= 3000 or not 1 <= maximum_contacts <= 150000:
            raise ValueError("Native strike evidence bound")
        self.maximum_ticks, self.maximum_contacts = maximum_ticks, maximum_contacts
        self.initial = self.mid = None
        self.intervals, self.decisions = [], []
        self.commands, self.observations, self.decision_engines = [], [], []
        self.video_ticks, self.video_states = [], []
        self.offsets, self.geoms, self.contacts = [0], [], []

    def run(self, env, controllers, *, seed, mid_tick=203, step_receipt=None):
        from grokbot.arena import policy_stream

        if not 0 < mid_tick < env.horizon_ticks <= self.maximum_ticks:
            raise ValueError("Native strike source/checkpoint extent")
        if {a: controllers[a].bundle_sha256 for a in "AB"} != env.bundles:
            raise ValueError("Frozen controllers differ from environment contract")
        env.reset(seed=seed)
        env.rngs = {
            "A": policy_stream(seed, "learner"),
            "B": policy_stream(seed, "opponent"),
        }
        env.policy_states = {a: controllers[a].snapshot() for a in "AB"}
        self.initial = env.snapshot()
        self.video_ticks.append(0)
        self.video_states.append(env.state().copy())
        while not env.done:
            if env.remaining_control == 0:
                commands, metadata, observations = choose(env, controllers)
                self.decisions.append(metadata)
                self.commands.append(np.stack([commands[a] for a in "AB"]))
                self.observations.append(observations)
                self.decision_engines.append(env.decision_state.copy())
                env.begin_control(commands)
            if step_receipt is not None:
                step_receipt("entered", env.tick)
            env.advance_substep()
            if step_receipt is not None:
                step_receipt("completed", env.tick - 1)
            row = env.evidence[-1]
            ids, values = encode_contacts(row["contacts"])
            if self.offsets[-1] + len(ids) > self.maximum_contacts:
                raise TimeoutError("Native strike contact allocation cap")
            self.geoms.append(ids)
            self.contacts.append(values)
            self.offsets.append(self.offsets[-1] + len(ids))
            self.intervals.append(
                {
                    "tick": row["tick"],
                    "gloves": copy.deepcopy(row["gloves"]),
                    "events": copy.deepcopy(row["events"]),
                    "engine_sha256": array_sha(env.state()),
                    "contact_geoms_sha256": array_sha(ids),
                    "contacts_sha256": array_sha(values),
                }
            )
            if env.tick == mid_tick:
                self.mid = env.snapshot()
            if env.tick % 40 == 0 or env.done:
                self.video_ticks.append(env.tick)
                self.video_states.append(env.state().copy())
            if len(self.intervals) > self.maximum_ticks:
                raise TimeoutError("Native strike interval allocation cap")
        return {
            "source_physics": len(self.intervals),
            "points": list(env.referee.points),
            "frames": len(self.video_ticks),
            "source_final_sha256": env.snapshot()["sha256"],
        }

    def save(self, env, folder, *, quarantined=False):
        folder = Path(folder)
        if self.initial is None:
            raise ValueError("Native source never initialized")
        write_json(folder / "initial.json", self.initial)
        write_json(folder / "contract.json", env.contract)
        if self.mid is not None:
            write_json(folder / "mid.json", self.mid)
        if not quarantined:
            write_json(folder / "final.json", env.snapshot())
        write_json(
            folder / "trace.json",
            {
                "schema": "joxing-native-strike-source-v1",
                "decisions": self.decisions,
                "intervals": self.intervals,
                "quarantined": quarantined,
                "classification": "frozen muscle-controller diagnostic, not qualified boxing",
            },
        )
        width = len(self.initial["engine"])
        save_npz(
            folder / "tape.npz",
            commands=np.asarray(self.commands, np.float32).reshape(-1, 2, 700),
            observations=np.asarray(self.observations, np.float32).reshape(-1, 2, 1970),
            decision_engines=np.asarray(self.decision_engines, np.float64).reshape(
                -1, width
            ),
            contact_offsets=np.array(self.offsets, np.int64),
            contact_geoms=np.concatenate(self.geoms)
            if self.geoms
            else np.empty((0, 2), np.int64),
            contacts=np.concatenate(self.contacts)
            if self.contacts
            else np.empty((0, 18), np.float64),
            video_ticks=np.array(self.video_ticks, np.int64),
            video_states=np.asarray(self.video_states, np.float64).reshape(-1, width),
        )


def continue_mid(env, controllers, source, *, step_receipt=None):
    """Actually execute the tail and compare each complete physical/contact byte hash.

    Hash receipts reference retained source values only after the corresponding
    actual tail step and command match. A mismatch raises, never counts as a draw.
    No actor call is permitted until the saved held command finishes.
    """
    if source.mid is None:
        raise ValueError("Missing native source checkpoint")
    env.restore(source.mid)
    for actor in "AB":
        controllers[actor].restore(env.policy_states[actor])
    start, held = env.tick, env.remaining_control
    lookup = {d["tick"]: i for i, d in enumerate(source.decisions)}
    actual, decisions = [], []
    while not env.done:
        if env.remaining_control == 0:
            if env.tick < start + held:
                raise ValueError("Unexpected actor decision during saved hold")
            commands, metadata, observations = choose(env, controllers)
            i = lookup[env.tick]
            if (
                metadata != source.decisions[i]
                or observations.tobytes() != source.observations[i].tobytes()
                or np.stack([commands[a] for a in "AB"]).tobytes()
                != source.commands[i].tobytes()
                or env.decision_state.tobytes() != source.decision_engines[i].tobytes()
            ):
                raise ValueError(
                    "Native continued decision/input/RNG/controller differs"
                )
            decisions.append(env.tick)
            env.begin_control(commands)
        tick = env.tick
        if step_receipt is not None:
            step_receipt("entered", tick)
        env.advance_substep()
        if step_receipt is not None:
            step_receipt("completed", tick)
        row = env.evidence[-1]
        ids, values = encode_contacts(row["contacts"])
        expected = source.intervals[tick]
        observed = {
            "tick": tick,
            "gloves": row["gloves"],
            "events": row["events"],
            "engine_sha256": array_sha(env.state()),
            "contact_geoms_sha256": array_sha(ids),
            "contacts_sha256": array_sha(values),
        }
        if observed != expected:
            raise ValueError(
                f"Native physical/contact/scoring continuation differs at {tick}"
            )
        actual.append(
            {
                "tick": tick,
                "actual_sha256": digest(observed),
                "expected_sha256": digest(expected),
            }
        )
    return {
        "status": "passed",
        "source_checkpoint_sha256": source.mid["sha256"],
        "start_tick": start,
        "end_tick": env.tick,
        "held_ticks_without_actor": held,
        "decision_ticks": decisions,
        "actual_physical_tail": actual,
        "new_physics": len(actual),
        "points": list(env.referee.points),
        "final_sha256": env.snapshot()["sha256"],
    }
