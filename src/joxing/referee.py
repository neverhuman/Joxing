"""Shared candidate points ledger; scene/sport qualification remains blocked."""

import copy
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .physics import point_velocity
from .rules import HitTracker, Rules, Touch


@dataclass(frozen=True)
class GloveInterval:
    touches: tuple[Touch, ...]
    clearance_m: float
    extension_m: float
    physical_evidence: dict | None = None


class MatchReferee:
    def __init__(self, scene, game_id):
        self.scene, self.game_id = scene, game_id
        self.rules = Rules(dt_s=float(scene.model.opt.timestep))
        self.registry = {g["id"]: g for g in scene.report["physical_geoms"]}
        self.gloves = {
            g["name"]: g["id"] for g in self.registry.values() if g["role"] == "glove"
        }
        if len(self.gloves) != 4:
            raise ValueError("Four semantic gloves required")
        self.trackers = {
            name: HitTracker(game_id, name, self.rules) for name in self.gloves
        }
        self.next_tick, self.ended = 0, False

    @property
    def events(self):
        return sorted(
            [
                copy.deepcopy(e)
                for t in self.trackers.values()
                for e in t.ledger.values()
            ],
            key=lambda e: e["event_id"],
        )

    @property
    def points(self):
        return tuple(
            sum(
                e["points"]
                for name, t in self.trackers.items()
                if name.startswith(actor + "_")
                for e in t.ledger.values()
            )
            for actor in ("A", "B")
        )

    def stage(self, data, contacts, pre_qvel):
        """Use cached force-evaluation geometry, before any post-step forward."""
        model = self.scene.model
        grouped = {name: [] for name in self.gloves}
        trunk_velocities = {name: [] for name in self.gloves}
        for index, contact in enumerate(contacts):
            if contact.geom1 not in self.registry or contact.geom2 not in self.registry:
                raise ValueError("Unknown physical contact semantic")
            for name, glove in self.gloves.items():
                if glove not in (contact.geom1, contact.geom2):
                    continue
                target = contact.geom2 if glove == contact.geom1 else contact.geom1
                other = self.registry[target]
                if other["actor"] in (None, name[0]):
                    continue
                normal = contact.normal if glove == contact.geom1 else -contact.normal
                velocity = (
                    contact.velocity1_at_contact
                    if glove == contact.geom1
                    else contact.velocity2_at_contact
                )
                body = model.geom_bodyid[glove]
                axis = data.xmat[body].reshape(3, 3) @ np.array([0.0, -1.0, 0.0])
                trunk = model.body(name[0] + "_thoracic12").id
                trunk_velocity = point_velocity(
                    model, data, trunk, contact.point, qvel=pre_qvel
                )
                role = other["role"]
                zone = (
                    "guard"
                    if role in {"guard", "glove"}
                    else role
                    if role in {"head", "torso"}
                    else "native_non_scoring"
                )
                grouped[name].append(
                    Touch(
                        f"{index}:{contact.geom1}:{contact.geom2}",
                        zone,
                        contact.normal_force_N,
                        contact.approach_mps,
                        float(np.dot(velocity - trunk_velocity, normal)),
                        float(np.dot(axis, normal)),
                        bool(np.dot(contact.point - data.geom_xpos[glove], axis) > 0),
                    )
                )
                trunk_velocities[name].append(trunk_velocity.tolist())
        result = {}
        for name, glove in self.gloves.items():
            trunk = model.body(name[0] + "_thoracic12").id
            extension = float(
                (
                    data.xmat[trunk].reshape(3, 3).T
                    @ (data.geom_xpos[glove] - data.xpos[trunk])
                )[0]
            )
            # Idle gloves do not need an expensive all-surface distance query.
            # During an encounter, clear distance covers every opponent collider.
            clearance = 0.0
            if (
                not any(t.normal_force_N > 0 for t in grouped[name])
                and self.trackers[name].current is not None
            ):
                opponent = "B" if name[0] == "A" else "A"
                clearance = min(
                    float(mujoco.mj_geomDistance(model, data, glove, gid, 10.0, None))
                    for gid, g in self.registry.items()
                    if g["actor"] == opponent
                )
            body = model.geom_bodyid[glove]
            result[name] = GloveInterval(
                tuple(grouped[name]),
                clearance,
                extension,
                {
                    "axis_world": (
                        data.xmat[body].reshape(3, 3) @ np.array([0.0, -1.0, 0.0])
                    ).tolist(),
                    "glove_position_world": data.geom_xpos[glove].tolist(),
                    "trunk_position_world": data.xpos[trunk].tolist(),
                    "trunk_rotation_world": data.xmat[trunk].reshape(3, 3).tolist(),
                    "trunk_point_velocities_world": trunk_velocities[name],
                    "stage": "cached force-evaluation geometry, before post-step forward",
                },
            )
        return result

    def commit(self, intervals, tick, *, bell=False):
        if self.ended:
            raise RuntimeError("Referee already ended")
        if tick != self.next_tick or set(intervals) != set(self.trackers):
            raise ValueError("Missing global referee interval/glove")
        for interval in intervals.values():
            if not np.isfinite([interval.clearance_m, interval.extension_m]).all():
                raise ValueError("Invalid geometric recovery evidence")
        before = {e["event_id"] for e in self.events}
        for name in sorted(self.trackers):
            row = intervals[name]
            self.trackers[name].update(
                tick,
                list(row.touches),
                clearance_m=row.clearance_m,
                extension_m=row.extension_m,
            )
        self.next_tick += 1
        if bell:
            # All last pre-bell intervals were integrated above. Finalize at
            # their end boundary without a new physics interval or impulse.
            for name in sorted(self.trackers):
                row = intervals[name]
                self.trackers[name].update(
                    self.next_tick,
                    [],
                    clearance_m=row.clearance_m,
                    extension_m=row.extension_m,
                    bell=True,
                )
            self.ended = True
        return [e for e in self.events if e["event_id"] not in before]

    def snapshot(self):
        return {
            "schema": "joxing-match-referee-candidate-v1",
            "game_id": self.game_id,
            "rules": asdict(self.rules),
            "next_tick": self.next_tick,
            "ended": self.ended,
            "trackers": {name: t.snapshot() for name, t in self.trackers.items()},
        }

    @classmethod
    def restore(cls, scene, state):
        r = cls(scene, state["game_id"])
        if (
            state["schema"] != "joxing-match-referee-candidate-v1"
            or Rules(**state["rules"]).hash != r.rules.hash
            or set(state["trackers"]) != set(r.trackers)
        ):
            raise ValueError("Foreign match referee snapshot")
        n = state["next_tick"]
        if type(n) is not int or n < 0 or type(state["ended"]) is not bool:
            raise ValueError("Malformed referee clock")
        restored = {
            name: HitTracker.restore(copy.deepcopy(s))
            for name, s in state["trackers"].items()
        }
        for name, t in restored.items():
            if (
                t.episode_id != r.game_id
                or t.glove_id != name
                or t.rules.hash != r.rules.hash
                or t.last_tick != (n if state["ended"] else n - 1)
            ):
                raise ValueError("Inconsistent per-glove referee state")
        r.trackers = restored
        r.next_tick = n
        r.ended = state["ended"]
        return r
