"""Explicit raw actor features for the diagnostic shared scene, not a teacher port.

Opponent features contain visible kinematics only. Raw proprioception and the
previous applied command belong to the observing actor. No critic input exists
on this assembly path. Normalization must be a separately frozen later bundle.
"""

from dataclasses import dataclass

import numpy as np

from .bodies.base import BodyMeasurements, owned_readonly
from .bodies.native import KEY_BODIES
from .provenance import digest

XYZ = 3 * len(KEY_BODIES)
ROTATION_VALUES = 9 * len(KEY_BODIES)
BLOCKS = (
    ("own_intrinsic_joint_position", 79, "native joint units"),
    ("own_intrinsic_joint_velocity", 79, "native joint units/s"),
    ("own_muscle_activation", 700, "activation"),
    ("own_body_position_ego", XYZ, "m"),
    ("own_body_linear_velocity_ego", XYZ, "m/s"),
    ("own_body_rotation_ego", ROTATION_VALUES, "rotation matrix"),
    ("own_body_angular_velocity_ego", XYZ, "rad/s"),
    ("opponent_body_position_ego", XYZ, "m"),
    ("opponent_body_linear_velocity_ego", XYZ, "m/s"),
    ("opponent_body_rotation_ego", ROTATION_VALUES, "rotation matrix"),
    ("opponent_body_angular_velocity_ego", XYZ, "rad/s"),
    ("round_remaining", 1, "s"),
    ("own_opponent_points", 2, "points"),
    ("own_pelvis_height", 1, "m"),
    ("ring_vertices_ego", 12, "m"),
    ("own_previous_applied_command", 700, "legacy normalized700 command"),
)
FEATURE_SPEC = []
offset = 0
for name, length, units in BLOCKS:
    FEATURE_SPEC.append(
        {"name": name, "start": offset, "stop": offset + length, "units": units}
    )
    offset += length
DIMENSION = offset
SCHEMA = {
    "schema": "joxing-actor-observation-diagnostic-v1",
    "blocks": FEATURE_SPEC,
    "bodies": KEY_BODIES,
    "dtype": "float32",
    "frame": "yaw-only observer pelvis; world gravity/height preserved",
    "normalization": "none",
    "opponent_velocity": "absolute world velocity rotated into observer yaw frame",
}
SCHEMA_HASH = digest(SCHEMA)


@dataclass(frozen=True)
class PublicState:
    game_id: str
    tick: int
    remaining_s: float
    points: tuple[int, int]
    ring_vertices_world_m: np.ndarray
    dt_s: float = 0.001

    def __post_init__(self):
        if not self.game_id or type(self.tick) is not int or self.tick < 0:
            raise ValueError("Invalid public game clock")
        if (
            not np.isfinite(self.dt_s)
            or self.dt_s <= 0
            or not np.isfinite(self.remaining_s)
            or not 0 <= self.remaining_s <= 60
        ):
            raise ValueError("Invalid public timing")
        if len(self.points) != 2 or any(
            type(x) is not int or x < 0 for x in self.points
        ):
            raise ValueError("Invalid public points")
        object.__setattr__(self, "points", tuple(self.points))
        if np.shape(self.ring_vertices_world_m) != (4, 3):
            raise ValueError("Four actual ring vertices required")
        object.__setattr__(
            self, "ring_vertices_world_m", owned_readonly(self.ring_vertices_world_m)
        )


@dataclass(frozen=True)
class ActorObservation:
    game_id: str
    player: str
    sampled_tick: int
    schema_hash: str
    values: np.ndarray


@dataclass(frozen=True)
class CriticState:
    """Separate future critic payload. Actor assembly never consumes this type."""

    game_id: str
    tick: int
    values: np.ndarray


def observe(
    own: BodyMeasurements,
    opponent: BodyMeasurements,
    public: PublicState,
    previous_command,
):
    if not isinstance(own, BodyMeasurements) or not isinstance(
        opponent, BodyMeasurements
    ):
        raise TypeError("Actor assembly requires physical body measurements")
    if (
        own.actor not in {"A", "B"}
        or opponent.actor != ({"A": "B", "B": "A"}[own.actor])
    ):
        raise ValueError("Incorrect corner ownership")
    if (
        own.time_s != opponent.time_s
        or abs(own.time_s - public.tick * public.dt_s) > 1e-9
    ):
        raise ValueError("Both actors must observe the same completed physics state")
    if own.body_names != KEY_BODIES or opponent.body_names != KEY_BODIES:
        raise ValueError("Foreign body feature layout")
    previous = np.asarray(previous_command)
    if (
        previous.shape != (700,)
        or not np.isfinite(previous).all()
        or np.any(np.abs(previous) > 1)
    ):
        raise ValueError("Invalid previous applied own command")
    forward = own.world_rotation_matrix[0, :, 0]
    norm = np.linalg.norm(forward[:2])
    if norm < 1e-6 or not np.isfinite(norm):
        raise ValueError(
            "Undefined observer yaw; quarantine instead of inventing orientation"
        )
    x, y = forward[0] / norm, forward[1] / norm
    world_to_ego = np.array([[x, y, 0], [-y, x, 0], [0, 0, 1]])
    origin = own.world_position_m[0]
    values = [own.qpos[6:], own.qvel[6:], own.activation]
    for body in (own, opponent):
        values.extend(
            [
                (body.world_position_m - origin) @ world_to_ego.T,
                body.world_linear_velocity_mps @ world_to_ego.T,
                np.einsum("ij,njk->nik", world_to_ego, body.world_rotation_matrix),
                body.world_angular_velocity_radps @ world_to_ego.T,
            ]
        )
    points = public.points if own.actor == "A" else public.points[::-1]
    ring = (public.ring_vertices_world_m - origin) @ world_to_ego.T
    order = np.argsort(np.arctan2(ring[:, 1], ring[:, 0]), kind="stable")
    values.extend(
        [
            np.array([public.remaining_s]),
            np.array(points),
            np.array([origin[2]]),
            ring[order],
            previous,
        ]
    )
    vector = np.concatenate([v.reshape(-1) for v in values]).astype(np.float32)
    if vector.shape != (DIMENSION,):
        raise ValueError("Actor feature dimensions differ from declared schema")
    return ActorObservation(
        public.game_id, own.actor, public.tick, SCHEMA_HASH, owned_readonly(vector)
    )
