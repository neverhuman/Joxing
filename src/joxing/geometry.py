"""Measured strike evidence for the named sphere-target motor fixture.

All transforms below refer to the same cached force-evaluation stage. This is a
small fixture adapter; complete dual-human surface/zone fitting is still gated.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from .physics import Contact, point_velocity
from .rules import Touch


@dataclass
class StrikeEvidence:
    touches: list[Touch]
    clearance_m: float
    extension_m: float
    normal_force_N: float
    max_penetration_m: float
    glove_position: np.ndarray


class StrikeAdapter:
    def __init__(
        self, model, glove_geom: int, target_geom: int, trunk_body: int, zone="torso"
    ):
        if zone not in ("torso", "head", "guard"):
            raise ValueError("Unknown fixture target zone")
        self.model = model
        self.glove = glove_geom
        self.target = target_geom
        self.trunk = trunk_body
        self.zone = zone

    def measure(
        self, data, contacts: list[Contact], pre_qvel: np.ndarray
    ) -> StrikeEvidence:
        glove_body = self.model.geom_bodyid[self.glove]
        # Native hand -Y points distally through the closed-glove proxy.
        face_axis = data.xmat[glove_body].reshape(3, 3) @ np.array([0.0, -1.0, 0.0])
        glove_position = data.geom_xpos[self.glove].copy()
        trunk_matrix = data.xmat[self.trunk].reshape(3, 3)
        extension = float(
            (trunk_matrix.T @ (glove_position - data.xpos[self.trunk]))[0]
        )
        clearance = float(
            mujoco.mj_geomDistance(
                self.model, data, self.glove, self.target, 10.0, None
            )
        )
        touches = []
        force, penetration = 0.0, 0.0
        for i, contact in enumerate(contacts):
            if {contact.geom1, contact.geom2} != {self.glove, self.target}:
                continue
            normal = contact.normal if contact.geom1 == self.glove else -contact.normal
            velocity = (
                contact.velocity1_at_contact
                if contact.geom1 == self.glove
                else contact.velocity2_at_contact
            )
            trunk_velocity = point_velocity(
                self.model, data, self.trunk, contact.point, qvel=pre_qvel
            )
            active_speed = float(np.dot(velocity - trunk_velocity, normal))
            face_cos = float(np.dot(face_axis, normal))
            on_face = bool(np.dot(contact.point - glove_position, face_axis) > 0.0)
            touches.append(
                Touch(
                    str(i),
                    self.zone,
                    contact.normal_force_N,
                    contact.approach_mps,
                    active_speed,
                    face_cos,
                    on_face,
                )
            )
            force += contact.normal_force_N
            penetration = max(penetration, contact.penetration_m)
        return StrikeEvidence(
            touches, clearance, extension, force, penetration, glove_position
        )
