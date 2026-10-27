"""Candidate allocation-reduced v8 collector; no different physics or rules.

The reference PhysicalMotorTask remains unchanged. This candidate preserves
per-substep checks, scalar contact matmul and accumulation order, owned complete
contact rows, scratch posture and held-action timing. Qualification is required.
"""

import mujoco
import numpy as np

from .physical_motor import ROLES, PhysicalMotorTask, SubstepMotorEvidence


def floor_evidence(model, data, floor, roles, wrench):
    loads = np.zeros(3)
    contacts = []
    for slot, contact in enumerate(data.contact):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if contact.efc_address < 0 or floor not in (g1, g2):
            continue
        other = g2 if g1 == floor else g1
        role = ROLES.index(roles[other])  # Keep unknown/mutated ownership a fault.
        wrench.fill(
            0.0
        )  # Same zero initial values as reference, even on sparse writes.
        mujoco.mj_contactForce(model, data, slot, wrench)
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        if g2 == floor:
            force = -force
        position = contact.pos
        if not np.isfinite(force).all() or not np.isfinite(position).all():
            raise RuntimeError("Nonfinite substep contact evidence")
        loads[role] += max(0.0, float(force[2]))
        # Python float owns the same float64 value; no mutable engine view or
        # scratch buffer escapes. Native scalars avoid recursive NumPy deepcopy
        # in the unchanged per-control snapshot's dataclasses.asdict path.
        contacts.append(
            (g1, g2, role, *(float(v) for v in force), *(float(v) for v in position))
        )
    return tuple(float(v) for v in loads), tuple(contacts)


class AllocationReducedTask(PhysicalMotorTask):
    """Named implementation candidate; same compiled and replay contract as v8."""

    def __init__(self, config, variant):
        super().__init__(config, variant)
        self._contact_wrench = np.zeros(6)

    def advance_substeps(self, count):
        if (
            not self.pending
            or type(count) is not int
            or not 0 <= count <= self.env.frame_skip - self.substep
        ):
            raise ValueError("Invalid pending substep count")
        for _ in range(count):
            self.check_assistance()
            if not np.array_equal(self.data.ctrl, self.excitation):
                raise RuntimeError("Held control changed")
            warnings = tuple(w.number for w in self.data.warning)
            before = float(self.data.time)
            self.physics_total += 1
            mujoco.mj_step(self.model, self.data)
            if (
                abs(self.data.time - before - self.model.opt.timestep) > 1e-9
                or any(
                    w.number > previous
                    for w, previous in zip(self.data.warning, warnings, strict=True)
                )
                or not all(
                    np.isfinite(a).all()
                    for a in (self.data.qpos, self.data.qvel, self.data.act)
                )
            ):
                raise RuntimeError("Invalid physical motor substep")
            self.scratch.qpos[:] = self.data.qpos
            mujoco.mj_kinematics(self.model, self.scratch)
            up = self.scratch.xmat[self.pelvis].reshape(3, 3)[:, 1]
            tilt = float(np.arccos(np.clip(up[2], -1, 1)))
            drop = self.initial_height - float(self.scratch.xpos[self.pelvis, 2])
            loads, contacts = floor_evidence(
                self.model, self.data, self.floor, self.roles, self._contact_wrench
            )
            row = SubstepMotorEvidence(
                before, float(self.data.time), loads, tilt, drop, contacts
            )
            reason = self.monitor.update(
                dict(zip(ROLES, loads, strict=True)),
                self.weight,
                tilt,
                drop,
                float(self.model.opt.timestep),
            )
            self.max_tilt = max(self.max_tilt, tilt)
            if reason is not None and self.physical_failure is None:
                self.physical_failure = reason
                self.physical_failure_s = row.end_s
            self.sampled_failure = self.physical_failure
            self.interval_evidence.append(row)
            self.evidence.append(row)
            self.substep += 1
            if (
                self.capture_mid_snapshot
                and self.registered_mid_snapshot is None
                and self.completed_controls == 10
                and self.substep == 3
            ):
                self.registered_mid_snapshot = self.snapshot()
