"""Signed floor support and elapsed-time motor-failure evidence, not boxing rules."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SupportMonitor:
    nonfoot_s: float = 0.0
    tilt_s: float = 0.0
    drop_s: float = 0.0

    def update(self, loads, weight_N, tilt_rad, drop_m, dt):
        if (
            not np.isfinite([*loads.values(), weight_N, tilt_rad, drop_m, dt]).all()
            or dt <= 0
            or weight_N <= 0
        ):
            raise ValueError("Invalid support evidence")
        self.nonfoot_s = (
            self.nonfoot_s + dt if loads["nonfoot"] >= 0.1 * weight_N else 0.0
        )
        self.tilt_s = self.tilt_s + dt if tilt_rad >= np.pi / 3 else 0.0
        self.drop_s = self.drop_s + dt if drop_m >= 0.30 else 0.0
        for reason, elapsed, threshold in (
            ("nonfoot_support", self.nonfoot_s, 0.05),
            ("excess_tilt", self.tilt_s, 0.10),
            ("pelvis_drop", self.drop_s, 0.10),
        ):
            if elapsed + 1e-12 >= threshold:
                return reason
        return None

    def snapshot(self):
        return asdict(self)


def floor_loads(contacts, floor_id, geom_roles):
    loads = {"right": 0.0, "left": 0.0, "nonfoot": 0.0}
    for c in contacts:
        if c.geom1 == floor_id:
            other, force = c.geom2, c.force_on_geom2
        elif c.geom2 == floor_id:
            other, force = c.geom1, -c.force_on_geom2
        else:
            continue
        role = geom_roles[
            other
        ]  # Unknown physical surfaces fail, never default to feet.
        loads[role] += max(0.0, float(force[2]))
    return loads
