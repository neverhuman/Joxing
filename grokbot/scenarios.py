"""Named grokbot reset variant: the native keyframe moved to boxing range.

``BoxingEnv`` registers three resets. The native keyframe starts the pelvises
1.6 m apart, where the athletes never touch. The close-overlap stress probes
start at 0.2 m, where the bodies interpenetrate from tick 0. Neither is a
boxing start. ``RangeBoxingEnv`` reuses the env's own reset-only mechanism,
which shifts each athlete's root ``pelvis_tx`` along its own facing direction
before the first observation, to start at ``separation_m``. It changes
nothing else: geometry, masses, muscles, poses, rules and every evaluated step
stay native.

The variant is explicit: it gets its own scenario label, and the label and
shift are part of the contract hash. It is a grokbot candidate, not a
registered jekko scenario.
"""

from __future__ import annotations

from joxing.env import BoxingEnv
from joxing.provenance import digest

NATIVE_SEPARATION_M = 1.6


class RangeBoxingEnv(BoxingEnv):
    def __init__(self, scene=None, *, separation_m=1.0, **kwargs):
        if not 0.4 <= separation_m <= NATIVE_SEPARATION_M:
            raise ValueError("Separation must be within [0.4, 1.6] m")
        # Construct through a registered non-native reset so the env applies its
        # own reset-only root shift, then relabel and re-hash the contract.
        super().__init__(scene, scenario="close-overlap-collision-probe-v1", **kwargs)
        self.separation_m = float(separation_m)
        self.scenario = f"grokbot-range-{separation_m:.2f}m-v0"
        self.contract = self.contract | {
            "reset_scenario": self.scenario,
            "reset_local_root_shift_m": (NATIVE_SEPARATION_M - separation_m) / 2,
        }
        self.contract_hash = digest(self.contract)
