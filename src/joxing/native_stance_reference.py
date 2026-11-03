"""Opt-in own-motion goal experiment. Never change real time, pose or velocity.

The released controller was trained to walk, not stand. This profile is a
hypothesis requiring physical qualification. The original walking driver and
its arithmetic, reset targets and contracts remain unchanged.
"""

import copy

import numpy as np

from .motor_residual import reference_hash
from .provenance import digest
from .shared_motor_task import NativeReferenceDriver

PROFILE = "walk-to-hold400ms-v1"
STOP_TIME_S = 0.4


def reference_contract():
    return {
        "profile": PROFILE,
        "stop_time_s": STOP_TIME_S,
        "phase_clock": "f(t)=t-t*t/(2*T) for 0<=t<T; T/2 thereafter; seconds",
        "reset_targets": "init_time+f(j*0.02), j=0..5; native reset lead=0",
        "later_targets": "init_time+f(engine_time+0.02+j*0.02), j=0..5",
        "real_clock": "unchanged 1ms physics,20ms muscle holds",
        "actual_reset": "unchanged native walking qpos/qvel/activation; no zeroing",
        "native_axes": "source X-forward/Y-up, shared world Z-up; NativeMotorMap",
        "classification": "own reference goal experiment; not qualified standing",
    }


def stopped_phase(seconds):
    """Map finite nonnegative physical seconds to bounded reference seconds."""
    t = np.asarray(seconds, np.float64)
    if not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("Nonnegative finite phase seconds required")
    # Clip BEFORE squaring, including very large finite times.
    x = np.minimum(t, STOP_TIME_S)
    return np.where(t >= STOP_TIME_S, STOP_TIME_S / 2, x - x * x / (2 * STOP_TIME_S))


class WalkToHoldReferenceDriver(NativeReferenceDriver):
    profile = PROFILE

    def __post_init__(self):
        super().__post_init__()
        model = self.task.model
        self.identity = reference_contract() | {
            "trajectory_sha256": reference_hash(self.task.env.trajectory),
            "native_axes_sha256": digest(
                {
                    "joints": [model.joint(i).name for i in range(model.njnt)],
                    "axis": model.jnt_axis.tolist(),
                    "type": model.jnt_type.tolist(),
                    "qpos0": model.qpos0.tolist(),
                    "qposadr": model.jnt_qposadr.tolist(),
                    "dofadr": model.jnt_dofadr.tolist(),
                }
            ),
        }
        self.identity_sha256 = digest(self.identity)
        self.reference = self._query(0, 0.0)
        self.initial_reference = self.reference

    def _check_identity(self):
        if (
            self.profile != PROFILE
            or digest(self.identity) != self.identity_sha256
            or any(self.identity.get(k) != v for k, v in reference_contract().items())
        ):
            raise ValueError("Native stopped-reference identity drift")

    def _query(self, tick, engine_time):
        # The source reset uses current target at init_time, not init_time+.02.
        # Preserve that convention, while warping its future OWN desired targets.
        lead = 0.0 if tick == 0 else 0.02
        offsets = engine_time + lead + np.arange(6, dtype=np.float64) * 0.02
        times = self.init_time + stopped_phase(offsets)
        qb, xb, _ = self.task.env.trajectory.query_batch(times, self.index)
        return self._state(qb[0], xb[0], qb[1:6])

    def sample(self, tick, engine_time):
        self._check_identity()
        if (
            type(tick) is not int
            or tick < self.tick
            or tick % 20
            or not np.isfinite(engine_time)
            or abs(engine_time - tick * 0.001) > 1e-9
        ):
            raise ValueError("Reference sample clock")
        self.reference = (
            self.initial_reference if tick == 0 else self._query(tick, engine_time)
        )
        self.tick, self.sample_time = tick, float(engine_time)
        return self.reference

    def snapshot(self):
        self._check_identity()
        return super().snapshot() | {
            "reference_profile": copy.deepcopy(self.identity),
            "reference_profile_sha256": self.identity_sha256,
        }
