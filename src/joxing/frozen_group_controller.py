"""Snapshot-capable bridge to a frozen diagnostic regional muscle controller.

The historical GroupPD controller uses own posture/velocity, not opponent state
or temporal memory. This bridge preserves that limitation and supplies exact
origin/RNG ownership for a new diagnostic; it is not a tactical boxing learner.
"""

import copy
import hashlib
import json

import numpy as np

from .provenance import ROOT, digest


class FrozenGroupController:
    def __init__(self, params, *, noise=0.05):
        from grokbot import motor_primitives, selfplay

        values = np.array(params, dtype=np.float64, copy=True)
        if not np.isfinite(values).all() or not np.isfinite(noise) or noise < 0:
            raise ValueError("Finite frozen controller required")
        values.flags.writeable = False
        self._params = values
        self._actor = motor_primitives.GroupPDPolicy(values, noise=float(noise))
        self._origin_dimension = len(selfplay.FEATURES)
        names = [
            "src/joxing/frozen_group_controller.py",
            "grokbot/motor_primitives.py",
            "grokbot/selfplay.py",
            "grokbot/arena.py",
            "third_party/msgym/DynSyn-SAC/configs/locomotionFull.json",
        ]
        self.bundle = {
            "schema": "joxing-frozen-group-controller-v1",
            "kind": "GroupPDPolicy",
            "noise": float(noise),
            "parameters_dtype": "<f8",
            "parameters_shape": list(values.shape),
            "parameters_sha256": hashlib.sha256(
                values.astype("<f8").tobytes()
            ).hexdigest(),
            "code_and_group_files": {
                n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names
            },
            "feature_indices": selfplay.FEATURES.tolist(),
            "groups": [g.tolist() for g in motor_primitives.GROUPS],
            "normalizer": "none; raw own features minus saved origin",
            "uses_opponent": False,
            "uses_history": False,
            "output": "float32[700] normalized commands; env owns excitation transform",
        }
        self.bundle_sha256 = digest(self.bundle)

    @classmethod
    def from_summary(cls, path, *, expected_sha256, noise=0.05):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("Frozen policy summary hash differs")
        return cls(json.loads(data)["final_params"], noise=noise)

    def __call__(self, observation, history, rng):
        # History is deliberately ignored by the historical policy, not erased
        # from the environment or described as learned opponent awareness.
        command = self._actor(observation, None, rng)
        if (
            command.dtype != np.float32
            or command.shape != (700,)
            or not np.isfinite(command).all()
            or np.any(np.abs(command) > 1)
        ):
            raise ValueError("Frozen muscle command contract failed")
        return command.copy()

    def snapshot(self):
        origin = self._actor.punch.posture.origin
        return {
            "schema": "joxing-frozen-group-state-v1",
            "bundle_sha256": self.bundle_sha256,
            "origin": None if origin is None else origin.tolist(),
        }

    def restore(self, state):
        saved = copy.deepcopy(state)
        if (
            set(saved) != {"schema", "bundle_sha256", "origin"}
            or saved["schema"] != "joxing-frozen-group-state-v1"
            or saved["bundle_sha256"] != self.bundle_sha256
        ):
            raise ValueError("Foreign frozen controller state")
        origin = (
            None
            if saved["origin"] is None
            else np.array(saved["origin"], dtype=np.float64)
        )
        if origin is not None and (
            origin.shape != (self._origin_dimension,) or not np.isfinite(origin).all()
        ):
            raise ValueError("Malformed frozen feature origin")
        self._actor.punch.posture.origin = origin
