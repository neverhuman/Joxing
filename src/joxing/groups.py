"""Immutable named full-body partition; no dependency on upstream Python imports."""

import json

import numpy as np

from .provenance import ROOT, digest
from .scene import SOURCE


class MuscleGroups:
    def __init__(self, model):
        import mujoco

        source = mujoco.MjModel.from_xml_path(str(SOURCE))
        path = ROOT / "third_party/msgym/DynSyn-SAC/configs/locomotionFull.json"
        indices = json.loads(path.read_text())["agent_kwargs"]["policy_kwargs"][
            "dynsyn"
        ]
        flattened = [i for group in indices for i in group]
        if sorted(flattened) != list(range(700)) or len(indices) != 138:
            raise ValueError("Expected exact 700-muscle/138-group partition")
        self.names = tuple(tuple(source.actuator(i).name for i in g) for g in indices)
        self.ids = tuple(
            np.array([model.actuator(n).id for n in g]) for g in self.names
        )
        self.hash = digest({"names": self.names, "transform": "sigmoid(5*(a-.5))"})

    def expand(self, action):
        action = np.asarray(action, dtype=float)
        if (
            action.shape != (138,)
            or not np.isfinite(action).all()
            or np.any(abs(action) > 1)
        ):
            raise ValueError("Expected 138 finite normalized actions in [-1,1]")
        excitation = 1 / (1 + np.exp(-5 * (action - 0.5)))
        ctrl = np.empty(700)
        for ids, value in zip(self.ids, excitation):
            ctrl[ids] = value
        return ctrl
