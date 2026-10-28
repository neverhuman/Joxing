"""Reuse authenticated full-model semantic metadata; never compile or step.

The binding artifact comes from an already passed regression using the intact
shared model. This avoids rebuilding three mesh-heavy MjSpecs just to obtain
integer IDs. It is not a proxy model, data-inferred mapping, or physics variant.
The caller must authenticate the artifact and its full-model test provenance.
"""

import copy

import numpy as np

from .evidence_storage import array_descriptor
from .motor_evidence_codec import require, validate_context
from .provenance import digest


def context_from_bindings(binding, trace):
    require(
        binding["schema"] == "joxing-compiled-model-bindings-v1", "Codec binding schema"
    )
    require(
        binding["scene_hash"] == trace["contract"]["scene_hash"],
        "Codec binding actual scene identity",
    )
    require(
        {k: binding["dimensions"][k] for k in ("nq", "nv", "nu", "na")}
        == {"nq": 170, "nv": 170, "nu": 1400, "na": 1400},
        "Codec binding complete native anatomy",
    )
    refs = np.asarray(
        [
            [
                np.r_[
                    row["references"][p]["qpos"],
                    np.asarray(row["references"][p]["future_qpos"]).ravel(),
                    np.asarray(
                        row["references"][p]["keypoints_relative_pelvis"]
                    ).ravel(),
                ]
                for p in ("A", "B")
            ]
            for row in trace["decisions"]
        ],
        np.float64,
    ).reshape(-1, 2, 528)
    c = {
        "schema": "joxing-motor-codec-context-v1",
        "trace_sha256": digest(trace),
        "scene_hash": binding["scene_hash"],
        "state_specification": binding["state_specification"],
        "state_width": binding["state_width"],
        "fields": copy.deepcopy(binding["fields"]),
        "normalizer": copy.deepcopy(trace["normalizer"]),
        "references": array_descriptor(refs),
        "maps": copy.deepcopy(binding["maps"]),
    }
    validate_context(c)
    return c, refs
