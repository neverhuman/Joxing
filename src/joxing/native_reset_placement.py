"""Fixed scene transforms center actual native resets; never steer live bodies.

The native teacher sees its original scalar coordinates/references/velocities.
Only two fixed root frames move, before evaluated MjData exists. The reference
walking clock and every other compiled model array remain unchanged.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .native_pair_motor_rl import NativePairMotorEnv, PairMotorConfig
from .provenance import ROOT, digest
from .scene import build_scene
from .shared_motor import NativeMotorMap
from .shared_motor_inputs import FrozenPairTeacher
from .shared_motor_task import NativeReferenceDriver

MODES = ("reset-centered-opposed-v1", "reset-centered-lanes-v1")
SEEDS = (71000, 71016, 71032, 71048)
QUERY_MAX_M = 0.05
MIN_CLEARANCE_M = 0.02
RECIPE = "compile original opposed constants; translate only fixed root XY to actual reset targets; no mj_setConst/XML-only rebuild"


@dataclass(frozen=True)
class NativeResetPlacementSpec:
    mode: str

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("Unknown explicit native reset placement")

    def targets(self):
        return {
            "A": [-0.8, -1.25 if self.mode == MODES[1] else 0.0],
            "B": [0.8, 1.25 if self.mode == MODES[1] else 0.0],
        }


def checked_initial(initial_by_player):
    if not isinstance(initial_by_player, dict) or set(initial_by_player) != set("AB"):
        raise ValueError("Both native reset owners required")
    output = {}
    for p in "AB":
        row = initial_by_player[p]
        if set(row) != {"qpos", "qvel", "activation"}:
            raise ValueError("Complete native reset state required")
        output[p] = {}
        for field, shape in (("qpos", (85,)), ("qvel", (85,)), ("activation", (700,))):
            a = np.asarray(row[field])
            if a.dtype != np.float64 or a.shape != shape or not np.isfinite(a).all():
                raise ValueError("Finite exact native reset vector required")
            output[p][field] = np.frombuffer(a.tobytes(), np.float64)
    return output


def initial_hashes(initial):
    return {
        p: {k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in row.items()}
        for p, row in initial.items()
    }


def reset_data(model, canonical, initial):
    """Scratch or actual reset initialization ONLY; no physics or zeroed speed."""
    data = mujoco.MjData(model)
    maps = {
        p: NativeMotorMap(
            model, canonical, p, spawn_yaw_rad=0.0 if p == "A" else float(np.pi)
        )
        for p in "AB"
    }
    for p, mapping in maps.items():
        data.qpos[mapping.qpos_ids] = initial[p]["qpos"]
        data.qvel[mapping.qvel_ids] = initial[p]["qvel"]
        data.act[mapping.activation_ids] = initial[p]["activation"]
    mujoco.mj_forward(model, data)
    return data, maps


def reset_geometry(model, data):
    """All cross-player physical surface distances, including native meshes.

    mj_geomDistance returns a signed distance capped at QUERY_MAX_M. A capped
    value is a LOWERBOUND, never an exact clearance. Native CCD is mandatory.
    """
    if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD):
        raise ValueError("Native CCD required for signed distance qualification")
    groups = {
        p: [
            i
            for i in range(model.ngeom)
            if model.geom(i).name.startswith(p + "_")
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
        for p in "AB"
    }
    if not all(groups.values()):
        raise ValueError("Missing physical reset surfaces")
    allowed = {
        int(x)
        for x in (
            mujoco.mjtGeom.mjGEOM_SPHERE,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_MESH,
        )
    }
    if any(
        int(model.geom_type[g]) not in allowed for ids in groups.values() for g in ids
    ):
        raise ValueError("Unsupported reset clearance geometry")
    minimum, closest, query_count, penetration_count = QUERY_MAX_M, None, 0, 0
    for a in groups["A"]:
        for b in groups["B"]:
            distance = float(
                mujoco.mj_geomDistance(model, data, a, b, QUERY_MAX_M, None)
            )
            if not np.isfinite(distance):
                raise ValueError("Nonfinite reset signed distance")
            query_count += 1
            penetration_count += distance < 0
            if distance < minimum:
                minimum, closest = distance, [model.geom(a).name, model.geom(b).name]
    positions = {p: data.xpos[model.body(p + "_pelvis").id].tolist() for p in "AB"}
    return {
        "world_pelvis_xyz_m": positions,
        "physical_surface_counts": {p: len(ids) for p, ids in groups.items()},
        "all_cross_player_pairs_queried": query_count,
        "query_max_m": QUERY_MAX_M,
        "minimum_signed_clearance_lower_bound_m": minimum,
        "closest_below_query_max_surfaces": closest,
        "penetrating_pair_count": penetration_count,
        "clearance_qualified": minimum >= MIN_CLEARANCE_M and penetration_count == 0,
    }


def build_reset_scene(canonical_model, initial_by_player, *, spec, seed):
    if (
        not isinstance(spec, NativeResetPlacementSpec)
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("Explicit placement/seed required")
    initial = checked_initial(initial_by_player)
    scene = build_scene()
    scratch, maps = reset_data(scene.model, canonical_model, initial)
    before = {p: scratch.xpos[scene.model.body(p + "_pelvis").id].copy() for p in "AB"}
    offsets = {}
    targets = spec.targets()
    for p in "AB":
        delta = np.asarray(targets[p]) - before[p][:2]
        offsets[p] = delta.tolist()
        scene.model.body_pos[scene.model.body(p + "_pelvis").id, :2] += delta
        scene.spec.frame(p + "_spawn").pos[:2] += delta
    # A separate scratch forward, not evaluated pose writes, validates placement.
    scratch, _ = reset_data(scene.model, canonical_model, initial)
    geometry = reset_geometry(scene.model, scratch)
    for p in "AB":
        actual = geometry["world_pelvis_xyz_m"][p]
        if (
            np.max(abs(np.asarray(actual[:2]) - targets[p])) > 1e-9
            or actual[2] != before[p][2]
        ):
            raise ValueError("Actual reset placement/height mismatch")
    if not geometry["clearance_qualified"]:
        raise ValueError("Initial interactor clearance failed")
    record = {
        "schema": "joxing-native-reset-placement-v1",
        "mode": spec.mode,
        "seed": seed,
        "recipe": RECIPE,
        "actual_target_xy_m": targets,
        "fixed_frame_offset_xy_m": offsets,
        "native_reset_sha256": initial_hashes(initial),
        "native_axes_sha256": digest(
            {
                "joint_names": [
                    canonical_model.joint(i).name for i in range(canonical_model.njnt)
                ],
                "axis": canonical_model.jnt_axis.tolist(),
                "map_rotations": {p: maps[p].native_to_world.tolist() for p in "AB"},
            }
        ),
        "fixed_root_positions": {
            p: scene.model.body_pos[scene.model.body(p + "_pelvis").id].tolist()
            for p in "AB"
        },
        "fixed_spec_frame_positions": {
            p: scene.spec.frame(p + "_spawn").pos.tolist() for p in "AB"
        },
        "initial_geometry": geometry,
    }
    scene.report.update(
        profile="pretrained-" + spec.mode,
        reset_placement=record,
        scene_hash=digest({"xml": scene.spec.to_xml(), "reset_placement": record}),
    )
    return scene


class PlacedNativePairMotorEnv(NativePairMotorEnv):
    """One immutable fixed-seed placement per world. No mutable-scene Vec reset."""

    def __init__(self, scene, *args, **kwargs):
        record = scene.report["reset_placement"]
        self._placement_sha256 = digest(record)
        self._placement_seed = record["seed"]
        self._root_ids = [scene.model.body(p + "_pelvis").id for p in "AB"]
        self._root_positions = scene.model.body_pos[self._root_ids].tobytes()
        self._spec_positions = np.stack(
            [scene.spec.frame(p + "_spawn").pos for p in "AB"]
        ).tobytes()
        super().__init__(scene, *args, **kwargs)

    def _placement_check(self):
        if (
            digest(self.scene.report["reset_placement"]) != self._placement_sha256
            or self.scene.model.body_pos[self._root_ids].tobytes()
            != self._root_positions
            or np.stack(
                [self.scene.spec.frame(p + "_spawn").pos for p in "AB"]
            ).tobytes()
            != self._spec_positions
        ):
            raise ValueError("Fixed native reset placement drift")

    def _check(self):
        self._placement_check()
        super()._check()

    def reset(self, *, seed):
        if type(seed) is not int or seed != self._placement_seed:
            raise ValueError("Scene/native reset seed mismatch")
        result = super().reset(seed=seed)
        initial = {p: d.initial for p, d in self.core.drivers.items()}
        if (
            initial_hashes(initial)
            != self.scene.report["reset_placement"]["native_reset_sha256"]
        ):
            raise ValueError("Native reset state/placement binding mismatch")
        return result

    def restore(self, snapshot):
        self._placement_check()
        super().restore(snapshot)


class ResetPlacementFactory:
    """Load actor once; serial fixed-seed worlds share no evaluated MjData."""

    def __init__(self, inputs):
        from .motor_residual import reference_hash
        from .native_transfer import load_controller
        from .upstream_motor import NativeTask
        from .wave import canonical_actor_hash

        self.model, self.normalizer = load_controller(Path(inputs))
        self.model.actor.eval()
        self.model.actor.requires_grad_(False)
        self.donor = NativeTask(
            json.loads((ROOT / "experiments/v0.0.6/protocol.json").read_text())[
                "environment"
            ]
        )
        self.bundle = {
            "actor": canonical_actor_hash(self.model.actor),
            "normalizer": self.normalizer.fingerprint,
            "reference": reference_hash(self.donor.env.trajectory),
        }
        self.groups = self.model.actor.dynsyn_layer.muscle_groups

    def make(self, *, seed, mode, guard=None, horizon=3000):
        spec = NativeResetPlacementSpec(mode)
        initial = {p: NativeReferenceDriver(self.donor, seed).initial for p in "AB"}
        scene = build_reset_scene(self.donor.model, initial, spec=spec, seed=seed)

        def drivers(requested_seed):
            if type(requested_seed) is not int or requested_seed != seed:
                raise ValueError("Fixed native reference seed mismatch")
            return {p: NativeReferenceDriver(self.donor, requested_seed) for p in "AB"}

        return PlacedNativePairMotorEnv(
            scene,
            drivers,
            FrozenPairTeacher(self.model, self.normalizer),
            self.groups,
            self.bundle,
            config=PairMotorConfig(placement=mode, horizon_ticks=horizon),
            guard=guard,
        )

    def close(self):
        self.donor.close()
