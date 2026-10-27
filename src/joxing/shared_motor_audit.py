"""Independent semantic readout of the shared native teacher inputs.

This deliberately does not call NativeMotorMap/native_teacher_observation. The
integration components are checked against a saved engine state. Solver caches
are actual read-only measurements at the completed solver phase, NOT values
derivable from an arbitrary post-step mj_forward. Exact physical continuation
must separately bind those caches to their trajectory; a self-consistent witness
alone is not a proof of the dynamics that produced it.
"""

import hashlib

import mujoco
import numpy as np

from .env import INTEGRATION

KEYS = ("sternum", "head_neck", "toes_r", "toes_l", "proximal_row_r", "proximal_row_l")
PLAYERS = ("A", "B")
CACHE_FIELDS = ("qacc", "actuator_force", "actuator_length", "actuator_velocity")


def byte_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()


class SemanticInputAudit:
    """Resolve names independently, retain the phase, check all 3601 features."""

    def __init__(self, shared, canonical):
        self.model = shared
        self.joints, self.muscles, self.activation, self.bodies = {}, {}, {}, {}
        if (canonical.nq, canonical.nv, canonical.nu, canonical.na) != (
            85,
            85,
            700,
            700,
        ):
            raise ValueError("Independent native anatomy")
        for p in PLAYERS:
            j = [shared.joint(p + "_" + canonical.joint(i).name).id for i in range(85)]
            a = [
                shared.actuator(p + "_" + canonical.actuator(i).name).id
                for i in range(700)
            ]
            if len(set(a)) != 700 or np.any(shared.actuator_actnum[a] != 1):
                raise ValueError("Independent muscle ownership")
            self.joints[p] = (shared.jnt_qposadr[j], shared.jnt_dofadr[j])
            self.muscles[p] = np.asarray(a)
            self.activation[p] = shared.actuator_actadr[a]
            self.bodies[p] = np.array(
                [shared.body(p + "_" + n).id for n in ("pelvis", *KEYS)]
            )
            # This proof supports its explicitly named zero-yaw parallel spawn.
            actual = shared.body(p + "_pelvis").quat
            expected = canonical.body("pelvis").quat
            if (
                min(np.max(abs(actual - expected)), np.max(abs(actual + expected)))
                > 1e-12
            ):
                raise ValueError("Independent parallel spawn yaw")
        if set(self.muscles["A"]) & set(self.muscles["B"]):
            raise ValueError("Overlapping motor owners")

    def witness(self, data, *, tick):
        """No forward/step/setters: preserve actual solver caches BEFORE callbacks."""
        if type(tick) is not int or tick % 20 or abs(data.time - tick * 0.001) > 1e-9:
            raise ValueError("Independent motor phase clock")
        return {
            "tick": tick,
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "act": data.act.copy(),
            **{k: getattr(data, k).copy() for k in CACHE_FIELDS},
            "key_xpos": np.stack([data.xpos[self.bodies[p]] for p in PLAYERS]),
        }

    def verify(self, witness, engine, inputs, references, *, game_id, tick):
        if set(witness) != {"tick", "qpos", "qvel", "act", *CACHE_FIELDS, "key_xpos"}:
            raise ValueError("Independent witness fields")
        if witness["tick"] != tick or type(tick) is not int or tick % 20:
            raise ValueError("Independent witness phase")
        if set(inputs) != set(PLAYERS) or set(references) != set(PLAYERS):
            raise ValueError("Independent input owners")
        shapes = {
            "qpos": (170,),
            "qvel": (170,),
            "act": (1400,),
            "qacc": (170,),
            "actuator_force": (1400,),
            "actuator_length": (1400,),
            "actuator_velocity": (1400,),
            "key_xpos": (2, 7, 3),
        }
        for k, shape in shapes.items():
            a = np.asarray(witness[k])
            if a.dtype != np.float64 or a.shape != shape or not np.isfinite(a).all():
                raise ValueError("Independent witness precision/shape: " + k)
        engine = np.asarray(engine)
        if (
            engine.dtype != np.float64
            or engine.shape != (mujoco.mj_stateSize(self.model, INTEGRATION),)
            or not np.isfinite(engine).all()
        ):
            raise ValueError("Independent engine precision/shape")
        scratch = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, scratch, engine, INTEGRATION)
        if abs(scratch.time - tick * 0.001) > 1e-9:
            raise ValueError("Independent engine phase")
        for k in ("qpos", "qvel", "act"):
            if not np.array_equal(getattr(scratch, k), witness[k]):
                raise ValueError("Independent input/engine mismatch: " + k)
        if np.any(scratch.qfrc_applied) or np.any(scratch.xfrc_applied):
            raise ValueError("Independent external assistance")
        for pi, p in enumerate(PLAYERS):
            sample = inputs[p]
            if (
                sample.player != p
                or sample.game_id != game_id
                or sample.sampled_tick != tick
            ):
                raise ValueError("Independent sample owner/game/phase")
            qp, qv = self.joints[p]
            a, act = self.muscles[p], self.activation[p]
            xyz = witness["key_xpos"][pi]
            ref = references[p]
            expected = np.concatenate(
                [
                    witness["qpos"][qp],
                    witness["qvel"][qv],
                    witness["qacc"][qv],
                    witness["act"][act],
                    np.clip(witness["actuator_force"][a] / 1000, -100, 100),
                    witness["actuator_length"][a],
                    np.clip(witness["actuator_velocity"][a], -100, 100),
                    (xyz[1:] - xyz[0]).ravel(),
                    ref.qpos,
                    ref.future_qpos.ravel(),
                    ref.keypoints_relative_pelvis.ravel(),
                ]
            ).astype(np.float32)
            if sample.values.tobytes() != expected.tobytes():
                raise ValueError("Independent semantic motor input: " + p)
        return {
            "passed": True,
            "integration_bound": ["qpos", "qvel", "activation"],
            "measured_solver_cache": [*CACHE_FIELDS, "key_xpos"],
            "cache_dynamics_bound_by_separate_exact_continuation": True,
        }
