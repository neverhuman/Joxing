"""Bounded, lossless shared-motor evidence; no policy or evaluated engine writes.

All physical intervals retain end qpos and signed contacts. Decisions retain
actual raw/normalized inputs, independently named solver-cache measurements,
engine state and all applied commands. Only the first fixed source is filmed.
"""

import copy
from pathlib import Path

import mujoco
import numpy as np

from .boxing_evidence import CONTACT_COLUMNS
from .env import excitation
from .evidence_storage import save_npz, write_json
from .shared_motor_audit import PLAYERS, SemanticInputAudit, byte_sha
from .support import SupportMonitor

ROLES = ("right", "left", "nonfoot")


class MotorTape:
    def __init__(self, task, *, recording=False, continuation=None):
        self.initial = task.snapshot()
        self.semantic = SemanticInputAudit(task.model, task.drivers["A"].task.model)
        self.decisions, self.intervals, self.hashes = [], [], []
        self.offsets, self.geoms, self.contacts = [0], [], []
        self.qpos, self.posture, self.loads = [], [], []
        self.monitors = {p: SupportMonitor() for p in PLAYERS}
        self.failure = dict.fromkeys(PLAYERS)
        self.tracking = dict.fromkeys(PLAYERS)
        self.initial_height = {}
        self.scratch = mujoco.MjData(task.model)
        self.floor = task.model.geom("floor").id
        self.roles, self.weights = {}, {}
        for p in PLAYERS:
            owned = {
                i
                for i in range(task.model.nbody)
                if task.model.body(i).name.startswith(p + "_")
            }
            self.weights[p] = float(sum(task.model.body_mass[list(owned)]) * 9.81)
            self.roles[p] = {}
            for i in range(task.model.ngeom):
                if int(task.model.geom_bodyid[i]) not in owned:
                    continue
                name = task.model.body(int(task.model.geom_bodyid[i])).name
                # The same explicit body names as NativeTask, scoped per athlete.
                self.roles[p][i] = (
                    "right"
                    if name in (p + "_calcn_r", p + "_toes_r")
                    else "left"
                    if name in (p + "_calcn_l", p + "_toes_l")
                    else "nonfoot"
                )
            self.initial_height[p] = float(
                task.data.xpos[task.model.body(p + "_pelvis").id, 2]
            )
        self.video = [(task.tick, task.state())] if recording else []
        self.recording = recording
        self.start_tick = task.tick
        self.last_valid_engine = task.state()
        self.mid = None
        if task.tick:
            if continuation is None or continuation["tick"] != task.tick:
                raise ValueError("Replay requires its actual prefix support state")
            self.initial_height = copy.deepcopy(continuation["initial_height"])
            self.monitors = {
                p: SupportMonitor(**continuation["monitors"][p]) for p in PLAYERS
            }
            self.failure = copy.deepcopy(continuation["physical_failure"])
            self.tracking = copy.deepcopy(continuation["tracking_failure"])
        elif continuation is not None:
            raise ValueError("Initial source must not inherit fabricated support")
        self.prefix = self.support_snapshot(task)

    def support_snapshot(self, task):
        return {
            "tick": task.tick,
            "initial_height": copy.deepcopy(self.initial_height),
            "monitors": {p: m.snapshot() for p, m in self.monitors.items()},
            "physical_failure": copy.deepcopy(self.failure),
            "tracking_failure": copy.deepcopy(self.tracking),
        }

    def collect_decision(self, task):
        """Capture actual pre-inference input, then both scalar decisions once."""
        if task.remaining or task.done:
            raise ValueError("Evidence requires an actual decision boundary")
        if self.decisions and self.decisions[-1]["tick"] >= task.tick:
            raise ValueError("Duplicate motor decision")
        witness = self.semantic.witness(task.data, tick=task.tick)
        engine = task.state()
        self.semantic.verify(
            witness,
            engine,
            task.inputs,
            {p: task.drivers[p].reference for p in PLAYERS},
            game_id=task.game_id,
            tick=task.tick,
        )
        raw = np.stack([task.inputs[p].values for p in PLAYERS])
        # FrozenNormalizer.apply is pure; teacher verifies the same state before/after.
        task.teacher.normalizer.check()
        normalized = np.stack([task.teacher.normalizer.apply(v) for v in raw])
        task.teacher.normalizer.check()
        refs = {p: task.drivers[p].snapshot() for p in PLAYERS}
        chosen = task.begin_control()
        self.decisions.append(
            {
                "tick": task.tick,
                "engine": engine,
                "raw": raw,
                "normalized": normalized,
                "witness": witness,
                "references": refs,
                "commands": np.stack([chosen[p] for p in PLAYERS]),
                "excitation": np.stack(
                    [task.data.ctrl[task.maps[p].actuator_ids] for p in PLAYERS]
                ),
            }
        )

    def capture_substep(self, task, contacts):
        expected = self.start_tick + len(self.intervals) + 1
        if task.tick != expected:
            raise ValueError("Missing/duplicate physical motor interval")
        self.intervals.append(task.tick)
        self.hashes.append(byte_sha(task.state()))
        self.last_valid_engine = task.state()
        self.qpos.append(task.data.qpos.copy())
        self.scratch.qpos[:] = task.data.qpos
        mujoco.mj_kinematics(
            task.model, self.scratch
        )  # Scratch posture only; no physics.
        posture, loads = [], []
        for p in PLAYERS:
            pid = task.model.body(p + "_pelvis").id
            up = self.scratch.xmat[pid].reshape(3, 3)[:, 1]
            tilt = float(np.arccos(np.clip(up[2], -1, 1)))
            drop = self.initial_height[p] - float(self.scratch.xpos[pid, 2])
            force = np.zeros(3)
            for c in contacts:
                if c.geom1 == self.floor and c.geom2 in self.roles[p]:
                    other, f = c.geom2, c.force_on_geom2
                elif c.geom2 == self.floor and c.geom1 in self.roles[p]:
                    other, f = c.geom1, -c.force_on_geom2
                else:
                    continue
                force[ROLES.index(self.roles[p][other])] += max(0.0, float(f[2]))
            reason = self.monitors[p].update(
                dict(zip(ROLES, force, strict=True)), self.weights[p], tilt, drop, 0.001
            )
            if reason and self.failure[p] is None:
                self.failure[p] = {"tick": task.tick, "reason": reason}
            posture.append([tilt, drop])
            loads.append(force)
        self.posture.append(posture)
        self.loads.append(loads)
        for c in contacts:
            self.geoms.append([c.geom1, c.geom2])
            self.contacts.append(
                np.r_[
                    c.point,
                    c.normal,
                    c.force_on_geom2,
                    c.velocity1_at_contact,
                    c.velocity2_at_contact,
                    c.normal_force_N,
                    c.approach_mps,
                    c.penetration_m,
                ]
            )
        self.offsets.append(len(self.contacts))
        if not task.remaining:
            # Native termination compares the just-finished interval with its OLD
            # reference. _capture has already updated to the next-control target.
            old = (
                self.decisions[-1]["references"]
                if self.decisions
                else self.initial["references"]
            )
            for p in PLAYERS:
                error = float(
                    np.mean(
                        abs(
                            task.data.qpos[task.maps[p].qpos_ids][3:]
                            - np.asarray(old[p]["qpos"])[3:]
                        )
                    )
                )
                if (
                    error > task.drivers[p].task.env.qpos_diff_th
                    and self.tracking[p] is None
                ):
                    self.tracking[p] = {"tick": task.tick, "error": error}
        if self.recording and (task.tick % 40 == 0 or task.done):
            self.video.append((task.tick, task.state()))

    def finish_recording(self, task):
        """Include the last valid frame of an early failure, without clip selection."""
        tick = self.start_tick + len(self.intervals)
        if self.recording and self.video[-1][0] != tick:
            self.video.append((tick, self.last_valid_engine.copy()))

    def arrays(self, task):
        width = len(task.state())
        ds = self.decisions
        arrays = {
            "decision_ticks": np.array([d["tick"] for d in ds], np.int64),
            "decision_engines": np.array([d["engine"] for d in ds], np.float64).reshape(
                -1, width
            ),
            "raw": np.array([d["raw"] for d in ds], np.float32).reshape(-1, 2, 3601),
            "normalized": np.array([d["normalized"] for d in ds], np.float32).reshape(
                -1, 2, 3601
            ),
            "commands": np.array([d["commands"] for d in ds], np.float32).reshape(
                -1, 2, 700
            ),
            "excitation": np.array([d["excitation"] for d in ds], np.float64).reshape(
                -1, 2, 700
            ),
            "ticks": np.array(self.intervals, np.int64),
            "end_qpos": np.array(self.qpos, np.float64).reshape(-1, task.model.nq),
            "posture": np.array(self.posture, np.float64).reshape(-1, 2, 2),
            "loads": np.array(self.loads, np.float64).reshape(-1, 2, 3),
            "contact_offsets": np.array(self.offsets, np.int64),
            "contact_geoms": np.array(self.geoms, np.int64).reshape(-1, 2),
            "contacts": np.array(self.contacts, np.float64).reshape(-1, 18),
            "video_ticks": np.array([t for t, _ in self.video], np.int64),
            "video_states": np.array([s for _, s in self.video], np.float64).reshape(
                -1, width
            ),
        }
        for k, shape in {
            "qpos": (170,),
            "qvel": (170,),
            "act": (1400,),
            "qacc": (170,),
            "actuator_force": (1400,),
            "actuator_length": (1400,),
            "actuator_velocity": (1400,),
            "key_xpos": (2, 7, 3),
        }.items():
            arrays["witness_" + k] = np.array(
                [d["witness"][k] for d in ds], np.float64
            ).reshape(-1, *shape)
        return arrays

    def metadata(self, task):
        task.teacher.normalizer.check()
        return {
            "schema": "joxing-shared-motor-evidence-v1",
            "contract": task.contract,
            "initial": self.initial,
            "mid": self.mid,
            "last_valid": task.snapshot() if not task.invalid else None,
            "decisions": [
                {"tick": d["tick"], "references": copy.deepcopy(d["references"])}
                for d in self.decisions
            ],
            "state_sha256": self.hashes,
            "contact_columns": CONTACT_COLUMNS,
            "floor_roles": {
                p: {str(k): v for k, v in r.items()} for p, r in self.roles.items()
            },
            "initial_height": self.initial_height,
            "weight_N": self.weights,
            "support_prefix": self.prefix,
            "physical_failure": self.failure,
            "tracking_failure": self.tracking,
            "competition": False,
            "optimizer_updates": 0,
            "normalizer": task.teacher.normalizer.export(),
            "cache_scope": "Actual completed-solver measurements; integration-bound qpos/qvel/act, separate exact trajectory continuation required for caches",
        }

    def save(self, task, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "trace.json", self.metadata(task))
        save_npz(folder / "tape.npz", **self.arrays(task))
        return {
            "ticks": len(self.intervals),
            "decisions": len(self.decisions),
            "physical_failure": self.failure,
            "tracking_failure": self.tracking,
            "video_frames": len(self.video),
        }


def check_actions(arrays):
    commands = arrays["commands"]
    if (
        commands.dtype != np.float32
        or not np.isfinite(commands).all()
        or np.any(abs(commands) > 1)
    ):
        raise ValueError("Saved native commands")
    if not np.array_equal(
        arrays["excitation"], excitation(commands).astype(np.float64)
    ):
        raise ValueError("Saved excitation transform")
