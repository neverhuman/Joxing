"""Explicit equipment variants of the frozen teacher; never a boxing environment."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from .provenance import ROOT, digest, manifest, write_json
from .stance_train import file_hash

# experiment boundary exceptions are persisted before halting peers
# ruff: noqa: BLE001
from .upstream_motor import (
    MODEL_ROOT,
    FrozenNormalizer,
    NativeTask,
    load_dynsyn,
    pinned_import,
    set_decoder_mode,
)

PROTOCOL_SHA = "cd07fccb4d600d67b0f97c5f557430a2613b688280e7bf26bb0dd2b27fb94d83"
KEYS = ("sternum", "head_neck", "toes_r", "toes_l", "proximal_row_r", "proximal_row_l")
VARIANTS = ("native_2ms", "gloves_2ms", "native_1ms", "gloves_1ms")


def protocol():
    return json.loads((ROOT / "experiments/v0.0.7/protocol.json").read_text())


def validate_protocol(p):
    if digest(p) != PROTOCOL_SHA:
        raise ValueError("Registered v7 protocol changed")
    return copy.deepcopy(p)


@dataclass(frozen=True)
class ReferenceBodyMap:
    """Reference indices are canonical; actual IDs belong to the variant model."""

    names: tuple[str, ...]
    actual_ids: tuple[int, ...]

    @classmethod
    def from_names(cls, canonical, actual):
        if len(set(canonical)) != len(canonical) or len(set(actual)) != len(actual):
            raise ValueError("Duplicate body names")
        if set(canonical) - set(actual):
            raise ValueError("Missing canonical body")
        return cls(tuple(canonical), tuple(actual.index(n) for n in canonical))

    def indices(self, names):
        reference = [self.names.index(n) for n in names]
        return reference, [self.actual_ids[i] for i in reference]

    def export(self):
        return [
            {"name": n, "reference_index": i, "actual_id": self.actual_ids[i]}
            for i, n in enumerate(self.names)
        ]


def names(model, kind):
    return [
        getattr(model, kind)(i).name
        for i in range(
            getattr(
                model,
                "n"
                + {
                    "body": "body",
                    "joint": "jnt",
                    "actuator": "u",
                    "tendon": "tendon",
                    "equality": "eq",
                }[kind],
            )
        )
    ]


def native_invariants(canonical, model, mapping, variant):
    if (model.nq, model.nv, model.nu, model.na) != (85, 85, 700, 700):
        raise ValueError("Anatomy dimensions changed")
    for kind in ("joint", "actuator", "tendon", "equality"):
        if names(canonical, kind) != names(model, kind):
            raise ValueError(f"Native {kind} ordering changed")
    checks = {}
    for key in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        error = float(
            np.max(
                abs(
                    getattr(canonical, key)
                    - getattr(model, key)[list(mapping.actual_ids)]
                )
            )
        )
        if error > 1e-12:
            raise ValueError(f"Native {key} changed: {error}")
        checks[key + "_max_error"] = error
    for key in (
        "jnt_type",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "actuator_dynprm",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_lengthrange",
        "eq_data",
        "eq_type",
    ):
        if not np.array_equal(getattr(canonical, key), getattr(model, key)):
            raise ValueError(f"Native {key} changed")
    increment = float(sum(model.body_mass) - sum(canonical.body_mass))
    if abs(increment - (0.566 if variant["gloves"] else 0)) > 1e-12:
        raise ValueError("Equipment mass changed")
    if model.nbody != canonical.nbody + (2 if variant["gloves"] else 0):
        raise ValueError("Unexpected equipment bodies")
    if variant["gloves"]:
        for side in ("r", "l"):
            b = model.body("transfer_glove_" + side)
            if (
                model.body(model.body_parentid[b.id]).name != "hand_" + side
                or abs(float(b.mass[0]) - 0.283) > 1e-12
                or not np.allclose(
                    b.inertia, 0.4 * 0.283 * 0.062**2, rtol=0, atol=1e-12
                )
            ):
                raise ValueError("Glove ownership/inertia changed")
    return checks | {
        "mass_increment_kg": increment,
        "muscles": model.nu,
        "joint_names": names(model, "joint"),
        "actuator_names": names(model, "actuator"),
        "tendon_names": names(model, "tendon"),
        "equality_names": names(model, "equality"),
    }


def build_variant(variant):
    if variant not in protocol()["variants"]:
        raise ValueError("Unknown physical variant")
    source = MODEL_ROOT / "MS-Human-700.xml"
    canonical = mujoco.MjModel.from_xml_path(str(source))
    spec = mujoco.MjSpec.from_file(str(source))
    spec.option.timestep = variant["physics_dt_s"]
    if variant["gloves"]:
        for side, color in (("r", [1, 0.15, 0.1, 1]), ("l", [0.1, 0.3, 1, 1])):
            body = spec.body("hand_" + side).add_body(
                name="transfer_glove_" + side, pos=[0, -0.045, 0]
            )
            body.mass = 0.283
            body.inertia = np.full(3, 0.4 * 0.283 * 0.062**2)
            body.add_geom(
                name="transfer_glove_" + side,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.062, 0, 0],
                mass=0,
                contype=1,
                conaffinity=1,
                rgba=color,
            )
    model = spec.compile()
    mapping = ReferenceBodyMap.from_names(
        names(canonical, "body"), names(model, "body")
    )
    invariant = native_invariants(canonical, model, mapping, variant)
    return model, mapping, invariant, spec.to_xml()


def mapped_observation(env, mapping):
    ref_keys, actual_keys = mapping.indices(KEYS)
    ref_pelvis, actual_pelvis = mapping.indices(["pelvis"])
    key = env.data.xpos[actual_keys].copy() - env.data.xpos[actual_pelvis[0]].copy()
    key_ref = env.xpos_ref[ref_keys].copy() - env.xpos_ref[ref_pelvis[0]].copy()
    env.key_xpos_ref = key_ref.flat.copy()
    blocks = {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "qacc": env.data.qacc.copy(),
        "act": env.data.act.flat.copy(),
        "actuator_forces": (env.data.actuator_force.flat.copy() / 1000).clip(-100, 100),
        "actuator_length": env.data.actuator_length.flat.copy(),
        "actuator_velocity": env.data.actuator_velocity.flat.copy().clip(-100, 100),
        "key_xpos": key.flat.copy(),
        "qpos_ref": env.qpos_ref,
        "qpos_ref_future": env.qpos_ref_future.flat.copy(),
        "key_xpos_ref": env.key_xpos_ref,
    }
    return np.concatenate(list(blocks.values())).astype(np.float32, copy=False), blocks


def make_environment(config, variant, model, mapping):
    module = pinned_import("msgym.envs.locomotionFull_v1")

    class MappedEnv(module.LocomotionFullEnvV1):
        metadata = copy.deepcopy(module.LocomotionFullEnvV1.metadata)
        reference_body_count = len(mapping.names)

        def _initialize_simulation(self):
            # Use MjSpec's compiled model directly; XML pretty-printing must not
            # round native physical parameters on a save/reparse path.
            model.vis.global_.offwidth = self.width
            model.vis.global_.offheight = self.height
            return model, mujoco.MjData(model)

        def _get_obs(self):
            return mapped_observation(self, mapping)

        def step(self, action):
            if self.kinematic_play:
                raise RuntimeError("Kinematic transfer is forbidden")
            self.do_simulation(action, self.frame_skip)
            rk, ak = mapping.indices(KEYS)
            rp, ap = mapping.indices(["pelvis"])
            pelvis, pelvis_ref = self.data.xpos[ap[0]], self.xpos_ref[rp[0]]
            qpos = (
                self._get_qpos_reward(self.data.qpos, self.qpos_ref)
                * self.reward_weight["w_qpos"]
            )
            xpos = (
                self._get_xpos_reward(
                    self.data.xpos[ak] - pelvis, self.xpos_ref[rk] - pelvis_ref
                )
                * self.reward_weight["w_xpos"]
            )
            pelvis_reward = (
                self._get_pelvis_reward(pelvis, pelvis_ref)
                * self.reward_weight["w_pelvis"]
            )
            energy = self._get_energy_reward() * self.reward_weight["w_energy"]
            healthy = self._get_healthy_reward() * self.reward_weight["w_healthy"]
            reward = qpos + xpos + pelvis_reward + energy + healthy
            terminated = self.terminated
            truncated = self.data.time >= self.terminate_time * self.cycles
            ref_time = self.data.time + self.init_time + self.dt
            times = (
                ref_time
                + np.arange(self.future_traj_steps + 1, dtype=np.float64) * self.dt
            )
            qb, xb, _ = self.trajectory.query_batch(times, self.current_traj_index)
            self.qpos_ref, self.xpos_ref = qb[0], xb[0]
            self.qpos_ref_future[:, :] = qb[1 : 1 + self.future_traj_steps]
            obs, _ = self._get_obs()
            return (
                obs,
                reward,
                terminated,
                truncated,
                {
                    "reward_qpos": qpos,
                    "reward_xpos": xpos,
                    "reward_pelvis": pelvis_reward,
                    "reward_energy": energy,
                    "reward_healthy": healthy,
                    "total_reward": reward,
                },
            )

    cfg = copy.deepcopy(config)
    cfg["skip_frames"] = round(0.02 / variant["physics_dt_s"])
    with (
        patch.object(
            module, "get_ms_human_model_path", lambda name: str(MODEL_ROOT / name)
        ),
        patch.object(module, "get_render_fps", lambda *_: 50),
    ):
        return MappedEnv(**cfg)


class TransferTask(NativeTask):
    def __init__(self, config, variant):
        self.variant = copy.deepcopy(variant)
        model, self.mapping, self.invariants, self.xml = build_variant(variant)
        env = make_environment(config, variant, model, self.mapping)
        super().__init__(config, environment=env, physics_dt=variant["physics_dt_s"])

    def specification(self):
        base = super().specification()
        base.update(
            variant=self.variant,
            reference_body_map=self.mapping.export(),
            reference_body_count=len(self.mapping.names),
            native_invariants=self.invariants,
            xml_sha256=__import__("hashlib").sha256(self.xml.encode()).hexdigest(),
            model_path_adapter="MjSpec compiled native model plus explicit gloves/dt; canonical reference indices",
        )
        return base

    def paired_fields(self):
        return {
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "activation": self.data.act.tolist(),
            "phase": float(self.env.init_time),
            "rng": copy.deepcopy(self.env.np_random.bit_generator.state),
        }


def feature_oracle(config, seed=73091, steps=12):
    variant = protocol()["variants"][0]
    a, b = NativeTask(config), TransferTask(config, variant)
    try:
        a.reset(seed)
        b.reset(seed)
        errors = []
        for i in range(steps + 1):
            oa, blocks = a.env._get_obs()
            ob, mapped = b.env._get_obs()
            if list(blocks) != list(mapped):
                raise ValueError("Feature order changed")
            errors.append(
                max(float(np.max(abs(blocks[k] - mapped[k]))) for k in blocks)
            )
            if (
                not np.array_equal(oa, ob)
                or a.snapshot()["state"] != b.snapshot()["state"]
            ):
                raise RuntimeError("Native state/feature oracle failed")
            if i == steps:
                break
            action = np.full(700, -0.6, dtype=np.float32)
            ra, rb = a.step(action), b.step(action)
            if ra[1:5] != rb[1:5]:
                raise RuntimeError("Native reward/termination oracle failed")
            if a.done:
                break
        return {
            "passed": True,
            "max_feature_error": max(errors),
            "steps_compared": len(errors),
            "physics_steps": a.physics_total + b.physics_total,
            "seed": seed,
            "scope": "Complete feature/state/reward/native termination equivalence; bounded synthetic action probe.",
        }
    finally:
        a.close()
        b.close()


def planned_rows(p, variant=None):
    return [
        {
            "attempt_id": f"{v['id']}:{seed}",
            "variant": v["id"],
            "seed": seed,
            "status": "not_run",
            "reason": "pending_dependencies",
        }
        for v in p["variants"]
        if variant is None or v["id"] == variant
        for seed in p["seeds"]
    ]


def summarize(rows):
    p = protocol()
    if len(rows) != 80 or {r["attempt_id"] for r in rows} != {
        r["attempt_id"] for r in planned_rows(p)
    }:
        raise ValueError("Missing/duplicate attempted denominator")
    groups = {}
    for v in VARIANTS:
        subset = [r for r in rows if r["variant"] == v]
        done = [r for r in subset if r["status"] == "completed"]
        groups[v] = {
            "planned": 20,
            "completed": len(done),
            **{
                s: sum(r["status"] == s for r in subset)
                for s in ("invalid", "incomplete", "not_run")
            },
            "horizon_completions": sum(r["horizon_reached"] for r in done),
            "mean_elapsed_sim_s": float(np.mean([r["elapsed_sim_s"] for r in done]))
            if done
            else None,
            "sampled_support_failure_episodes": sum(
                r["sampled_failure"] is not None for r in done
            ),
        }
    return {
        "variants": groups,
        "attempts": 80,
        "invalid": sum(r["status"] == "invalid" for r in rows),
        "all_attempts_completed": all(r["status"] == "completed" for r in rows),
        "completion_thresholds_met": all(
            g["completed"] == 20 and g["horizon_completions"] >= 16
            for g in groups.values()
        ),
        "scope": "Frozen long native locomotion; no training, stance, guard or boxing qualification.",
    }


def load_controller(inputs):
    import torch

    from .wave import canonical_actor_hash

    torch.set_num_threads(1)
    if (
        file_hash(inputs / "model.zip")
        != "0a3607c0a9d745a6bb5a5ffceb7539a7405d455943abf283d4f8107f22a47829"
    ):
        raise ValueError("Unknown actor archive member")
    if (
        file_hash(inputs / "normalizer.pkl")
        != "e359b32fa026a02746cffcba6ee3ef9cab4389d37c74b19bd21caa0f3abc0086"
    ):
        raise ValueError("Unknown normalizer member")
    model = load_dynsyn(inputs / "model.zip")
    if (
        canonical_actor_hash(model.actor)
        != "963d2f56c5887f79e094ddf998316730475f979efb100073237cda81817e272e"
    ):
        raise ValueError("Actor tensors changed")
    if set_decoder_mode(model, "scheduled_restored") != 0.075:
        raise ValueError("Decoder schedule changed")
    n = FrozenNormalizer.load(inputs / "normalizer.pkl", 3601)
    if (
        n.fingerprint
        != "34f38e6e87544d77b87b9a155fe6b9996629fd671f235b773c27f004d6fd5df8"
    ):
        raise ValueError("Normalizer statistics changed")
    return model, n


def predict(model, normalizer, obs):
    normalized = normalizer.apply(obs)
    action, hidden = model.predict(normalized, deterministic=True)
    if hidden is not None:
        raise RuntimeError("Unexpected hidden policy state")
    return action, float(np.mean(abs(normalized) >= normalizer.normalizer.clip_obs))


def source_host_oracle(p):
    old = Path(p["prerequisite"]["v6_run"])
    audit = json.loads((old.parent / "audit-linux.json").read_text())
    if not audit["native_teacher_diagnostic_passed"]:
        raise RuntimeError("v6 qualification missing")
    config = json.loads((old / "protocol.json").read_text())["environment"]
    rec = json.loads((old / p["prerequisite"]["oracle_v6_snapshot"]).read_text())
    a = NativeTask(config)
    b = TransferTask(config, p["variants"][0])
    try:
        if a.contract != json.loads((old / "specification.json").read_text()):
            raise RuntimeError("v6 compiled contract changed")
        a.restore(rec["snapshot"])
        snap = copy.deepcopy(rec["snapshot"])
        snap["contract_hash"] = b.contract_hash
        b.restore(snap)
        for action in rec["continuation_actions"]:
            aa, bb = a.step(action), b.step(action)
            if (
                aa[1:5] != bb[1:5]
                or a.snapshot()["state"] != b.snapshot()["state"]
                or not np.array_equal(a.last_obs, b.last_obs)
            ):
                raise RuntimeError("v6 actual continuation oracle failed")
        error = float(
            np.max(
                abs(np.asarray(b.snapshot()["state"]) - rec["final_snapshot"]["state"])
            )
        )
        if error != 0 or not np.array_equal(
            b.last_obs, rec["final_snapshot"]["last_raw_observation"]
        ):
            raise RuntimeError("v6 source-host reference replay failed")
        return {
            "passed": True,
            "max_reference_state_error": error,
            "observation_error": 0,
            "physics_steps": a.physics_total + b.physics_total,
            "v6_audit_sha256": file_hash(old.parent / "audit-linux.json"),
            "source_snapshot_sha256": file_hash(
                old / p["prerequisite"]["oracle_v6_snapshot"]
            ),
        }
    finally:
        a.close()
        b.close()


def benchmark_worker(inputs, p, seed):
    import os
    import time
    import traceback

    loaded = time.perf_counter()
    task = None
    try:
        model, normalizer = load_controller(Path(inputs))
        task = TransferTask(p["environment"], p["variants"][0])
        task.reset(seed)
        startup = time.perf_counter() - loaded
        started = time.perf_counter()
        resets = 1
        for i in range(p["benchmark"]["control_steps_per_world"]):
            if task.done:
                task.reset(seed + i + 1)
                resets += 1
            task.snapshot()
            action, _ = predict(model, normalizer, task.last_obs)
            task.step(action)
        normalizer.check()
        return {
            "status": "completed",
            "pid": os.getpid(),
            "seed": seed,
            "startup_s": startup,
            "loop_s": time.perf_counter() - started,
            "control_steps": p["benchmark"]["control_steps_per_world"],
            "physics_steps": task.physics_total,
            "reset_attempts": resets,
            "invalid": 0,
        }
    except Exception as error:
        return {
            "status": "invalid",
            "pid": os.getpid(),
            "seed": seed,
            "physics_steps": task.physics_total if task else 0,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "invalid": 1,
        }
    finally:
        if task:
            task.close()


def benchmark(inputs, p, out):
    import multiprocessing as mp
    import time
    from concurrent.futures import ProcessPoolExecutor

    rows = [
        {
            "workers": workers,
            "repeat": repeat,
            "all_status": "not_run",
            "samples": [],
            "physics_steps": 0,
        }
        for repeat in range(p["benchmark"]["repeats"])
        for workers in p["benchmark"]["workers"]
    ]

    def result():
        return {
            "cases": rows,
            "physics_steps": sum(r["physics_steps"] for r in rows),
            "scope": "Policy+normalization+CPU physics+sampled support+snapshot; optimizer/referee absent. Loop TPS excludes construction/disk output; parent TPS includes process construction/shutdown.",
        }

    write_json(out / "benchmark.json", result())
    for row in rows:
        workers, repeat = row["workers"], row["repeat"]
        started = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    benchmark_worker,
                    str(inputs),
                    p,
                    p["benchmark"]["seed_base"] + repeat * 100 + i,
                )
                for i in range(workers)
            ]
            samples = []
            for f in futures:
                try:
                    samples.append(f.result())
                except Exception as error:
                    samples.append(
                        {
                            "status": "invalid",
                            "physics_steps": 0,
                            "counter_is_lower_bound": True,
                            "error": str(error),
                        }
                    )
        wall = time.perf_counter() - started
        ok = all(s["status"] == "completed" for s in samples)
        row.update(
            samples=samples,
            all_status="completed" if ok else "invalid",
            parent_wall_s=wall,
            physics_steps=sum(s["physics_steps"] for s in samples),
        )
        if ok:
            count = sum(s["control_steps"] for s in samples)
            row.update(
                end_to_end_steps_per_s=count / wall,
                concurrent_loop_steps_per_s=count / max(s["loop_s"] for s in samples),
            )
        write_json(out / "benchmark.json", result())
        if not ok:
            break
    return result()


def variant_worker(inputs, p, variant, out, deadline, physics_allowance):
    import os
    import time
    import traceback

    out = Path(out)
    out.mkdir(exist_ok=False)
    parent = out.parent
    rows = planned_rows(p, variant["id"])
    write_json(out / "evaluations.json", rows)
    write_json(
        out / "process.json",
        {"pid": os.getpid(), "status": "running", "variant": variant["id"]},
    )
    task = None
    last_valid = None
    action = None
    active = None
    actions = []
    trace = []
    started = time.perf_counter()
    last_progress = -10
    previous_action = None

    def progress(force=False):
        nonlocal last_progress
        now = time.perf_counter()
        if force or now - last_progress >= 2:
            write_json(
                out / "progress.json",
                {
                    "pid": os.getpid(),
                    "wall_s": now - started,
                    "physics_steps": task.physics_total if task else 0,
                    "completed": sum(r["status"] == "completed" for r in rows),
                    "seed": active["seed"] if active else None,
                },
            )
            last_progress = now

    try:
        model, n = load_controller(Path(inputs))
        task = TransferTask(p["environment"], variant)
        write_json(out / "specification.json", task.contract)
        (out / "model.xml").write_text(task.xml)
        for row in rows:
            active = row
            actions = []
            trace = []
            last_valid = None
            action = None
            previous_action = None
            task.reset(row["seed"])
            initial = task.snapshot()
            paired = task.paired_fields()
            write_json(out / f"starts/{row['seed']}.json", initial)
            row.update(
                initial_snapshot_hash=digest(initial),
                paired_fields=paired,
                paired_fields_hash=digest(paired),
                normalizer_stats_hash=n.fingerprint,
            )
            replay = None
            clip_sum = 0.0
            variation_sum = 0.0
            total_reward = 0.0
            while not task.done:
                if (parent / "stop.json").exists():
                    raise TimeoutError("Parent/peer stop")
                if (
                    time.perf_counter() >= deadline
                    or task.physics_total + task.env.frame_skip > physics_allowance
                ):
                    raise TimeoutError("Registered wall/allocated physics cap")
                last_valid = task.snapshot()
                progress()
                action, clipped = predict(model, n, task.last_obs)
                _obs, reward, terminated, truncated, _, telemetry = task.step(action)
                clip_sum += clipped
                delta = (
                    float(np.mean(abs(action - previous_action)))
                    if previous_action is not None
                    else 0.0
                )
                variation_sum += delta
                previous_action = action.copy()
                total_reward += reward
                actions.append(np.asarray(action, dtype=np.float32))
                trace.append(
                    telemetry
                    | {
                        "normalizer_clipped_fraction": clipped,
                        "mean_absolute_action_change": delta,
                    }
                )
                if (
                    row["seed"] == p["replay"]["seed"]
                    and len(actions) == p["replay"]["snapshot_control_step"]
                ):
                    replay = task.snapshot()
            n.check()
            row.update(
                status="completed",
                reason="native_horizon" if truncated else "native_tracking_failure",
                horizon_reached=bool(
                    task.data.time + 1e-9 >= task.env.terminate_time * task.env.cycles
                ),
                control_steps=len(actions),
                physics_steps=len(actions) * task.env.frame_skip,
                elapsed_sim_s=float(task.data.time),
                sampled_failure=task.sampled_failure,
                native_terminated=terminated,
                native_truncated=truncated,
                total_reward=total_reward,
                max_tilt_rad=task.max_tilt,
                mean_clipped_fraction=clip_sum / len(actions),
                mean_absolute_action_change=variation_sum / max(1, len(actions) - 1),
                mean_native_qpos_error=float(
                    np.mean([t["native_qpos_mean_error"] for t in trace])
                ),
            )
            np.savez_compressed(
                out / f"actions-{row['seed']}.npz", actions=np.asarray(actions)
            )
            write_json(out / f"trace-{row['seed']}.json", trace)
            if row["seed"] == p["replay"]["seed"]:
                write_json(
                    out / "replay.json",
                    {
                        "snapshot": replay,
                        "final_snapshot": task.snapshot(),
                        "actions_file": f"actions-{row['seed']}.npz",
                        "continuation_start": p["replay"]["snapshot_control_step"],
                        "reason": None
                        if replay is not None
                        and len(actions) > p["replay"]["snapshot_control_step"]
                        else "first_scenario_ended_before_continuation",
                    },
                )
            write_json(out / "evaluations.json", rows)
            progress(True)
        return {
            "variant": variant["id"],
            "status": "completed",
            "physics_steps": task.physics_total,
        }
    except Exception as error:
        if active is not None and active["status"] != "completed":
            active.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
                partial_physics_steps=len(actions)
                * (task.env.frame_skip if task else 0),
            )
            np.savez_compressed(
                out / f"partial-actions-{active['seed']}.npz",
                actions=np.asarray(actions),
            )
            write_json(out / f"partial-trace-{active['seed']}.json", trace)
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "halted_after_fault_or_cap"
        write_json(
            out / "fault.json",
            {
                "error": str(error),
                "type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "last_valid": last_valid,
                "action": np.asarray(action).tolist() if action is not None else None,
            },
        )
        return {
            "variant": variant["id"],
            "status": "halted",
            "physics_steps": task.physics_total if task else 0,
        }
    finally:
        write_json(out / "evaluations.json", rows)
        progress(True)
        write_json(
            out / "process.json",
            {"pid": os.getpid(), "status": "finished", "variant": variant["id"]},
        )
        if task:
            task.close()


def run_transfer(protocol_path, archive, out):
    import fcntl
    import importlib.metadata
    import multiprocessing as mp
    import os
    import time
    import traceback
    from concurrent.futures import ProcessPoolExecutor

    from .provenance import source_hash
    from .upstream_motor import RUNTIME, extract_archive, input_hashes
    from .wave import check_resources

    p = validate_protocol(json.loads(Path(protocol_path).read_text()))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".native-transfer.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    deadline = started + p["caps"]["wall_s"]
    prov = manifest()
    rows = planned_rows(p)
    diagnostic_steps = 0
    results = []
    write_json(out / "evaluations.json", rows)
    write_json(out / "protocol.json", p)
    write_json(
        out / "process.json", {"pid": os.getpid(), "status": "running", "run": str(out)}
    )
    journal = (out / "events.jsonl").open("a")

    def event(phase, status, **extras):
        journal.write(
            json.dumps(
                {
                    "phase": phase,
                    "status": status,
                    "wall_s": time.perf_counter() - started,
                    **extras,
                }
            )
            + "\n"
        )
        journal.flush()

    def collect():
        collected = []
        physics = diagnostic_steps
        for v in p["variants"]:
            folder = out / v["id"]
            collected.extend(
                json.loads((folder / "evaluations.json").read_text())
                if (folder / "evaluations.json").exists()
                else planned_rows(p, v["id"])
            )
            if (folder / "progress.json").exists():
                physics += json.loads((folder / "progress.json").read_text())[
                    "physics_steps"
                ]
        return collected, physics

    def health():
        rr, physics = collect()
        resource = check_resources(out, p["caps"])
        raw = 0
        for f in out.rglob("*"):
            try:
                if f.is_file():
                    raw += f.stat().st_size
            except FileNotFoundError:
                # A worker atomically renamed its own temporary receipt.
                continue
        if any((out / v["id"] / "fault.json").exists() for v in p["variants"]):
            raise RuntimeError("Worker fault: parent halting peers")
        if source_hash() != prov["source_hash"]:
            raise RuntimeError("Executing source changed")
        if (
            time.perf_counter() >= deadline
            or physics >= p["caps"]["physics_steps"]
            or raw >= p["caps"]["raw_bytes"]
        ):
            raise TimeoutError("Registered parent cap")
        write_json(out / "evaluations.json", rr)
        write_json(
            out / "progress.json",
            {
                "wall_s": time.perf_counter() - started,
                "physics_steps": physics,
                "completed": sum(r["status"] == "completed" for r in rr),
                "raw_bytes": raw,
                "resources": resource,
            },
        )
        return rr, physics

    try:
        check_resources(out, p["caps"], True)
        versions = {k: importlib.metadata.version(k) for k in RUNTIME}
        if any(versions[k].split("+")[0] != v for k, v in RUNTIME.items()):
            raise RuntimeError("Runtime differs")
        if prov["upstream"] != p["upstream"]:
            raise RuntimeError("Upstream pins differ")
        inputs = input_hashes()
        prov.update(
            protocol_hash=digest(p),
            input_hashes=inputs,
            runtime=versions,
            command=[
                "scripts/run_native_transfer.py",
                "--protocol",
                str(protocol_path),
                "--archive",
                str(archive),
                "--out",
                str(out),
            ],
        )
        write_json(out / "provenance.json", prov)
        event("inputs", "running")
        extract_archive(Path(archive), out / "inputs")
        model, n = load_controller(out / "inputs")
        write_json(out / "normalizer.json", n.export())
        del model, n
        event("T0", "running")
        oracle = feature_oracle(p["environment"])
        diagnostic_steps += oracle["physics_steps"]
        actual = source_host_oracle(p)
        diagnostic_steps += actual["physics_steps"]
        write_json(
            out / "native-oracle.json",
            {"feature_oracle": oracle, "v6_actual_continuation": actual},
        )
        event("T0", "completed")
        health()
        event("H1", "running")
        bench = benchmark(out / "inputs", p, out)
        diagnostic_steps += bench["physics_steps"]
        if any(r["all_status"] != "completed" for r in bench["cases"]):
            raise RuntimeError("Benchmark fault: all attempted cases retained")
        write_json(out / "benchmark.json", bench)
        event("H1", "completed")
        health()
        event("T1", "running")
        with ProcessPoolExecutor(
            max_workers=4, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = []
            remaining = p["caps"]["physics_steps"] - diagnostic_steps
            for v in p["variants"]:
                allowance = int(
                    remaining * (1 if v["physics_dt_s"] == 0.002 else 2) / 6
                )
                futures.append(
                    pool.submit(
                        variant_worker,
                        str(out / "inputs"),
                        p,
                        v,
                        str(out / v["id"]),
                        deadline,
                        allowance,
                    )
                )
            try:
                while not all(f.done() for f in futures):
                    health()
                    if any(f.done() and f.exception() is not None for f in futures):
                        raise RuntimeError("Worker process crashed; halt peers")
                    time.sleep(1)
            except Exception as error:
                write_json(out / "stop.json", {"reason": str(error)})
                raise
            finally:
                for f in futures:
                    try:
                        results.append(f.result())
                    except Exception as error:
                        results.append({"status": "crashed", "error": str(error)})
        rows, physics = collect()
        event(
            "T1",
            "completed" if all(r["status"] == "completed" for r in rows) else "halted",
        )
    except Exception as error:
        write_json(out / "stop.json", {"reason": str(error)})
        write_json(
            out / "fault.json",
            {
                "error": str(error),
                "type": type(error).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        event("run", "halted", reason=str(error))
    finally:
        rows, physics = collect()
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = "halted_before_attempt"
        write_json(out / "evaluations.json", rows)
        result = summarize(rows) | {
            "wall_s": time.perf_counter() - started,
            "physics_steps": physics,
            "diagnostic_physics_steps": diagnostic_steps,
            "worker_results": results,
            "source_unchanged": source_hash() == prov["source_hash"],
            "inputs_unchanged": input_hashes() == prov.get("input_hashes"),
            "qualification_pending_replay_audit": True,
        }
        write_json(out / "result.json", result)
        write_json(
            out / "process.json",
            {"pid": os.getpid(), "status": "finished", "run": str(out)},
        )
        journal.close()
        lease.close()
    return result
