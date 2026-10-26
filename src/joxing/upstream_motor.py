"""Frozen native-task qualification; no teacher training or boxing transfer."""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
import pickle
import shutil
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from .provenance import ROOT, digest, manifest, source_hash, write_json
from .stance_train import file_hash
from .support import SupportMonitor
from .wave import canonical_actor_hash

ARCHIVE_SHA = "2b2f0f1007d530db76463e7d3d5fb309b52c7b2a53649920ddfd9cf8a6880c57"
MODES = ("serialized_default", "scheduled_restored")
RUNTIME = {
    "mujoco": "3.8.0",
    "numpy": "2.4.2",
    "gymnasium": "1.2.3",
    "stable-baselines3": "2.7.1",
    "torch": "2.10.0",
}
MODEL_ROOT = ROOT / "third_party/ms-human-700"
MSGYM_ROOT = ROOT / "third_party/msgym"


def validate_protocol(p):
    if set(p) != {
        "schema",
        "description",
        "archive_sha256",
        "upstream",
        "seeds",
        "modes",
        "environment",
        "expected",
        "runtime",
        "caps",
        "criteria",
        "replay_seed",
        "replay_control_step",
        "experiments",
    }:
        raise ValueError("Unknown or missing protocol fields")
    if p["schema"] != "joxing-upstream-motor-0.0.6":
        raise ValueError("Unknown qualification protocol")
    if (
        p["seeds"] != list(range(71000, 71020))
        or p["modes"] != list(MODES)
        or p["archive_sha256"] != ARCHIVE_SHA
        or p["runtime"] != RUNTIME
    ):
        raise ValueError("Registered inputs cannot change")
    if p["caps"] != {
        "wall_s": 900,
        "physics_steps": 1000000,
        "raw_bytes": 1073741824,
        "minimum_disk_bytes": 10737418240,
        "minimum_memory_bytes": 4294967296,
        "simulation_workers": 1,
        "threads": 1,
    }:
        raise ValueError("Registered caps cannot change")
    if p["criteria"] != {
        "native_teacher_diagnostic_min_horizon_completions_restored": 16,
        "attempts_per_mode": 20,
        "invalid_allowed": 0,
        "same_host_replay_max_error": 1e-9,
        "reset_pairing_exact": True,
        "normalizer_immutable": True,
    }:
        raise ValueError("Registered criteria cannot change")
    if p["environment"] != {
        "gait_cycles": 3,
        "skip_frames": 10,
        "reset_noise_scale": 0.001,
        "random_init": True,
        "qpos_diff_th": 0.06,
        "kinematic_play": False,
        "reward_dict": {
            "w_qpos": 50,
            "w_xpos": 50,
            "w_pelvis": 100,
            "w_energy": 0.1,
            "w_healthy": 100,
        },
    }:
        raise ValueError("Registered native environment cannot change")
    if p["expected"] != {
        "nq": 85,
        "nv": 85,
        "nu": 700,
        "na": 700,
        "observation": 3601,
        "groups": 138,
        "scheduled_amplitude": 0.075,
        "physics_dt_s": 0.002,
        "control_dt_s": 0.02,
    }:
        raise ValueError("Registered interface cannot change")
    if p["replay_seed"] != 71000 or p["replay_control_step"] != 10:
        raise ValueError("Registered replay selection cannot change")
    return copy.deepcopy(p)


def pinned_import(name):
    for path in (
        MSGYM_ROOT,
        MSGYM_ROOT / "DynSyn-SAC",
        MSGYM_ROOT / "DynSyn-SAC/SB3-Scripts",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(MSGYM_ROOT.resolve()):
        raise RuntimeError(f"Unexpected upstream module: {name}")
    return module


def input_hashes():
    files = [
        p
        for p in MODEL_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".png"}
    ]
    files += list((MSGYM_ROOT / "msgym/envs").glob("*.py"))
    files += list((MSGYM_ROOT / "DynSyn-SAC/DynSyn").glob("*.py"))
    files += [
        MSGYM_ROOT / "msgym/motion_data/walking_gait.npz",
        MSGYM_ROOT / "LICENSE",
        MODEL_ROOT / "LICENSE",
    ]
    return {
        str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(files)) if p.exists()
    }


def extract_archive(archive, out):
    if file_hash(archive) != ARCHIVE_SHA:
        raise ValueError("Unrecognized release archive")
    out.mkdir(exist_ok=False)
    names = {
        "model.zip": "LocomotionFull/checkpoint/best_model.zip",
        "normalizer.pkl": "LocomotionFull/checkpoint/best_env.zip",
        "config.json": "LocomotionFull/locomotionFull.json",
    }
    with zipfile.ZipFile(archive) as z:
        for target, member in names.items():
            with z.open(member) as src, (out / target).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return json.loads((out / "config.json").read_text())


class FrozenNormalizer:
    def __init__(self, normalizer, observation_size):
        self.normalizer = normalizer
        self.size = observation_size
        normalizer.training = False
        normalizer.norm_reward = False
        if (
            not normalizer.norm_obs
            or normalizer.obs_rms.mean.shape != (self.size,)
            or not np.isfinite(normalizer.obs_rms.mean).all()
            or not np.isfinite(normalizer.obs_rms.var).all()
            or np.any(normalizer.obs_rms.var < 0)
            or normalizer.epsilon <= 0
        ):
            raise ValueError("Invalid saved observation normalizer")
        self.fingerprint = self.stats_hash()

    @classmethod
    def load(cls, path, observation_size):
        # The caller verifies the exact user-selected upstream release hash first.
        from stable_baselines3.common.vec_env import VecNormalize

        with Path(path).open("rb") as f:
            value = pickle.load(f)
        if not isinstance(value, VecNormalize):
            raise TypeError("Expected upstream VecNormalize")
        return cls(value, observation_size)

    def export(self):
        n = self.normalizer
        return {
            "mean": n.obs_rms.mean.tolist(),
            "var": n.obs_rms.var.tolist(),
            "count": float(n.obs_rms.count),
            "clip": float(n.clip_obs),
            "epsilon": float(n.epsilon),
            "norm_obs": n.norm_obs,
            "training": n.training,
            "norm_reward": n.norm_reward,
        }

    def stats_hash(self):
        return digest(self.export())

    def check(self):
        if self.stats_hash() != self.fingerprint:
            raise RuntimeError("Frozen normalizer changed")

    def apply(self, observation):
        a = np.asarray(observation, dtype=np.float32)
        if a.shape != (self.size,) or not np.isfinite(a).all():
            raise ValueError("Malformed raw observation")
        if self.normalizer.training or self.normalizer.norm_reward:
            raise RuntimeError("Evaluation normalizer is not frozen")
        value = self.normalizer.normalize_obs(a)
        if value.shape != a.shape or not np.isfinite(value).all():
            raise ValueError("Malformed normalized observation")
        return value


def load_dynsyn(path):
    from stable_baselines3 import SAC

    cls = pinned_import("DynSyn.SAC_DynSyn").SAC_DynSyn
    aliases = dict(SAC.policy_aliases)
    try:
        return cls.load(path, device="cpu")
    finally:
        # Upstream writes the inherited alias dictionary during construction.
        SAC.policy_aliases.clear()
        SAC.policy_aliases.update(aliases)


def set_decoder_mode(model, mode):
    if mode not in MODES:
        raise ValueError("Unknown decoder mode")
    amplitude = (
        None
        if mode == MODES[0]
        else model.get_dynsyn_weight_amp(
            model.dynsyn_k, model.dynsyn_a, model.num_timesteps
        )
    )
    # Both locations are required for an ordinary SAC save/load roundtrip.
    model.dynsyn_weight_amp = amplitude
    model.actor.dynsyn_layer.update_dynsyn_weight_amp(amplitude)
    return amplitude


def loader_oracle(folder):
    model = load_dynsyn(folder / "model.zip")
    default_amp = model.actor.dynsyn_layer.dynsyn_weight_amp
    actor_hash = canonical_actor_hash(model.actor)
    probes = np.stack(
        [
            np.zeros(3601),
            np.linspace(-0.1, 0.1, 3601),
            np.random.default_rng(71029).normal(size=3601),
        ]
    ).astype(np.float32)
    before, _ = model.predict(probes, deterministic=True)
    amplitude = set_decoder_mode(model, MODES[1])
    after, _ = model.predict(probes, deterministic=True)
    model.save(folder / "restored-model.zip")
    restored = load_dynsyn(folder / "restored-model.zip")
    again, _ = restored.predict(probes, deterministic=True)
    if (
        before.shape != (3, 700)
        or after.shape != before.shape
        or not np.isfinite(np.r_[before.ravel(), after.ravel(), again.ravel()]).all()
        or max(np.max(abs(before)), np.max(abs(after))) > 1
    ):
        raise RuntimeError("Invalid loader probe actions")
    error = float(np.max(abs(after - again)))
    if (
        amplitude != 0.075
        or restored.actor.dynsyn_layer.dynsyn_weight_amp != amplitude
        or canonical_actor_hash(restored.actor) != actor_hash
        or error != 0
    ):
        raise RuntimeError("Decoder roundtrip does not preserve behavior")
    groups = model.actor.dynsyn_layer.muscle_groups
    if (
        sorted(i for group in groups for i in group) != list(range(700))
        or len(groups) != 138
    ):
        raise ValueError("Wrong muscle ownership")
    np.savez_compressed(
        folder / "loader-probes.npz",
        normalized_observations=probes,
        default_actions=before,
        restored_actions=after,
        roundtrip_actions=again,
    )
    return restored, {
        "default_layer_amplitude": default_amp,
        "restored_amplitude": amplitude,
        "default_load_omits_schedule": default_amp is None,
        "probe_mean_absolute_action_difference": float(np.mean(abs(after - before))),
        "probe_max_absolute_action_difference": float(np.max(abs(after - before))),
        "roundtrip_max_action_error": error,
        "canonical_actor_hash": actor_hash,
        "groups": groups,
        "probe_shape": list(probes.shape),
        "probe_file_sha256": file_hash(folder / "loader-probes.npz"),
        "model_sha256": file_hash(folder / "model.zip"),
        "restored_model_sha256": file_hash(folder / "restored-model.zip"),
    }


class NativeTask:
    def __init__(self, config):
        module = pinned_import("msgym.envs.locomotionFull_v1")
        # Only file lookup is adapted; no upstream files or model options change.
        with patch.object(
            module, "get_ms_human_model_path", lambda name: str(MODEL_ROOT / name)
        ):
            self.env = module.LocomotionFullEnvV1(**config)
        self.physics_total = 0
        self.last_obs = None
        self.model = self.env.model
        m = self.model
        if (m.nq, m.nv, m.nu, m.na) != (85, 85, 700, 700):
            raise ValueError("Unknown native anatomy")
        if m.opt.timestep != 0.002 or self.env.dt != 0.02:
            raise ValueError("Unknown native cadence")
        self.floor = m.geom("floor").id
        self.pelvis = m.body("pelvis").id
        self.weight = float(sum(m.body_mass) * 9.81)
        self.roles = {
            i: "right"
            if m.body(m.geom_bodyid[i]).name in ("calcn_r", "toes_r")
            else "left"
            if m.body(m.geom_bodyid[i]).name in ("calcn_l", "toes_l")
            else "nonfoot"
            for i in range(m.ngeom)
        }
        self.contract = self.specification()
        self.contract_hash = digest(self.contract)

    @property
    def data(self):
        return self.env.data

    def specification(self):
        _, blocks = self.env._get_obs()
        offset = 0
        slices = {}
        for name, values in blocks.items():
            slices[name] = [offset, offset + values.size]
            offset += values.size
        if offset != 3601:
            raise ValueError("Unknown native observation contract")
        return {
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "na": self.model.na,
            "dt": self.env.dt,
            "physics_dt": float(self.model.opt.timestep),
            "integrator": int(self.model.opt.integrator),
            "muscle_names": [self.model.actuator(i).name for i in range(self.model.nu)],
            "body_mass": self.model.body_mass.tolist(),
            "body_inertia": self.model.body_inertia.tolist(),
            "actuator_parameters_hash": digest(
                {
                    name: getattr(self.model, name).tolist()
                    for name in (
                        "actuator_dynprm",
                        "actuator_gainprm",
                        "actuator_biasprm",
                        "actuator_ctrlrange",
                    )
                }
            ),
            "observation_order": list(blocks),
            "observation_slices": slices,
            "action_transform": "sigmoid(5*(normalized700-action-0.5))",
            "model_path_adapter": "canonical pinned model lookup only",
            "support_telemetry": "sampled after each native control step; not per-substep sporting evidence",
        }

    def reset(self, seed):
        self.last_obs, _ = self.env.reset(seed=seed)
        self.initial_height = float(self.data.xpos[self.pelvis, 2])
        self.monitor = SupportMonitor()
        self.sampled_failure = None
        self.max_tilt = 0.0
        self.done = False
        self.check_assistance()
        return self.last_obs.copy()

    def check_assistance(self):
        if (
            self.env.kinematic_play
            or not np.array_equal(self.model.opt.gravity, [0, 0, -9.81])
            or np.any(self.model.body_gravcomp)
            or np.any(self.data.qfrc_applied)
            or np.any(self.data.xfrc_applied)
        ):
            raise RuntimeError("Unexpected native task assistance")

    def telemetry(self):
        d, m = self.data, self.model
        loads = {"right": 0.0, "left": 0.0, "nonfoot": 0.0}
        for slot, c in enumerate(d.contact):
            if c.efc_address < 0 or self.floor not in (c.geom1, c.geom2):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, slot, wrench)
            force = c.frame.reshape(3, 3).T @ wrench[:3]
            other = c.geom2 if c.geom1 == self.floor else c.geom1
            sign = 1 if c.geom1 == self.floor else -1
            loads[self.roles[int(other)]] += max(0.0, float(sign * force[2]))
        up = d.xmat[self.pelvis].reshape(3, 3)[:, 1]
        tilt = float(np.arccos(np.clip(up[2], -1, 1)))
        drop = self.initial_height - float(d.xpos[self.pelvis, 2])
        self.max_tilt = max(self.max_tilt, tilt)
        reason = self.monitor.update(loads, self.weight, tilt, drop, self.env.dt)
        self.sampled_failure = self.sampled_failure or reason
        return {
            "time_s": float(d.time),
            "tilt_rad": tilt,
            "drop_m": drop,
            "floor_loads_N": loads,
            "sampled_failure": self.sampled_failure,
            "native_qpos_mean_error": float(
                np.mean(abs(d.qpos[3:] - self.env.qpos_ref[3:]))
            ),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Native reset required")
        self.check_assistance()
        # The released policy/wrapper uses float32. Preserve that arithmetic
        # for both online inference and JSON/NPZ command continuation.
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (700,) or not np.isfinite(a).all() or np.max(abs(a)) > 1:
            raise ValueError("Malformed native muscle action")
        before_time = self.data.time
        warnings = np.array([w.number for w in self.data.warning])
        obs, reward, terminated, truncated, info = self.env.step(
            1 / (1 + np.exp(-5 * (a - 0.5)))
        )
        self.physics_total += self.env.frame_skip
        if (
            abs(self.data.time - before_time - self.env.dt) > 1e-9
            or np.any(np.array([w.number for w in self.data.warning]) > warnings)
            or not np.isfinite(
                np.concatenate(
                    [self.data.qpos, self.data.qvel, self.data.act, obs, [reward]]
                )
            ).all()
        ):
            raise RuntimeError("Invalid native physics transition")
        self.last_obs = obs.copy()
        self.done = bool(terminated or truncated)
        return (
            obs.copy(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
            self.telemetry(),
        )

    def snapshot(self):
        state = np.empty(
            mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        )
        mujoco.mj_getState(
            self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        app = {
            k: np.asarray(getattr(self.env, k)).tolist()
            for k in ("qpos_ref", "xpos_ref", "qpos_ref_future")
        }
        app.update(
            {k: float(getattr(self.env, k)) for k in ("init_time", "terminate_time")}
        )
        app["current_traj_index"] = int(self.env.current_traj_index)
        return {
            "contract_hash": self.contract_hash,
            "state": state.tolist(),
            "last_raw_observation": self.last_obs.tolist(),
            "application": app,
            "rng": copy.deepcopy(self.env.np_random.bit_generator.state),
            "initial_height": self.initial_height,
            "monitor": self.monitor.snapshot(),
            "sampled_failure": self.sampled_failure,
            "max_tilt": self.max_tilt,
            "done": self.done,
        }

    def restore(self, s):
        if s["contract_hash"] != self.contract_hash:
            raise ValueError("Snapshot contract mismatch")
        state = np.asarray(s["state"], dtype=float)
        expected = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        obs = np.asarray(s["last_raw_observation"], dtype=np.float32)
        if (
            state.shape != (expected,)
            or obs.shape != (3601,)
            or not np.isfinite(np.r_[state, obs]).all()
        ):
            raise ValueError("Malformed native snapshot")
        app = s["application"]
        for key, shape in (
            ("qpos_ref", (85,)),
            ("xpos_ref", (self.model.nbody, 3)),
            ("qpos_ref_future", (5, 85)),
        ):
            value = np.asarray(app[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Malformed native reference state")
        if (
            not np.isfinite(
                [
                    app["init_time"],
                    app["terminate_time"],
                    s["initial_height"],
                    s["max_tilt"],
                ]
            ).all()
            or app["init_time"] < 0
            or app["terminate_time"] <= 0
            or app["current_traj_index"]
            not in range(self.env.trajectory.num_trajectories)
            or not isinstance(s["done"], bool)
        ):
            raise ValueError("Malformed native application state")
        if not np.array_equal(obs[:85], state[1:86].astype(np.float32)):
            raise ValueError(
                "Cached observation is inconsistent with integration state"
            )
        mujoco.mj_setState(
            self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(self.model, self.data)
        # Forward refreshes caches; reinstate the recorded solver integration inputs.
        mujoco.mj_setState(
            self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        for k, value in s["application"].items():
            setattr(
                self.env,
                k,
                np.asarray(value) if k.endswith(("ref", "future")) else value,
            )
        self.env.np_random.bit_generator.state = copy.deepcopy(s["rng"])
        self.last_obs = obs.copy()
        self.initial_height = s["initial_height"]
        self.monitor = SupportMonitor(**s["monitor"])
        self.sampled_failure, self.max_tilt, self.done = (
            s["sampled_failure"],
            s["max_tilt"],
            s["done"],
        )
        return obs.copy()

    def close(self):
        self.env.close()


def summarize(rows):
    expected = {f"{mode}:{seed}" for seed in range(71000, 71020) for mode in MODES}
    if len(rows) != 40 or {r["attempt_id"] for r in rows} != expected:
        raise ValueError("Missing or duplicate planned attempt")
    modes = {}
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        complete = [r for r in subset if r["status"] == "completed"]
        modes[mode] = {
            "planned": len(subset),
            "completed": len(complete),
            "invalid": sum(r["status"] == "invalid" for r in subset),
            "not_run": sum(r["status"] == "not_run" for r in subset),
            "incomplete": sum(r["status"] == "incomplete" for r in subset),
            "horizon_completions": sum(r["horizon_reached"] for r in complete),
            "mean_elapsed_sim_s": float(np.mean([r["elapsed_sim_s"] for r in complete]))
            if complete
            else None,
            "sampled_support_failure_episodes": sum(
                r["sampled_failure"] is not None for r in complete
            ),
        }
    return {
        "modes": modes,
        "all_attempts_completed": all(r["status"] == "completed" for r in rows),
        "invalid": sum(r["status"] == "invalid" for r in rows),
        "native_task_completion_threshold_met": len(rows) == 40
        and all(r["status"] == "completed" for r in rows)
        and modes[MODES[1]]["horizon_completions"] >= 16,
        "scope": "Frozen native locomotion diagnostic; not Joxing stance, boxing, self-play or a training-improvement proof.",
    }


def run_qualification(protocol_path, archive, out):
    import fcntl
    import os

    import torch

    from .wave import check_resources

    p = validate_protocol(json.loads(Path(protocol_path).read_text()))
    torch.set_num_threads(1)
    versions = {k: importlib.metadata.version(k) for k in RUNTIME}
    if any(versions[k].split("+")[0] != v for k, v in RUNTIME.items()):
        raise RuntimeError(f"Checkpoint runtime mismatch: {versions}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lease = (out.parent / ".upstream-motor.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    inputs = input_hashes()
    prov = manifest() | {
        "protocol_hash": digest(p),
        "input_hashes": inputs,
        "runtime": versions,
        "pid": os.getpid(),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "resources": check_resources(out, p["caps"], True),
    }
    if prov["upstream"] != p["upstream"]:
        raise RuntimeError("Pinned upstream revisions differ")
    write_json(out / "protocol.json", p)
    write_json(out / "provenance.json", prov)
    write_json(
        out / "process.json", {"pid": os.getpid(), "status": "running", "run": str(out)}
    )
    rows = [
        {
            "attempt_id": f"{mode}:{seed}",
            "seed": seed,
            "mode": mode,
            "status": "not_run",
            "reason": "pending_dependencies",
        }
        for seed in p["seeds"]
        for mode in MODES
    ]
    write_json(out / "evaluations.json", rows)
    task, last_valid, current_action = None, None, None
    active_index = None
    last_progress = -10.0

    def event(phase, status, **data):
        row = {
            "phase": phase,
            "status": status,
            "wall_s": time.perf_counter() - started,
            **data,
        }
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")

    def budget(force=False, **progress):
        nonlocal last_progress
        elapsed = time.perf_counter() - started
        physics = task.physics_total if task is not None else 0
        if elapsed >= p["caps"]["wall_s"] or physics + 10 > p["caps"]["physics_steps"]:
            raise TimeoutError("Registered wall/physics cap")
        if force or elapsed - last_progress >= 5:
            if source_hash() != prov["source_hash"]:
                raise RuntimeError("Executing source changed")
            resource = check_resources(out, p["caps"])
            raw = sum(x.stat().st_size for x in out.rglob("*") if x.is_file())
            if raw >= p["caps"]["raw_bytes"]:
                raise TimeoutError("Registered artifact cap")
            write_json(
                out / "progress.json",
                {
                    "wall_s": elapsed,
                    "physics_steps": physics,
                    "raw_bytes": raw,
                    "resources": resource,
                    "completed_attempts": sum(r["status"] == "completed" for r in rows),
                    **progress,
                },
            )
            last_progress = elapsed

    try:
        event("L0", "running")
        config = extract_archive(archive, out / "inputs")
        expected_config = dict(p["environment"])
        expected_config.pop("kinematic_play")
        if config["single_env_kwargs"] != expected_config:
            raise ValueError("Saved task configuration differs")
        model, oracle = loader_oracle(out / "inputs")
        normalizer = FrozenNormalizer.load(out / "inputs/normalizer.pkl", 3601)
        oracle["normalizer_sha256"] = file_hash(out / "inputs/normalizer.pkl")
        oracle["normalizer_stats_hash"] = normalizer.fingerprint
        write_json(out / "normalizer.json", normalizer.export())
        if model.observation_space.shape != (3601,) or model.action_space.shape != (
            700,
        ):
            raise ValueError("Checkpoint interface mismatch")
        task = NativeTask(p["environment"])
        write_json(out / "specification.json", task.contract)
        write_json(out / "loader-oracle.json", oracle)
        event("L0", "completed", result="loader-oracle.json")
        event("N0", "running")
        for index, row in enumerate(rows):
            active_index = index
            last_valid, current_action = None, None
            mode, seed = row["mode"], row["seed"]
            amplitude = set_decoder_mode(model, mode)
            obs = task.reset(seed)
            initial = task.snapshot()
            write_json(out / f"starts/{mode}-{seed}.json", initial)
            initial_hash = digest(initial)
            paired = next((r for r in rows[:index] if r["seed"] == seed), None)
            if paired is not None and paired["initial_snapshot_hash"] != initial_hash:
                raise RuntimeError("Unpaired native starts")
            event(row["attempt_id"], "running")
            trace, actions, snapshot, continuation = [], [], None, []
            total_reward, terminated, truncated = 0.0, False, False
            while not task.done:
                last_valid = task.snapshot()
                budget(mode=mode, seed=seed, elapsed_sim_s=float(task.data.time))
                current_action, hidden = model.predict(
                    normalizer.apply(obs), deterministic=True
                )
                if hidden is not None:
                    raise ValueError("Unexpected policy hidden state")
                obs, reward, terminated, truncated, _, telemetry = task.step(
                    current_action
                )
                total_reward += reward
                actions.append(np.asarray(current_action, dtype=np.float32))
                trace.append(telemetry)
                if seed == p["replay_seed"]:
                    if snapshot is not None:
                        continuation.append(current_action.tolist())
                    if len(actions) == p["replay_control_step"]:
                        snapshot = task.snapshot()
            normalizer.check()
            row.update(
                status="completed",
                reason="native_horizon"
                if task.data.time + 1e-9 >= task.env.terminate_time * task.env.cycles
                else "native_tracking_failure",
                initial_snapshot_hash=initial_hash,
                amplitude=amplitude,
                elapsed_sim_s=float(task.data.time),
                control_steps=len(actions),
                physics_steps=len(actions) * task.env.frame_skip,
                horizon_reached=bool(
                    task.data.time + 1e-9 >= task.env.terminate_time * task.env.cycles
                ),
                native_healthy_after_reference_update=bool(task.env.is_healthy),
                native_terminated=terminated,
                native_truncated=truncated,
                sampled_failure=task.sampled_failure,
                total_reward=total_reward,
                max_tilt_rad=task.max_tilt,
                normalizer_stats_hash=normalizer.fingerprint,
                initial_phase_s=initial["application"]["init_time"],
            )
            np.savez_compressed(
                out / f"actions-{mode}-{seed}.npz", actions=np.asarray(actions)
            )
            write_json(out / f"trace-{mode}-{seed}.json", trace)
            if seed == p["replay_seed"]:
                write_json(
                    out / f"replay-{mode}.json",
                    {
                        "snapshot": snapshot,
                        "continuation_actions": continuation,
                        "final_snapshot": task.snapshot(),
                        "result": dict(row),
                        "normalizer_stats_hash": normalizer.fingerprint,
                        "reason": None
                        if snapshot is not None and continuation
                        else "first_scenario_ended_before_replay_continuation",
                    },
                )
            write_json(out / "evaluations.json", rows)
            event(row["attempt_id"], "completed", reason=row["reason"])
            budget(force=True, mode=mode, seed=seed)
        event("N0", "completed")
    except Exception as error:
        current = (
            rows[active_index]
            if active_index is not None and rows[active_index]["status"] != "completed"
            else None
        )
        if current is not None:
            current.update(
                status="incomplete" if isinstance(error, TimeoutError) else "invalid",
                reason=str(error),
            )
        for row in rows:
            if row["status"] == "not_run":
                row["reason"] = (
                    "halted_after_cap"
                    if isinstance(error, TimeoutError)
                    else "halted_after_fault"
                )
        write_json(
            out / "fault.json",
            {
                "error": str(error),
                "type": type(error).__name__,
                "last_valid": last_valid,
                "action": np.asarray(current_action).tolist()
                if current_action is not None
                else None,
            },
        )
        write_json(out / "evaluations.json", rows)
        event("run", "halted", reason=str(error))
        raise
    finally:
        result = summarize(rows) | {
            "physics_steps": task.physics_total if task else 0,
            "wall_s": time.perf_counter() - started,
            "source_unchanged": source_hash() == prov["source_hash"],
            "inputs_unchanged": input_hashes() == inputs,
            "qualification_pending_replay_audit": True,
        }
        write_json(out / "result.json", result)
        write_json(
            out / "process.json",
            {"pid": os.getpid(), "status": "finished", "run": str(out)},
        )
        if task is not None:
            task.close()
        lease.close()
    if not result["source_unchanged"] or not result["inputs_unchanged"]:
        raise RuntimeError("Executing source or assets changed")
    return result
