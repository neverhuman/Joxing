"""Read-only dual-body recording. Rendering never advances evaluated physics.

This first format is diagnostic-only until a connected referee and motor gates
qualify bouts. A successful file write is not evidence of boxing competence.
"""

import hashlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from .provenance import digest, manifest, write_json

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
SCHEMA = "joxing-dual-recording-v1"


class BoutRecorder:
    def __init__(self, scene, policies, *, max_frames=1500, max_bytes=64 * 1024**2):
        if set(policies) != {"A", "B"} or any(not v for v in policies.values()):
            raise ValueError("Both corner controller identities are required")
        if any(len(scene.actuator_ids[a]) != 700 for a in ("A", "B")):
            raise ValueError("Recording requires both complete anatomies")
        if max_frames < 1 or max_bytes < 1:
            raise ValueError("Positive recording limits required")
        self.scene, self.max_frames, self.max_bytes = scene, max_frames, max_bytes
        self.states, self.rows = [], []
        self.header = manifest() | {
            "schema": SCHEMA,
            "classification": "DIAGNOSTIC - NOT A QUALIFIED BOXING BOUT",
            "scene_hash": scene.report["scene_hash"],
            "policies": dict(policies),
            "physics_dt_s": float(scene.model.opt.timestep),
            "state_specification": int(STATE),
            "dimensions": scene.report["dimensions"],
            "gates_pending": scene.report["gates_pending"],
            "score_status": "unavailable: connected match referee not qualified",
            "states_are_visual_samples": True,
            "full_application_replay_required_separately": True,
            "new_render_physics_steps": 0,
        }

    def capture(self, data, *, physics_steps, contact_peak_N=0.0):
        """Copy a completed shared state; no setters, forward or step on live data."""
        if len(self.states) >= self.max_frames:
            raise TimeoutError("Recording frame limit")
        if not isinstance(physics_steps, int) or physics_steps < 0:
            raise ValueError("Exact cumulative physics count required")
        if not np.isfinite(contact_peak_N) or contact_peak_N < 0:
            raise ValueError("Invalid contact telemetry")
        state = np.empty(mujoco.mj_stateSize(self.scene.model, STATE))
        mujoco.mj_getState(self.scene.model, data, state, STATE)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite recorded state")
        if self.rows and (
            data.time <= self.rows[-1]["time_s"]
            or physics_steps <= self.rows[-1]["physics_steps"]
        ):
            raise ValueError("Recording time/count must increase")
        if self.rows and not np.isclose(
            data.time - self.rows[-1]["time_s"],
            (physics_steps - self.rows[-1]["physics_steps"])
            * self.header["physics_dt_s"],
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError("Recorded clock/count disagree")
        if (len(self.states) + 1) * state.nbytes > self.max_bytes:
            raise TimeoutError("Recording state allocation limit")
        self.states.append(state)
        self.rows.append(
            {
                "time_s": float(data.time),
                "physics_steps": physics_steps,
                "contact_peak_N": float(contact_peak_N),
                "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            }
        )

    def save(self, folder):
        if not self.states:
            raise ValueError("Empty recordings are not evidence")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, states=np.stack(self.states))
        payload = buffer.getvalue()
        report = self.header | {
            "frames": self.rows,
            "tape_sha256": hashlib.sha256(payload).hexdigest(),
            "tape_bytes": len(payload),
        }
        encoded = json.dumps(report, allow_nan=False).encode()
        if len(payload) + len(encoded) + 4096 > self.max_bytes:
            raise TimeoutError("Serialized recording limit")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "states.npz").write_bytes(payload)
        write_json(folder / "recording.json", report)
        return report


def load_recording(scene, folder):
    folder = Path(folder)
    report = json.loads((folder / "recording.json").read_text())
    if (
        report["schema"] != SCHEMA
        or report["scene_hash"] != scene.report["scene_hash"]
        or report["dimensions"] != scene.report["dimensions"]
        or report["physics_dt_s"] != scene.model.opt.timestep
        or report["state_specification"] != int(STATE)
        or report["classification"] != "DIAGNOSTIC - NOT A QUALIFIED BOXING BOUT"
    ):
        raise ValueError("Foreign model/recording contract")
    payload = (folder / "states.npz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != report["tape_sha256"]:
        raise ValueError("Recording checksum mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        states = archive["states"].copy()
    if states.shape != (len(report["frames"]), mujoco.mj_stateSize(scene.model, STATE)):
        raise ValueError("Recording state dimensions")
    if len(states) == 0 or states.dtype != np.float64 or not np.isfinite(states).all():
        raise ValueError("Empty/nonfinite/foreign precision recording")
    for state, row in zip(states, report["frames"], strict=True):
        if hashlib.sha256(state.tobytes()).hexdigest() != row["state_sha256"]:
            raise ValueError("Recording frame checksum mismatch")
        if state[0] != row["time_s"]:
            raise ValueError("Recorded state/clock mismatch")
    times = np.array([r["time_s"] for r in report["frames"]])
    ticks = np.array([r["physics_steps"] for r in report["frames"]])
    if np.any(np.diff(ticks) <= 0) or not np.allclose(
        np.diff(times), np.diff(ticks) * report["physics_dt_s"], rtol=0, atol=1e-9
    ):
        raise ValueError("Recorded clock/count disagree")
    return report, states


def frame_schedule(report, fps=25):
    """Nearest preceding frame: never reveal a future state in the video."""
    times = np.array([r["time_s"] for r in report["frames"]])
    if fps <= 0 or not np.isfinite(fps) or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid video timing")
    targets = (
        times[0] + np.arange(int(np.floor((times[-1] - times[0]) * fps)) + 1) / fps
    )
    return np.searchsorted(times, targets, side="right") - 1


def recording_hash(report):
    return digest(report)
