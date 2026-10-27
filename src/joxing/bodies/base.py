"""Owned instantaneous measurements, before actor-specific observation assembly."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyMeasurements:
    actor: str
    time_s: float
    qpos: np.ndarray
    qvel: np.ndarray
    activation: np.ndarray
    body_names: tuple[str, ...]
    world_position_m: np.ndarray
    world_linear_velocity_mps: np.ndarray
    world_rotation_matrix: np.ndarray
    world_angular_velocity_radps: np.ndarray


def owned_readonly(array):
    result = np.asarray(array).copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite body measurement")
    result.flags.writeable = False
    return result
