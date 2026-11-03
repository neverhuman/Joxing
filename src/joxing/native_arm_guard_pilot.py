"""H35 PPO/checkpoint adapter. Does not launch or authorize a native study."""

import copy
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .native_arm_guard import SCHEMA
from .native_pair_motor_pilot import PilotAuditedPPO
from .native_pair_motor_train import actor_hash
from .native_tape_storage import sha_file


class GuardTransitions(BaseCallback):
    def __init__(self, vec, writer, guard):
        super().__init__()
        self.vec, self.writer, self.guard = vec, writer, guard

    def _on_training_start(self):
        self.model.pilot.save(self.model)

    def _on_step(self):
        self.guard(0)
        if self.vec.world.contract_hash != self.writer.contract_hash:
            raise ValueError("Guard learner/writer world drift")
        boundary = self.model.pilot.checkpoints[-1]
        transition = self.vec.last_transition
        transition["info"]["behavior"] = {
            "boundary": boundary["number"],
            "policy_sha256": boundary["policy"]["sha256"],
            "actor_hash": boundary["actor_hash"],
        }
        self.writer.append(
            transition,
            raw_actions=self.locals["actions"].copy(),
            values=self.locals["values"].detach().cpu().numpy().copy(),
            log_probs=self.locals["log_probs"].detach().cpu().numpy().copy(),
        )
        return True


def restore_guard_checkpoint(folder, row, vec):
    """Authenticate a locally owned checkpoint, then restore its logical epoch.

    The ordinary world.restore rejects old episodes. Only this explicit trusted
    continuation path may migrate epochs. No inference or physics during restore;
    existing physical charges are not rewound. RNG restoration is LAST.
    """
    folder = Path(folder)
    n = row["number"]
    if type(n) is not int or n < 0 or vec.pending is not None or vec.waiting:
        raise ValueError("Guard checkpoint idle Vec/boundary")
    for suffix, key, maximum in (
        ("zip", "policy", 13 * 1024**2),
        ("pt", "continuation", 1024**2),
    ):
        path = folder / f"boundary-{n:03d}.{suffix}"
        if (
            not 0 < path.stat().st_size <= maximum
            or path.stat().st_size != row[key]["bytes"]
            or sha_file(path) != row[key]["sha256"]
        ):
            raise ValueError("Guard checkpoint external identity")
    # Pickles are only loaded after authentication against the owned receipt.
    value = torch.load(
        folder / f"boundary-{n:03d}.pt", map_location="cpu", weights_only=False
    )
    state = value["world"]
    if (
        value["schema"] != "joxing-native-pilot-continuation-v1"
        or value["contract_hash"] != vec.world.contract_hash
        or state["schema"] != SCHEMA
        or state["sha256"] != row["world_snapshot_sha256"]
        or value["world_episode_serial"] != state["episode_serial"]
        or value["world_completed_episodes"] != state["completed_episodes"]
        or value["num_timesteps"] != row["actor_samples"]
        or state["base"]["core"]["remaining"]
    ):
        raise ValueError("Guard checkpoint contract/counter/clock")
    expected_fields = {
        "next_seed",
        "episode_returns",
        "episode_lengths",
        "reset_infos",
        "_seeds",
        "_options",
    }
    if set(value["vec"]) != expected_fields:
        raise ValueError("Guard checkpoint Vec fields")
    if (
        value["vec"]["next_seed"] != state["episode_seed"] + 1
        or value["vec"]["episode_lengths"] * 20 != state["base"]["core"]["tick"]
        or value["vec"]["episode_returns"].shape != (2,)
        or not np.isfinite(value["vec"]["episode_returns"]).all()
    ):
        raise ValueError("Guard checkpoint coupled episode state")
    cached = np.stack(
        [np.asarray(state["base"]["cached"][p], np.float32) for p in "AB"]
    )
    for i, p in enumerate("AB"):
        cached[i, -9:] = state["guard_goals"][p]
    if (
        value["last_obs"].dtype != np.float32
        or cached.tobytes() != value["last_obs"].tobytes()
        or value["last_episode_starts"].shape != (2,)
        or value["last_episode_starts"].dtype != np.bool_
        or value["last_episode_starts"][0] != value["last_episode_starts"][1]
    ):
        raise ValueError("Guard checkpoint causal observation/masks")
    model = PilotAuditedPPO.load(
        folder / f"boundary-{n:03d}.zip", env=vec, device="cpu", force_reset=False
    )
    if (
        model.num_timesteps != row["actor_samples"]
        or actor_hash(model.policy) != row["actor_hash"]
        or model.action_space.shape != (46,)
        or model.observation_space.shape != (7901,)
    ):
        raise ValueError("Guard checkpoint learner/action identity")
    # Validate saved RNG states without changing global streams before commit.
    torch.Generator().set_state(value["torch"])
    np.random.RandomState().set_state(value["numpy"])
    random.Random().setstate(value["python"])
    vec.world.restore(state, restore_epoch=True)
    for name, item in value["vec"].items():
        setattr(vec, name, copy.deepcopy(item))
    vec.last_transition = None
    model._last_obs = value["last_obs"].copy()
    model._last_episode_starts = value["last_episode_starts"].copy()
    torch.set_rng_state(value["torch"])
    np.random.set_state(value["numpy"])
    random.setstate(value["python"])
    return model
