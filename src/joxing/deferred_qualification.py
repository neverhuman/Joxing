"""Opt-in native parity observer. Stdlib only until the watchdog is ready."""

import argparse
import contextlib
import hashlib
import json
import math
import random
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

from .physics_credit import PhysicsCredit, process_identity


def value_digest(value):
    import numpy as np

    def encode(item):
        if hasattr(item, "detach"):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject or not np.isfinite(item).all():
                raise ValueError("Nonfinite/object trace array")
            return [
                "array",
                item.dtype.str,
                list(item.shape),
                np.ascontiguousarray(item).tobytes().hex(),
            ]
        if isinstance(item, np.generic):
            return encode(item.item())
        if isinstance(item, dict):
            return [
                "dict",
                [
                    [encode(k), encode(v)]
                    for k, v in sorted(
                        item.items(), key=lambda kv: (type(kv[0]).__name__, str(kv[0]))
                    )
                ],
            ]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [encode(v) for v in item]]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Nonfinite trace scalar")
            return ["float", item.hex()]
        if item is None or type(item) in (bool, int, str):
            return [type(item).__name__, item]
        raise TypeError("Unsupported trace value " + type(item).__name__)

    return hashlib.sha256(
        json.dumps(encode(value), separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def semantic_info(info):
    # Only the explicitly registered Monitor wall-clock field is excluded.
    return {
        k: ({x: y for x, y in v.items() if x != "t"} if k == "episode" else v)
        for k, v in info.items()
    }


class Journal:
    def __init__(self, folder, name):
        self.path = Path(folder) / "traces" / (name + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.owner = process_identity()

    def write(self, phase, **fields):
        row = {
            "owner": self.owner,
            "monotonic_s": time.monotonic(),
            "semantic": {"phase": phase, **fields},
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def install_physics_hooks(stack, folder, ledger_path, binding, slot, name):
    from .balance import BalanceModel
    from .stance_env import StanceEnv

    ledger = PhysicsCredit(ledger_path, binding)
    journal = Journal(folder, name)
    prepare, step, reset = BalanceModel.prepare, StanceEnv.step, StanceEnv.reset

    def prepared(rig, *args, **kwargs):
        amount = round(rig.config.preparation_s / rig.config.dt_s)
        token = ledger.reserve(amount, slot, "preparation")
        try:
            result = prepare(rig, *args, **kwargs)
        except BaseException as error:
            ledger.unknown(token, type(error).__name__)
            raise
        ledger.commit(token, amount)
        return result

    def stepped(env, action):
        token = ledger.reserve(
            round(env.config.control_s / env.config.dt_s), slot, "step"
        )
        before = env.physics_total
        try:
            result = step(env, action)
        except BaseException as error:
            ledger.unknown(token, type(error).__name__)
            raise
        ledger.commit(token, env.physics_total - before)
        journal.write(
            "step",
            action=value_digest(action),
            result=value_digest((result[:4], semantic_info(result[4]))),
            state=value_digest(env.snapshot()),
        )
        return result

    def reset_env(env, *args, **kwargs):
        result = reset(env, *args, **kwargs)
        if env.model.nu != 700:
            raise ValueError("Changed native muscle count")
        journal.write(
            "reset",
            result=value_digest(result),
            scenario=env.scenario_id,
            rng=value_digest(env.np_random.bit_generator.state),
            muscles=int(env.model.nu),
            model_hash=env.rig.hash,
            action_hash=env.groups.hash,
        )
        return result

    stack.enter_context(patch.object(BalanceModel, "prepare", prepared))
    stack.enter_context(patch.object(StanceEnv, "step", stepped))
    stack.enter_context(patch.object(StanceEnv, "reset", reset_env))


def instrumented_worker(out, index, seed, scale, *, folder, ledger_path, binding, slot):
    import gymnasium as gym

    from . import wave

    stack = contextlib.ExitStack()
    install_physics_hooks(
        stack, folder, ledger_path, binding, slot, f"native-world-{index}"
    )
    original = wave.make_worker(out, index, seed, scale)
    journal = Journal(folder, f"worker-{index}")

    class Observed(gym.Wrapper):
        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            journal.write(
                "reset",
                result=value_digest(result),
                reset_rng=value_digest(self.env.env.reset_rng.bit_generator.state),
            )
            return result

        def step(self, action):
            result = self.env.step(action)
            journal.write(
                "step",
                action=value_digest(action),
                result=value_digest((result[:4], semantic_info(result[4]))),
            )
            return result

        def close(self):
            journal.write(
                "close",
                state=value_digest(self.unwrapped.snapshot()),
                reset_rng=value_digest(self.env.env.reset_rng.bit_generator.state),
            )
            self.env.close()

    wrapped = Observed(original)
    wrapped._diagnostic_hooks = stack
    return wrapped


def rng_digest():
    import numpy as np
    import torch

    return value_digest(
        (random.getstate(), np.random.get_state(), torch.get_rng_state())
    )


def model_checkpoint(model, replay=False):
    values = {
        "actor": model.actor.state_dict(),
        "critic": model.critic.state_dict(),
        "critic_target": model.critic_target.state_dict(),
        "actor_optimizer": model.actor.optimizer.state_dict(),
        "critic_optimizer": model.critic.optimizer.state_dict(),
        "entropy": model.log_ent_coef
        if model.ent_coef_optimizer is not None
        else model.ent_coef_tensor,
        "entropy_optimizer": model.ent_coef_optimizer.state_dict()
        if model.ent_coef_optimizer is not None
        else None,
    }
    row = {
        "rng": rng_digest(),
        "state": value_digest(values),
        "updates": int(model._n_updates),
    }
    if replay:
        buffer = model.replay_buffer
        end = buffer.buffer_size if buffer.full else buffer.pos
        row["replay"] = value_digest(
            {
                "pos": buffer.pos,
                "full": buffer.full,
                "n_envs": buffer.n_envs,
                **{
                    k: getattr(buffer, k)[:end]
                    for k in (
                        "observations",
                        "next_observations",
                        "actions",
                        "rewards",
                        "dones",
                        "timeouts",
                    )
                },
            }
        )
    return row


def wait_ready(path, timeout_s):
    until = time.monotonic() + timeout_s
    while not Path(path).exists():
        if time.monotonic() >= until:
            raise TimeoutError("No qualified watcher readiness")
        time.sleep(0.01)


def physics_upper_bound(config, transitions=64, workers=4, ids=1, slots=4):
    control = round(config["control_s"] / config["dt_s"])
    preparation = round(config["preparation_s"] / config["dt_s"])
    horizon = round(config["episode_s"] / config["dt_s"])
    # Constructor + first vector reset + possible reset after EVERY transition.
    preparations = workers + workers + transitions + 1 + ids
    return slots * (
        transitions * control + 2 * ids * horizon + preparations * preparation
    )


def run_slot(folder, slot, packet, ledger_path, binding, ready):
    wait_ready(ready, 15)
    # No Torch/NumPy/native import is allowed above the readiness barrier.
    from . import deferred_trial, wave
    from .provenance import manifest, write_json

    folder = Path(folder)
    journal = Journal(folder, "learner")
    original_eval = wave.evaluate
    sac = wave.timed_sac_class()

    class ObservedSAC(sac):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            journal.write("model_initial", **model_checkpoint(self))

        def train(self, *args, **kwargs):
            journal.write("update_before", **model_checkpoint(self))
            result = super().train(*args, **kwargs)
            journal.write("update_after", **model_checkpoint(self))
            return result

        def learn(self, *args, **kwargs):
            journal.write("learn_before", **model_checkpoint(self))
            result = super().learn(*args, **kwargs)
            journal.write("learn_after", **model_checkpoint(self, replay=True))
            return result

    def evaluation(policy, env, ids, name, out, actor_hash=None):
        journal.write(
            "evaluation_before", name=name, actor=actor_hash, rng=rng_digest()
        )
        result = original_eval(policy, env, ids, name, out, actor_hash)
        journal.write(
            "evaluation_after",
            name=name,
            rows=value_digest(result),
            state=value_digest(env.snapshot()),
            rng=rng_digest(),
        )
        return result

    worker = partial(
        instrumented_worker,
        folder=folder,
        ledger_path=ledger_path,
        binding=binding,
        slot=slot["id"],
    )
    with contextlib.ExitStack() as stack:
        install_physics_hooks(
            stack, folder, ledger_path, binding, slot["id"], "native-evaluation"
        )
        for module in (wave, deferred_trial):
            stack.enter_context(patch.object(module, "make_worker", worker))
            stack.enter_context(
                patch.object(module, "timed_sac_class", lambda: ObservedSAC)
            )
            stack.enter_context(patch.object(module, "evaluate", evaluation))
        result = wave.train_run(
            folder / "trial",
            packet["trial_config"],
            slot["arm"],
            packet["seed"],
            packet["evaluation_ids"],
            manifest(),
            lambda _: None,
            defer_initial_evaluation=True,
            allocation_lifecycle=slot["variant"],
        )
    semantic_keys = (
        "summary",
        "scheduled_training_complete",
        "stop_reason",
        "training_transitions",
        "evaluation_physics_substeps",
        "evaluation_preparation_substeps",
        "updates",
        "initial_actor_hash",
        "final_actor_hash",
        "final_diagnostic",
        "source_unchanged",
    )
    summary = {k: result[k] for k in semantic_keys}
    summary["counts"] = {
        k: result["counts"][k]
        for k in (
            "physics_substeps",
            "preparation_substeps",
            "raw_reward_sum",
            "scaled_reward_sum",
        )
    }
    summary["specification"] = json.loads(
        (folder / "trial/specification.json").read_text()
    )
    write_json(folder / "semantic-result.json", summary)


def compare_slots(folders):
    expected = {
        "learner",
        "native-evaluation",
        *[f"{prefix}-{i}" for prefix in ("worker", "native-world") for i in range(4)],
    }
    rows = []
    for arm in ("A", "B"):
        a, b = Path(folders["legacy" + arm]), Path(folders["candidate" + arm])
        left = json.loads((a / "semantic-result.json").read_text())
        right = json.loads((b / "semantic-result.json").read_text())
        if left != right:
            raise ValueError("Semantic trial mismatch: " + arm)
        if (
            left["training_transitions"] != 64
            or left["updates"] != 12
            or not left["scheduled_training_complete"]
            or not left["source_unchanged"]
        ):
            raise ValueError("Incomplete actual learner schedule: " + arm)
        if left["initial_actor_hash"] == left["final_actor_hash"]:
            raise ValueError("No actor change: " + arm)
        for folder in (a, b):
            if {p.stem for p in (folder / "traces").glob("*.jsonl")} != expected:
                raise ValueError("Missing/foreign trace file")
        for name in sorted(expected):
            traces = []
            for folder in (a, b):
                values = [
                    json.loads(line)["semantic"]
                    for line in (folder / "traces" / (name + ".jsonl"))
                    .read_text()
                    .splitlines()
                ]
                if not values:
                    raise ValueError("Empty trace: " + name)
                traces.append(values)
            if traces[0] != traces[1]:
                index = next(
                    (i for i, (x, y) in enumerate(zip(*traces)) if x != y),
                    min(map(len, traces)),
                )
                raise ValueError(
                    f"Exact trace mismatch: arm={arm} file={name} row={index}"
                )
            counts = {
                phase: sum(r["phase"] == phase for r in traces[0])
                for phase in {r["phase"] for r in traces[0]}
            }
            if name == "learner":
                if counts != {
                    "model_initial": 1,
                    "learn_before": 1,
                    "learn_after": 1,
                    "update_before": 12,
                    "update_after": 12,
                    "evaluation_before": 2,
                    "evaluation_after": 2,
                }:
                    raise ValueError("Incomplete learner/RNG history")
                if [
                    r["updates"] for r in traces[0] if r["phase"] == "update_after"
                ] != list(range(1, 13)):
                    raise ValueError("Wrong actual update sequence")
            if name.startswith("worker-") and (
                sum(row["phase"] == "step" for row in traces[0]) != 16
                or traces[0][-1]["phase"] != "close"
            ):
                raise ValueError("Incomplete worker history")
            if name.startswith("native-world-") and (
                counts.get("step") != 16 or counts.get("reset", 0) < 2
            ):
                raise ValueError("Incomplete native worker history")
            if name == "native-evaluation" and (
                counts.get("reset") != 3 or counts.get("step", 0) < 2
            ):
                raise ValueError("Incomplete native evaluation history")
            rows.append(
                {"arm": arm, "trace": name, "rows": len(traces[0]), "exact": True}
            )
    return {
        "schema": "joxing-deferred-native-comparison-v1",
        "qualified": True,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("folder", "slot", "packet", "ledger", "binding", "ready"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    packet = json.loads(Path(args.packet).read_text())
    slot = next(s for s in packet["slots"] if s["id"] == args.slot)
    run_slot(args.folder, slot, packet, args.ledger, args.binding, args.ready)
