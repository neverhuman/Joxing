"""Constructor-only instrumentation. Never a training or motor qualification."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch


class ProfileExit(BaseException):
    """Leave train_run through finally without its checkpoint-on-fault path."""


class GuardViolation(BaseException):
    """A prohibited operation was attempted; never convert this into readiness."""


def identity(pid, proc_root=Path("/proc")):
    stat = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "parent_pid": int(stat[1]),
        "start_ticks": int(stat[19]),
        "state": stat[0],
    }


def memory_sample(bound, proc_root=Path("/proc")):
    """Only read bound PID/start identities; missing/reused identities have no sample."""
    try:
        current = identity(bound["pid"], proc_root)
        if current["start_ticks"] != bound["start_ticks"] or current["state"] == "Z":
            return None
        folder = proc_root / str(bound["pid"])
        status = (folder / "status").read_text()
        rss = (
            int(
                next(
                    s.split()[1] for s in status.splitlines() if s.startswith("VmRSS:")
                )
            )
            * 1024
        )
        rollup = {}
        for line in (folder / "smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("Pss", "Private_Clean", "Private_Dirty", "Private_Hugetlb"):
                rollup[key] = int(value.split()[0]) * 1024
        torch_mapped = "libtorch" in (folder / "maps").read_text()
        if identity(bound["pid"], proc_root)["start_ticks"] != bound["start_ticks"]:
            return None
        return current | {
            "rss_bytes": rss,
            "pss_bytes": rollup.get("Pss"),
            "private_bytes": sum(
                v for k, v in rollup.items() if k.startswith("Private_")
            ),
            "libtorch_mapped": torch_mapped,
        }
    except (OSError, StopIteration):
        return None


def emit(folder, trial, role, phase, **values):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    own = identity(os.getpid())
    row = dict(
        monotonic_s=time.monotonic(),
        trial=trial,
        role=role,
        phase=phase,
        identity=own,
        self_memory=memory_sample(own),
        **values,
    )
    # A single process owns each file. Interrupted final lines remain raw evidence.
    with (folder / f"events-{own['pid']}-{own['start_ticks']}.jsonl").open(
        "a"
    ) as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
    return row


def read_events(folder):
    rows = []
    for path in sorted(Path(folder).glob("events-*.jsonl")):
        raw = path.read_text()
        for line in raw.splitlines(keepends=True):
            if line.endswith("\n"):
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["monotonic_s"])


def validate_events(rows, owned):
    """Fail closed on foreign/unbound roles and impossible constructor ordering."""
    owners = {(r["pid"], r["start_ticks"]) for r in owned}
    phases = {}
    roles = {}
    for row in rows:
        ident = row["identity"]
        key = (ident["pid"], ident["start_ticks"])
        if key not in owners or row["trial"] not in ("A", "B"):
            raise ValueError("Unbound profile event")
        role = row["role"]
        if role != "learner" and role not in {f"world-{i}" for i in range(4)}:
            raise ValueError("Unknown profile role")
        if key in roles and roles[key] != (row["trial"], role):
            raise ValueError("Process role changed")
        roles[key] = (row["trial"], role)
        seq = phases.setdefault(key, [])
        phase = row["phase"]
        for after, before in [
            ("evaluation_env_after", "evaluation_env_before"),
            ("vector_after", "vector_before"),
            ("sac_after", "sac_before"),
            ("learn_ready", "sac_after"),
            ("world_after", "world_before"),
        ]:
            if phase == after and before not in seq:
                raise ValueError("Impossible profile phase order")
        if phase == "guard_violation":
            raise ValueError("Prohibited inference/training attempted")
        seq.append(phase)
    return {
        f"{pid}:{start}": {"trial": t, "role": r}
        for (pid, start), (t, r) in roles.items()
    }


def sample_owned(rows, *, sample=memory_sample, is_live=None):
    """A still-live unreadable process invalidates the aggregate, never counts as zero."""
    if is_live is None:

        def is_live(row):
            try:
                actual = identity(row["pid"])
                return (
                    actual["start_ticks"] == row["start_ticks"]
                    and actual["state"] != "Z"
                )
            except OSError:
                return False

    samples = []
    for row in rows:
        current = sample(row)
        if current is not None:
            samples.append(current)
        elif is_live(row):
            raise ValueError("Unreadable live owned process; aggregate incomplete")
    return samples


def profile_denominator(rows):
    """Completed constructor counters are lower bounds when another constructor stops."""
    complete = [
        e for e in rows if e["phase"] in ("world_after", "evaluation_env_after")
    ]
    expected = {
        (arm, role)
        for arm in ("A", "B")
        for role in ("learner", "world-0", "world-1", "world-2", "world-3")
    }
    actual = [(e["trial"], e["role"]) for e in complete]
    if len(actual) != len(set(actual)) or any(e["muscles"] != 700 for e in complete):
        raise ValueError("Duplicate constructor or changed muscle count")
    return {
        "expected_worlds": 10,
        "completed_worlds": len(actual),
        "all_worlds_constructed": set(actual) == expected,
        "known_preparation_substeps_lower_bound": sum(
            e["preparation_substeps"] for e in complete
        ),
        "known_fixture_physics_substeps_lower_bound": sum(
            e["physics_substeps"] for e in complete
        ),
        "interrupted_constructor_work": "unknown"
        if set(actual) != expected
        else "none",
    }


def guard(folder, trial, role, operation):
    def blocked(*args, **kwargs):
        emit(folder, trial, role, "guard_violation", operation=operation)
        raise GuardViolation(operation)

    return blocked


def install_guards(stack, folder, trial, role, torch, actor_class, env_class):
    """Patch in the diagnostic process only; no source edits or global service state."""
    for name in ("predict", "_predict", "forward", "action_log_prob"):
        if hasattr(actor_class, name):
            stack.enter_context(
                patch.object(
                    actor_class, name, guard(folder, trial, role, "actor." + name)
                )
            )
    stack.enter_context(
        patch.object(env_class, "step", guard(folder, trial, role, "env.step"))
    )
    for name in dir(torch.optim):
        cls = getattr(torch.optim, name)
        if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
            stack.enter_context(
                patch.object(
                    cls, "step", guard(folder, trial, role, "optimizer." + name)
                )
            )


def environment_facts(env):
    return {
        "specification": env.specification(),
        "muscles": int(env.model.nu),
        "preparation_substeps": int(env.preparation_total),
        "physics_substeps": int(env.physics_total),
    }


def profile_worker(out, index, seed, scale, *, events, trial):
    role = f"world-{index}"
    emit(events, trial, role, "world_before")
    import torch
    from stable_baselines3.sac.policies import Actor

    from . import wave

    stack = contextlib.ExitStack()
    install_guards(stack, events, trial, role, torch, Actor, wave.StanceEnv)
    env = wave.make_worker(out, index, seed, scale)
    # Keep guards installed for this child's whole command loop.
    env._joxing_profile_guards = stack
    emit(events, trial, role, "world_after", **environment_facts(env.unwrapped))
    return env


class TrialMemoryProfile:
    """Instrument exact original constructors, then stop at learn entry."""

    def __init__(self, folder, trial, hold_s=60):
        self.folder, self.trial, self.hold_s = Path(folder), trial, hold_s
        self.events = self.folder / "events"
        self.stack = contextlib.ExitStack()

    def event(self, phase, **values):
        return emit(self.events, self.trial, "learner", phase, **values)

    def __enter__(self):
        self.event("imports_before")
        import torch
        from stable_baselines3.common import vec_env
        from stable_baselines3.sac.policies import Actor

        from . import wave

        self.event("imports_after")
        profile = self
        original_env, original_vec, sac = (
            wave.StanceEnv,
            vec_env.SubprocVecEnv,
            wave.timed_sac_class(),
        )
        install_guards(
            self.stack, self.events, self.trial, "learner", torch, Actor, original_env
        )

        class ProfileEnv(original_env):
            def __init__(self, *args, **kwargs):
                profile.event("evaluation_env_before")
                super().__init__(*args, **kwargs)
                profile.event("evaluation_env_after", **environment_facts(self))

        class ProfileVec(original_vec):
            def __init__(self, *args, **kwargs):
                profile.event("vector_before")
                super().__init__(*args, **kwargs)
                profile.event("vector_after", worker_count=len(self.processes))

        class ProfileSAC(sac):
            def __init__(self, *args, **kwargs):
                profile.event("sac_before")
                super().__init__(*args, **kwargs)
                profile.event(
                    "sac_after",
                    actor_hash=wave.canonical_actor_hash(self.actor),
                    updates=self._n_updates,
                )

            def learn(self, *args, **kwargs):
                actor_file = (
                    profile.folder
                    / "fixtures"
                    / f"R-{profile.trial}"
                    / "initial-actor.pt"
                )
                profile.event(
                    "learn_ready",
                    actor_hash=wave.canonical_actor_hash(self.actor),
                    updates=self._n_updates,
                    initial_actor_file={
                        "bytes": actor_file.stat().st_size,
                        "sha256": hashlib.sha256(actor_file.read_bytes()).hexdigest(),
                    },
                )
                deadline = time.monotonic() + profile.hold_s
                while (
                    time.monotonic() < deadline
                    and not (profile.folder / "release").exists()
                ):
                    time.sleep(0.05)
                if not (profile.folder / "release").exists():
                    raise GuardViolation("Readiness barrier not released")
                raise ProfileExit()

            train = guard(self.events, self.trial, "learner", "model.train")
            predict = guard(self.events, self.trial, "learner", "model.predict")
            _setup_learn = guard(
                self.events, self.trial, "learner", "model._setup_learn"
            )

        self.stack.enter_context(patch.object(wave, "StanceEnv", ProfileEnv))
        # A top-level callable is imported afresh in vector workers; no inherited patches assumed.
        self.stack.enter_context(
            patch.object(
                wave,
                "make_worker",
                partial(profile_worker, events=self.events, trial=self.trial),
            )
        )
        self.stack.enter_context(patch.object(vec_env, "SubprocVecEnv", ProfileVec))
        self.stack.enter_context(
            patch.object(wave, "timed_sac_class", lambda: ProfileSAC)
        )
        return self

    def __exit__(self, *exc):
        self.stack.close()


def trial_target(folder, arm, config):
    profile = TrialMemoryProfile(folder, arm)
    try:
        with profile:
            from .provenance import manifest
            from .wave import train_run

            train_run(
                Path(folder) / "fixtures" / f"R-{arm}",
                config,
                arm,
                1097,
                [67010],
                manifest(),
                lambda _: None,
                defer_initial_evaluation=True,
            )
    except ProfileExit:
        profile.event("profile_exit", training_executed=False, inference_executed=False)
    except BaseException as error:
        profile.event("profile_fault", error=type(error).__name__ + ": " + str(error))
        raise


def run_children(folder, config):
    """Match two learners/eight worlds plus original manager/resource-tracker overhead."""
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        children = [
            context.Process(target=trial_target, args=(folder, arm, config))
            for arm in ("A", "B")
        ]
        for child in children:
            child.start()
        from multiprocessing import resource_tracker

        controls = [
            {"identity": identity(os.getpid()), "role": "orchestrator"},
            {"identity": identity(manager._process.pid), "role": "manager"},
        ]
        if resource_tracker._resource_tracker._pid is not None:
            controls.append(
                {
                    "identity": identity(resource_tracker._resource_tracker._pid),
                    "role": "resource_tracker",
                }
            )
        controls += [
            {"identity": identity(child.pid), "role": "learner", "trial": arm}
            for arm, child in zip(("A", "B"), children, strict=True)
        ]
        temporary = Path(folder, "control-roles.tmp")
        temporary.write_text(json.dumps(controls) + "\n")
        temporary.replace(Path(folder, "control-roles.json"))
        for child in children:
            child.join()
        if any(child.exitcode != 0 for child in children):
            raise RuntimeError("Profile child failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--packet", required=True)
    args = parser.parse_args()
    packet = json.loads(Path(args.packet).read_text())
    run_children(args.folder, packet["trial_config"])
