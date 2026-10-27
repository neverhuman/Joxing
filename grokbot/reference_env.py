"""BoxingEnv with the grokbot reference ending as its terminal rule and reward.

The diagnostic ``BoxingEnv`` ends only at the bell and pays only points, so
without contact every transition reward is zero. This wrapper leaves the env
untouched and, after each control step, feeds the new evidence rows to an
incremental ``endings.BoutMonitor`` and checks complete exits at the control
boundary. The bout ends at the first decisive event of the reference table
(``grokbot.endings``):

* a sporting result: rewards are the repo's own ``match_reward`` with the
  utility as terminal outcome, so per-step rewards are zero-sum and the
  shaped return telescopes exactly to the utility;
* a blocked bout (unresolved attribution): it terminates with
  ``info["blocked"]``, and the reward returns the accumulated potential to
  zero, so the return is 0 and a learner can drop the episode.

Engine faults propagate from the env unchanged. They are invalid, never a
result.
"""

from __future__ import annotations

import numpy as np

from grokbot import endings
from joxing.env import PLAYERS
from joxing.rules import match_reward, potential

RING_HALF_SIZE_M = 2.5


class ReferenceEndingEnv:
    def __init__(self, env):
        self.env = env
        self.registry = {g["id"]: g for g in env.scene.report["physical_geoms"]}
        self.bodies = {
            a: [
                b
                for b in range(env.model.nbody)
                if env.model.body(b).name.startswith(a + "_")
            ]
            for a in PLAYERS
        }

    def reset(self, *, seed):
        observations, info = self.env.reset(seed=seed)
        self.monitor = endings.BoutMonitor(self.registry)
        self.seen = 0
        self.exits = {}
        self.diff = 0
        self.ending = None
        return observations, info

    def _exits(self):
        for a, bodies in self.bodies.items():
            xy = self.env.data.xpos[bodies, :2]
            if a not in self.exits and np.all(
                np.abs(xy).max(axis=1) > RING_HALF_SIZE_M
            ):
                t = self.env.tick - 1
                self.exits[a] = endings.LossEvent(
                    a, t, "exit", self.monitor.attribute(t, t)
                )

    def step(self, commands, *, delays=None):
        if self.ending is not None:
            raise RuntimeError("Bout already decided; reset first")
        result = self.env.step(commands, delays=delays)
        self.monitor.update(self.env.evidence[self.seen :])
        self.seen = len(self.env.evidence)
        self._exits()
        points = self.env.referee.points
        bell = self.env.done and not self.env.truncated
        events = self.monitor.losses + list(self.exits.values())
        ending = endings.decide(points, events, bell=bell, bell_tick=self.env.tick - 1)
        decided = ending.utility is not None or ending.reason.startswith("blocked")
        new_diff = points[0] - points[1]
        if not decided:
            reward_a, _ = match_reward(self.diff, new_diff, terminal=False)
        elif ending.utility is None:
            reward_a = -potential(self.diff)  # blocked: return telescopes to 0
        else:
            reward_a, _ = match_reward(
                self.diff, new_diff, terminal=True, outcome=ending.utility[0]
            )
        self.diff = new_diff
        if decided:
            self.ending = ending
        info = dict(result["info"])
        info.update(
            ending=None
            if not decided
            else {
                "utility": ending.utility,
                "reason": ending.reason,
                "tick": ending.tick,
            },
            blocked=decided and ending.utility is None,
            losses=[vars(e) for e in events],
        )
        return {
            "observations": result["observations"],
            "rewards": {"A": reward_a, "B": -reward_a},
            "terminated": decided,
            "info": info,
        }
