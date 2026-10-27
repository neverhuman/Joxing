"""Atomic engine+application snapshots for the diagnostic shared world."""

import copy

import mujoco
import numpy as np

from .history import TimedHistory
from .observations import SCHEMA_HASH
from .provenance import digest
from .referee import MatchReferee
from .timing import ActionQueue


class ReplaySnapshot:
    @staticmethod
    def capture(env):
        if (
            not env.initialized
            or env.invalid_reason
            or digest(env.contract) != env.contract_hash
        ):
            raise ValueError("Cannot checkpoint uninitialized/invalid/drifted world")
        state = env.state()
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite engine snapshot")
        app = {
            "game_id": env.game_id,
            "episode_serial": env.episode_serial,
            "tick": env.tick,
            "remaining_control": env.remaining_control,
            "control_start": env.control_start,
            "done": env.done,
            "truncated": env.truncated,
            "end_reason": env.end_reason,
            "commands": {a: c.tolist() for a, c in env.commands.items()},
            "referee": env.referee.snapshot(),
            "queues": {a: q.snapshot() for a, q in env.queues.items()},
            "histories": {a: h.snapshot() for a, h in env.histories.items()},
            "rngs": {
                a: copy.deepcopy(r.bit_generator.state) for a, r in env.rngs.items()
            },
            "policy_states": copy.deepcopy(env.policy_states),
            "observation_tick": env.observation_tick,
            "decision_state": env.decision_state.tolist(),
            "points_at_observation": list(env.points_at_observation),
            "commands_at_observation": {
                a: c.tolist() for a, c in env.commands_at_observation.items()
            },
            "cached_observations": {
                a: o.values.tolist() for a, o in env.cached_observations.items()
            },
            "last_events": copy.deepcopy(env.last_events),
            "last_rewards": list(env.last_rewards),
        }
        value = {
            "schema": "joxing-shared-replay-v1",
            "contract_hash": env.contract_hash,
            "engine": state.tolist(),
            "application": app,
        }
        return value | {"sha256": digest(value)}

    @staticmethod
    def restore(env, snapshot):
        from .env import INTEGRATION, PLAYERS, excitation

        s = copy.deepcopy(snapshot)
        expected = s.pop("sha256")
        if (
            digest(s) != expected
            or set(s) != {"schema", "contract_hash", "engine", "application"}
            or s["schema"] != "joxing-shared-replay-v1"
            or s["contract_hash"] != env.contract_hash
            or digest(env.contract) != env.contract_hash
        ):
            raise ValueError("Foreign/corrupt shared snapshot")
        app = s["application"]

        def state(values):
            raw = np.asarray(values, np.float64)
            if (
                raw.shape != (mujoco.mj_stateSize(env.model, INTEGRATION),)
                or not np.isfinite(raw).all()
            ):
                raise ValueError("Malformed engine state")
            data = mujoco.MjData(env.model)
            mujoco.mj_setState(env.model, data, raw, INTEGRATION)
            if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
                raise ValueError("Snapshot contains external assistance")
            mujoco.mj_forward(env.model, data)
            if any(w.number for w in data.warning):
                raise ValueError("Restored state emits engine warnings")
            # Derived kinematics are current; preserve the exact integration
            # warm-start/control state after forward's diagnostic solve.
            mujoco.mj_setState(env.model, data, raw, INTEGRATION)
            return data

        def commands(values):
            if set(values) != set(PLAYERS):
                raise ValueError("Both command owners required")
            result = {}
            for a, command_values in values.items():
                raw = np.asarray(command_values, np.float64)
                if (
                    raw.shape != (700,)
                    or not np.isfinite(raw).all()
                    or np.any(np.abs(raw) > 1)
                ):
                    raise ValueError("Malformed held command")
                result[a] = raw.astype(np.float32)
            return result

        tick = app["tick"]
        remaining = app["remaining_control"]
        start = app["control_start"]
        if (
            any(
                type(v) is not int
                for v in [
                    tick,
                    remaining,
                    start,
                    app["observation_tick"],
                    app["episode_serial"],
                ]
            )
            or not 0 <= start <= tick <= env.horizon_ticks
            or not 0 <= remaining <= 20
            or not 0 <= tick - start <= 20
            or app["episode_serial"] < 1
        ):
            raise ValueError("Invalid shared control clock")
        if remaining and tick - start + remaining != 20:
            raise ValueError("Invalid pending interval")
        if app["observation_tick"] != (start if remaining else tick):
            raise ValueError("Invalid decision cache clock")
        candidate = copy.copy(env)
        candidate.data = state(s["engine"])
        if abs(candidate.data.time - tick * 0.001) > 1e-9:
            raise ValueError("Engine/application clock mismatch")
        candidate.game_id = app["game_id"]
        candidate.episode_serial = app["episode_serial"]
        if not candidate.game_id:
            raise ValueError("Missing game identity")
        candidate.commands = commands(app["commands"])
        candidate.referee = MatchReferee.restore(env.scene, app["referee"])
        if (
            candidate.referee.game_id != candidate.game_id
            or candidate.referee.next_tick != tick
        ):
            raise ValueError("Referee clock/owner differs from physics")
        candidate.queues = {
            a: ActionQueue.restore(
                app["queues"][a],
                game_id=candidate.game_id,
                player=a,
                bundle_hash=env.bundles[a],
            )
            for a in PLAYERS
        }
        for a, q in candidate.queues.items():
            applied = (
                np.zeros(700, np.float32)
                if q.last_applied is None
                else q.last_applied.command
            )
            if (
                q.last_polled_tick != tick - 1
                or not np.array_equal(applied, candidate.commands[a])
                or not np.array_equal(
                    candidate.data.ctrl[env.bodies[a].actuator_ids], excitation(applied)
                )
            ):
                raise ValueError("Held actuator state disagrees with applied queue")
        candidate.done = app["done"]
        candidate.truncated = app["truncated"]
        candidate.end_reason = app["end_reason"]
        if type(candidate.done) is not bool or type(candidate.truncated) is not bool:
            raise ValueError("Invalid global ending flags")
        if candidate.truncated:
            if (
                not candidate.done
                or candidate.referee.ended
                or not str(candidate.end_reason).startswith("external: ")
            ):
                raise ValueError("Invalid external truncation")
        elif (
            candidate.done != (tick == env.horizon_ticks)
            or candidate.referee.ended != candidate.done
            or candidate.end_reason != ("diagnostic_bell" if candidate.done else None)
        ):
            raise ValueError("Invalid bell/termination state")
        if candidate.done and remaining:
            raise ValueError("Ended world cannot hold pending physics")
        candidate.tick = tick
        candidate.control_start = start
        candidate.remaining_control = remaining
        candidate.observation_tick = app["observation_tick"]
        candidate.decision_state = np.asarray(app["decision_state"], np.float64)
        decision = state(app["decision_state"])
        if abs(decision.time - candidate.observation_tick * 0.001) > 1e-9:
            raise ValueError("Invalid cached decision state")
        candidate.commands_at_observation = commands(app["commands_at_observation"])
        candidate.points_at_observation = tuple(app["points_at_observation"])
        candidate.cached_observations = candidate._make_observations(
            decision,
            candidate.observation_tick,
            candidate.points_at_observation,
            candidate.commands_at_observation,
        )
        for a, o in candidate.cached_observations.items():
            if not np.array_equal(
                o.values, np.array(app["cached_observations"][a], np.float32)
            ):
                raise ValueError(
                    "Cached actor input differs from its actual decision state"
                )
        candidate.histories = {
            a: TimedHistory.restore(
                app["histories"][a],
                game_id=candidate.game_id,
                player=a,
                schema_hash=SCHEMA_HASH,
            )
            for a in PLAYERS
        }
        for a, h in candidate.histories.items():
            if any(
                r["received_tick"] > candidate.observation_tick
                for r in h.snapshot()["samples"]
            ):
                raise ValueError("Future history packet in actor cache")
            rows = h.snapshot()["samples"]
            if (
                not rows
                or rows[-1]["sampled_tick"] != candidate.observation_tick
                or rows[-1]["received_tick"] != candidate.observation_tick
                or not np.array_equal(
                    np.array(rows[-1]["values"], np.float32),
                    candidate.cached_observations[a].values,
                )
            ):
                raise ValueError("Latest actor history differs from its decision input")
        candidate.rngs = {a: np.random.default_rng() for a in PLAYERS}
        for a in PLAYERS:
            candidate.rngs[a].bit_generator.state = app["rngs"][a]
        candidate.policy_states = copy.deepcopy(app["policy_states"])
        candidate.last_events = copy.deepcopy(app["last_events"])
        candidate.last_rewards = tuple(app["last_rewards"])
        if (
            len(candidate.last_rewards) != 2
            or not np.isfinite(candidate.last_rewards).all()
            or sum(candidate.last_rewards) != 0
        ):
            raise ValueError("Invalid two-player reward state")
        candidate.invalid_reason = None
        candidate.initialized = True
        candidate.evidence = []
        # physics_total is intentionally not restored: attempted physics is a
        # monotonic audit counter even when the game clock is rewound.
        env.__dict__.update(candidate.__dict__)
