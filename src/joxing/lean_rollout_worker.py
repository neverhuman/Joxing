"""Opt-in pinned SB3 2.7.1 Gymnasium pipe loop; no eager heavy imports.

Adapted from stable-baselines3/common/vec_env/subproc_vec_env.py (MIT).
Only target/serialization/helper imports differ. Legacy/default paths unchanged.
Full upstream license: lean_rollout_worker_LICENSE.txt alongside this module.
"""

from typing import Any


class LeanCloudpickleWrapper:
    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.var)

    def __setstate__(self, var):
        import cloudpickle

        self.var = cloudpickle.loads(var)


def _patch_env(env):
    import gymnasium as gym

    if not isinstance(env, gym.Env):
        raise TypeError(
            "lean-gymnasium-v1 requires Gymnasium Env; legacy Gym is unsupported"
        )
    return env


def unwrap_wrapper(env, wrapper_class):
    import gymnasium as gym

    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env, wrapper_class):
    return unwrap_wrapper(env, wrapper_class) is not None


def lean_pipe_worker(remote, parent_remote, env_fn_wrapper):

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: dict[str, Any] | None = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except KeyboardInterrupt:
            break
