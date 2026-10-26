"""Inspect actual observation/control contracts and recorded command variation.

This is an offline code/trace diagnostic, never an additional motor proof.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from joxing.provenance import manifest, write_json
from joxing.stance_env import StanceEnv
from joxing.stance_train import file_hash


def main():
    import torch
    from stable_baselines3.sac.policies import Actor

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    env = StanceEnv(training=False)
    obs, _ = env.reset(options={"scenario_id": 67099})
    start, end = env.obs_slices["velocity"]
    assert np.array_equal(
        obs[start:end], np.tanh(env.data.qvel / 10).astype(np.float32)
    )
    traces = []
    for arm in "ABCD":
        for name, checkpoint in (("initial", "initial"), ("trained", "final")):
            path = args.run / f"S-{arm}/{name}-replay.json"
            record = json.loads(path.read_text())
            actions = np.asarray(record["actions"])
            assert actions.ndim == 2 and actions.shape[1] == 138 and len(actions) > 1
            assert np.isfinite(actions).all() and np.max(abs(actions)) <= 1
            actor = Actor.load(
                args.run / f"S-{arm}/{checkpoint}-actor.pt", device="cpu"
            )
            a, hidden = actor.predict(obs, deterministic=True)
            assert hidden is None
            counterfactual = obs.copy()
            dof = int(env.model.joint("pelvis_tilt").dofadr[0])
            counterfactual[start + dof] = np.tanh((env.data.qvel[dof] + 1) / 10)
            b, _ = actor.predict(counterfactual, deterministic=True)
            traces.append(
                {
                    "arm": arm,
                    "policy": name,
                    "scenario_id": record["result"]["scenario_id"],
                    "trace_sha256": file_hash(path),
                    "actor_sha256": file_hash(
                        args.run / f"S-{arm}/{checkpoint}-actor.pt"
                    ),
                    "recorded_commands_after_snapshot": len(actions),
                    "snapshot_sim_s": record["snapshot"]["state"][0],
                    "saturated_normalized_action_fraction_abs_ge_0_99": float(
                        np.mean(abs(actions) >= 0.99)
                    ),
                    "groups_varying_std_above_1e_6": int(
                        sum(np.std(actions, axis=0) > 1e-6)
                    ),
                    "mean_absolute_normalized_command_change": float(
                        np.mean(abs(np.diff(actions, axis=0)))
                    ),
                    "counterfactual_velocity_input_mean_action_change": float(
                        np.mean(abs(a - b))
                    ),
                    "counterfactual_scope": "Only the offline observation channel changes (+1rad/s pelvis_tilt). No simulated trajectory or motor outcome is measured.",
                    "actor_module_types": sorted(
                        {type(m).__name__ for m in actor.modules()}
                    ),
                    "predict_hidden_state": None,
                }
            )
    result = manifest() | {
        "audit_script_sha256": file_hash(Path(__file__)),
        "scope": "offline observation/network/recorded-action review; no motor trial or self-play claim",
        "specification": env.specification(),
        "observation_size": len(obs),
        "native_muscles": env.model.nu,
        "own_generalized_velocity_channels_verified": end - start,
        "control_sim_s": env.config.control_s,
        "physics_sim_s": env.config.dt_s,
        "previous_action_channels": env.obs_slices["previous_action"][1]
        - env.obs_slices["previous_action"][0],
        "traces": traces,
        "limitations": [
            "One preselected recorded scenario per actor; variation is not stability or correct feedback.",
            "Command differences are normalized action variation, not measured limb jerk or a causal jitter study.",
            "Counterfactual input sensitivity does not establish useful motor response.",
            "Replay buffers and logged traces are not policy recurrent memory.",
            "No opponent observation, match state, temporal encoder, simulated delay or self-play exists in the current StanceEnv path.",
        ],
    }
    write_json(args.out, result)
    print(
        json.dumps(
            {
                "observation_size": len(obs),
                "verified_velocity_channels": end - start,
                "traces": len(traces),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
