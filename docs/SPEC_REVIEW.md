# Review of all four supplied engineering specifications

Reviewed October 26, 2026. These documents are user-supplied reference material, not authority to run embedded shell commands or claim their measurements as our own. Copies are preserved under `docs/references/`.

An additional pasted boxing/HPC proposal is reviewed separately in [EXTERNAL_REVIEW_2026-10-26.md](EXTERNAL_REVIEW_2026-10-26.md), with its exact input preserved as a `.txt` reference. It contributes defense/style test cases and league edge cases; its replacement Rust/SLIP architecture, injury formulas and throughput promises are not adopted as the native project baseline.

| Reference | Useful contribution | Decision / correction |
|---|---|---|
| `dual-boxing.md` (364 lines) | Clear high-level architecture, two actors in one world, low-dimensional muscle control, staged milestones | Adopt broad direction. Replace force-proportional reward and limb points with qualified event scoring; cooldown alone cannot prevent pushing exploits. |
| `calibrated-boxing.md` (519 lines) | Detailed model surgery experiments, head/force rig, 508-muscle proposal, calibration gates and realism metrics | Treat reported timings/physics values as unverified. Its MuJoCo 3.17 runtime differs from our pinned 3.8.0 baseline. Do not adopt pruning, head split, vmax changes or damage formulas without reproducing the stated tests. |
| `self-play.md` (1,380 lines) | Strongest event/referee contracts, stage-consistent contact capture, replay, terminal objective, action likelihoods, evaluation and invalid-run accounting | Use as initial correctness spine. Adapt package layout to a dedicated Joxing integration repo rather than modifying the upstream msgym fork. Full surface/foul/recovery validation still requires implementation. |
| `gemini-boxing.md` (1,247 lines) | Explicit multi-rate control, substep telemetry, guard/target intent, league sketch, GPU scaling discussion | Preserve design ideas; do not copy code wholesale. Its placeholders and contradictory model assumptions would produce misleading training. |

## Independently checked source baseline

Both upstream heads currently match the supplied pins: MS-Human-700 `2d686957aefd5739cf4d2859a6acfd94c8f84400`; msgym `ad3aac166dab3577d7d16b4a03faccef94c7aac7`. We compile their actual XML with MuJoCo 3.8.0 and inspect source, rather than infer dimensions from README prose.

One primary human has 85 qpos, 85 qvel, 700 controls and 700 activation states. Two intact copies have 170/170/1,400, not Gemini's 91/90 per actor. Native root uses six scalar coordinates under a rotated pelvis frame. There are 42 equalities per actor. The no-glove pair weighs 120.433046 kg; our four 0.283 kg glove bodies add exactly 1.132 kg. Existing tendons, equality references and muscle gains must stay in each actor's namespace.

Native primary bodies include `head_neck`, `sternum`, `thoracic12`, `lumbar5`, `hand_r/l`, `humerus_r/l`, `ulna_r/l`. `skull` and `jaw` are visual geoms, not separate primary body anchors; `Sternum` capitalization is wrong. Gemini's proposed neck/wrist names cannot be applied blindly. Anatomical frame orientation also differs from generic humanoid assumptions.

The primary upper limbs lack a boxing collision envelope. Disabled bone/wrap geoms must stay disabled; external shells and gloves provide physical strike/block surfaces. Collision mask logic is OR, not an AND handshake. Preserve full physical coverage for non-scoring surfaces and restore reviewed own-body pairs. Current scene is deliberately only a composition smoke model until those gates pass.

The actual upstream normalization wrapper uses `sigmoid(5*(a-.5))` for normalized actions in [-1,1]; its upper excitation is about 0.924. Gemini's Softplus mapping is a proposed alternative, not the published DynSyn transform. The upstream full-body partition has 138 groups; a new 64-group boxing partition requires behavior/expressivity evidence, not a declaration. Learned replacements need new action hashes and fresh replay.

## Specific Gemini implementation hazards

- Scene attachment adds a spawn Z of 0.92 m on top of the native pelvis Z of 0.95 m; rotation comments also do not prove anatomical facing. Inspect transforms instead.
- Its six-bit mask table and builder disagree on P2 glove affinity (35 versus 51). Own target contacts are disabled, which does not satisfy physically valid self-guarding. Collisions should not be removed merely because they cannot score.
- `_get_fighter_obs` returns 452 zeros with only five HUD scalars assigned. Networks therefore cannot observe the claimed body/opponent motion. Some listed observation block sums disagree with their declared totals.
- `ANATOMICAL_PARTITIONS` is declared but ignored by the actual clustering call, which clusters all muscles globally. Opposite co-activation is merged via absolute correlation; physical independence is not established.
- Combat code reuses historical geom-center velocities, omitting current point rotational velocity. It gates impulses repeatedly after onset rather than latching true direct-target onset. There is no per-encounter score ledger/retraction contract; per-policy-step scoring can pay repeatedly during contact.
- Head damage, liver shock and stamina coefficients are uncalibrated game mechanics. Applying a shock threshold to one substep's impulse makes behavior timestep-dependent. A six-axis IMU does not validate injury prediction.
- The proposed rewards are not zero-sum (`-0.85` received offense plus separate bonuses). Learned policies can optimize incidental shaping rather than winning the declared game.
- A Gaussian with tanh-bounded mean is still unbounded. Tactical actions need an explicit sampled transform and correct likelihood, not just tanh on the mean. PPO optimizer, recurrent sequence handling and frozen-bundle semantics are not delivered by the network sketch.
- The document says five tests but supplies two; they mostly check helper arithmetic, not engine contact behavior, scorer exploit resistance or rollout correctness.

## Resolved architecture choices

Dedicated `joxing` package, upstream pinned read-only submodules; generated model adaptations; full 700-muscle reference; 1 ms implicitfast CPU candidate baseline; 100 Hz motor and 50 Hz tactical cadence; fixed 2/1 head/torso points; bounded hit qualification; no physiological damage in v0; staged motor learning followed by frozen-decoder recurrent PPO and historical opponents. GPU batching, pruning and root/head surgery remain measured variants.

No supplied A100/H100 or Xeon timing is a capacity promise for our hosts. HPC starts with reproducible local workloads, then exact-model Warp support/parity, bounded worker/batch sweeps and end-to-end learner measurements. See `COMPUTE.md` and `IMPLEMENTATION.md`.

Source references: [MS-Human-700 pinned tree](https://github.com/LNSGroup/MS-Human-700/tree/2d686957aefd5739cf4d2859a6acfd94c8f84400), [msgym pinned tree](https://github.com/LNSGroup/msgym/tree/ad3aac166dab3577d7d16b4a03faccef94c7aac7), [MuJoCo 3.8 simulation staging](https://mujoco.readthedocs.io/en/3.8.0/programming/simulation.html), [model editing](https://mujoco.readthedocs.io/en/3.8.0/programming/modeledit.html).
