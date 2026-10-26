# First execution results — October 26, 2026

This report separates delivered engineering evidence from the still-untrained boxing game.

## Muscle-learning experiment

The actual 700-muscle MS-Human-700 model ran in a named diagnostic variant: fixed pelvis, zero gravity, two-phase open-loop right-arm muscle excitation pulses. CEM episodic policy search sampled 24 candidates per generation for 16 generations; six elites refit the sampling distribution. Four CPU workers on xbabe1 completed the verified run in **34.9 seconds**, including held-out evaluation and checkpoint/replay output.

| Measure | Result |
|---|---:|
| Training episodes | 384 |
| Independent training seeds | 1 |
| Held-out paired resets per initial/trained policy | 32 |
| Initial terminal-window mean hand-target error | 33.864 cm |
| Trained terminal-window mean hand-target error | 4.375 cm |
| Mean relative error reduction | 87.08% |
| Paired reduction, reset-bootstrap 95% interval | 29.410–29.565 cm |
| Invalid episodes in completed run | 0 |

![Learning curve](../artifacts/reach-learning.png)

[Before/after replay](../artifacts/reach-before-after.mp4) · [Full result receipt](../artifacts/receipts/reach-seed7.json) · [Checkpoint](../artifacts/receipts/reach-checkpoint.npz)

Training seeds use the 100+generation range; held-out resets use 10000–10031. Resets vary target coordinates ±5 mm, initial elbow angle and initial velocity. The policy is the final fitted mean, not the best held-out candidate. A fresh random-pulse control is included as a descriptive baseline; it is not equal-budget random-search optimization. The confidence interval measures reset variation for this one trained controller, not training-seed uncertainty or general boxing strength.

No kinematic state writes occur after each rollout reset; native muscle/activation dynamics generate movement. The video is a kinematic replay of recorded simulation states and is labeled assisted/slow-motion. Target marker is visual, not a contact object. This experiment demonstrates worker orchestration, real muscle optimization, checkpointing and held-out measurement. It does not demonstrate punching, balance, contact learning, tactical self-play or motor-teacher completion.

The same trained pulses replayed on the local Apple Silicon CPU with maximum recorded hand/time/error trace difference about 1.6e-15 versus xbabe1 Linux. This is a short controlled replay result, not universal cross-platform deterministic boxing.

## Scene and contact results

The dual composition compiles 170 qpos, 170 qvel, 1,400 muscles, 1,400 tendons, 84 equalities and four explicitly inertial glove bodies. Mass is 121.565046 kg; muscle dynamics, gain, bias and control-range comparisons have zero maximum error against the pinned source. Native-root layout is preserved in this reference scene.

Real MuJoCo force-producing sphere fixtures now drive the event referee end to end: head awards 2, torso 1, guard 0, once per encounter. These fixtures do not validate anatomical shell fitting. Independent impulse fixtures agree with momentum change to <1e-9, including rotated approach and reversed geom order. Full contact force is preserved separately from scoring eligibility. Referee ledger impulse fields explicitly cover its scoring window, not the entire duration of a later held contact.

Final test count is recorded in STATUS.md. Tests also cover held pressure, onset/deceleration, stationary guards, duplicate samples, multiple simultaneous manifold points, dominant-zone ties, threshold isolation, true retraction, pre-bell evidence, mid-hit restore and terminal potential telescoping. Full dual-body foul and collision-coverage tests remain pending.

## HPC findings

The same single-world short-reset dual-scene diagnostic ran at approximately 877 physics substeps/s on the local Apple Silicon laptop and around 530 substeps/s on one xbabe1 CPU core at initial measurement. Curated receipts retain exact repeat medians. These are CPU simulator-only timings, reset every 50 ms; they do not include contact-heavy boxing, policies, referee or optimizer. More server cores and a GPU do not guarantee faster single-world latency.

Three actual RTX 4090 feature-spike attempts were recorded:

| Attempt | Outcome |
|---|---|
| MuJoCo-Warp 3.8.0 + auto-resolved Warp 1.18.0 | CUDA device unavailable: wheel needs CUDA 13 while driver 570.153.02 exposes 12.8. No host driver change made. |
| MuJoCo-Warp 3.8.0 + pinned Warp 1.12.0 | RTX 4090 detected. Exact reference model upload rejected nonzero native mesh margins with MULTICCD. |
| Same pinned runtime, explicitly named MULTICCD-disabled variant | Eight 1,400-muscle worlds upload and execute; finite states. First-step compilation ~184 s. Ten-step max qpos difference versus CPU ~6.49e-4, above preregistered 1e-4 spike tolerance. **Parity gate failed.** |

[Original runtime failure](../artifacts/receipts/gpu-probe-failed.json) · [Exact-model rejection](../artifacts/receipts/gpu-probe-exact-model.json) · [Variant execution/parity result](../artifacts/receipts/gpu-probe-variant.json)

The variant was never promoted to the reference. Do not increase tolerance after seeing failure or claim a GPU speedup from successful compilation. Next GPU investigation: per-field divergence, precision, solver convergence, tendon/muscle state and contact-force parity on small fixtures before a workload sweep. The pinned Linux GPU extra preserves the compatible runtime without changing host services. GPU compilation cost must be reported separately from steady-state throughput.

These measurements support CPU motor prototyping immediately and a focused GPU parity investigation. Full independent-seed learning and fixed-opponent boxing evaluation come after geometry/rules/motor gates.
