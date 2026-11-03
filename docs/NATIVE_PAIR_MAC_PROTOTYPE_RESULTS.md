# H31 first actual native two-humanoid PPO witness

The real-checkpoint native learning loop now runs: **32joint physical transitions,
64actor samples, four PPOrollout updates/eight optimizer epochs**, with changed
actor tensors and unchanged pretrained motor. All32transitions independently
replay bit-for-bit through the actual native engine; all700commands per body
reconstruct from the frozen teacher and learned residuals. Behavior values/log
probabilities independently match with0error at all four update boundaries.

Public source21012335dec35772279dfeb9eb4bb9748321c583; preregistered protocol
7fd7abe99e2a1e6d89d5b2e7fca52584ebd3d0073836798b6c91b8412bc3a87a.
One attempted/one valid prototype,0invalid/not_run,22fresh source cases pass.
Actual released actor963d2f56/normalizer34f38e6e/reference89f78fe4 hashes match.
The task uses two intact700muscle humans in ONEopposed shared1msimplicitfast
world. Both policies observe four causal own/opponent motion frames, explicit
limb velocities, own activation/commands and current public motor goals; no
opponent activation or hidden reference enters the learner. Actions are paired
before physics and held20ticks. No evaluated pose/root/gravity assistance.

This is motor RLtoward boxing, **not a qualified bout or demonstrated boxing
improvement**. Training spans only.64simulated seconds and does not finish a
training episode. Two fixed evaluation episodes each finish the1s horizon,
with neither physical-failure latch firing; both scores remain0–0. The first,
middle and last frames show approach to close body contact, mostly low gloves,
and no convincing punch/guard sequence. Tiny motor-return differences do not
establish improvement, generalization or competitive strength.

## Fixed complete recordings

[Before](../artifacts/v0.0.11/native-pair-mac-prototype-01/before.mp4) ·
[After](../artifacts/v0.0.11/native-pair-mac-prototype-01/after.mp4).
Both show the entire first fixed evaluation,51actual frames at50fps, including
initial/final state. No seed/clip selection by reward. Both recordings explicitly
label motor-prototype status and unqualified sporting points.

| Metric | Frozen walking baseline | After four PPOupdates |
|---|---:|---:|
| Fixed evaluation duration |1.000s|1.000s|
| Candidate points |0–0|0–0|
| Motor-failure latches |none|none|
| A motor return |0.7228156828355009|0.7237889770048552|
| B motor return |0.7228509245044249|0.7238590997048788|

One paired start, one training seed, a short horizon and correlated two-corner
samples cannot support a skill/improvement claim. This is a functional native
rollout→gradient→replay→recording proof. Historical-opponent competitive self-play
and the larger two-seed native pilot remain unrun.

## Audit, resources and every outcome

Original host epoch through completed source/native/render/audit/owner publication:
86.472088667s within240s. Native phase12.6548165s including model load, actual
checkpoint replay, training/checkpointing and both evaluations. Peak SAMPLED
owned RSS2,696,052,736B under8GiB; all11observed owner/child identities exited.
Raw130,172,115B under256MiB,53regular payloads with exact retained hashes.
All final JSON/JSONL/XML/log text totals840148B; conservative atomic-owner
temporary reserve32768B keeps text below1MiB. The prototype guarded aggregate raw/
RSS/wall online; the separate text ceiling was checked post-task, not by its own
online writer guard. Add that typed guard before scaling.
Checkpoints/tapes/RNG states remain outside Git in the dedicated frozen root;
Git contains small summaries/manifests/logs and the complete small fixed videos.

Primary guarded native entries2877: actual pretrained replay237, training640,
and evaluations2000. Independent audit adds640actual replay entries. Exact guarded
native total3517; conservative source-fixture reserve768 (fixed source loop count
563, not a measured768) yields4285charged under4608. Source fixture actors are
explicitly synthetic; only the real-checkpoint native trial establishes the new
PPOwitness. Earlier development loops/faults stay separate in accounting.json.

The registered platform is the local Mac with Python3.12.12/NumPy2.5.4/MuJoCo3.8.0/
Torch2.14.1/SB32.9.2. It does not relabel the pinned Linux stack, failed cross-host
control import, Linux100msretirement gate or old H27native prefix as qualified.
No old physics, rules, muscles or frozen model assets changed. No competitive
learning, qualified legal strike, historical-opponent league or v11tag.

[Summary](../artifacts/v0.0.11/native-pair-mac-prototype-01/summary.json),
[independent audit](../artifacts/v0.0.11/native-pair-mac-prototype-01/independent-audit.json),
[post-task source/exit/visual audit](../artifacts/v0.0.11/native-pair-mac-prototype-01/post-task-audit.json),
[all source/update/native outcomes](../artifacts/v0.0.11/native-pair-mac-prototype-01/owner.json),
[physics/accounting](../artifacts/v0.0.11/native-pair-mac-prototype-01/accounting.json),
[raw file hashes](../artifacts/v0.0.11/native-pair-mac-prototype-01/raw-manifest.json.gz).
Raw root: /Users/bentaylor/joxing/native-pair-mac-prototype-20261102-01/runs/v0.0.11/native-pair-mac-prototype-01.

## Next native win

[Continue H31with two-seed motor training](NATIVE_PAIR_MOTOR_TRAINING_CONTINUATION.md):
2048joint transitions/seed,3s fixed evaluations, explicit guard/reach-return goals
and unchanged balance/physics. First stream evidence/checkpoints within1GiB;
raising the old smoke loop unchanged would exhaust raw credit. Register fresh
source/protocol/caps before outcomes. H30stays secondary; no new fake benchmark
is needed to begin this behavioral work. Do not call a large fall/zero-point
sweep valid boxing games. The original main pilot/gates remain in the mission.
