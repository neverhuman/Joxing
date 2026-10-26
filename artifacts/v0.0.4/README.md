# v0.0.4 evidence

This directory records a native-root, ordinary-gravity **development balance experiment**, not a boxing match or passed stance/recovery proof. [Execution plan](../../docs/V0.0.4_PLAN.md), [status](../../docs/STATUS.md).

Outcome: 39,040 wall-capped training transitions, all 64 paired evaluations completed with zero invalids, mean survival 1.018 -> 1.023 seconds and 0/32 five-second survivors. [Full results](../../docs/V0.0.4_RESULTS.md).

- `protocol.json`, `run-launch.json`: immutable training/evaluation identity, exact command, source commit/hash, seed, budgets and launch PID.
- `model-audit.json`, `reset-audit.json`, `root-probes.json`, `probe-result.json`: muscle/inertia/equipment preservation, all 32 accepted development starts and twelve finite root-stress cases. Short probes do not exclude every root singularity.
- `compute-inventory.json`, `learner-diagnostic.json`: sanitized shared-host inventory and in-flight optimizer/entropy-scale diagnostic. The entropy diagnostic samples one reset; it does not prove a causal explanation or constitute an ablation.
- `development-replay-failure.json`, `development-replay-fixed.json`: the caught sorted-JSON support-channel defect and its corrected exact smoke continuation. Preserve the failure as well as the fix.
- `initial-cross-host-audit.json`: the initial policy's Linux-to-Mac full-state tolerance miss. Physical-coordinate drift is much smaller than solver warmstart drift; do not treat the missed criterion as passed.
- `tests-*.txt`, `lint.txt`: actual platform test/lint receipts. Initial Linux tests preceded the additional restore/learner tests; use the final Linux receipt for release verification.
- `result.json`, `execution-summary.json`, `pilot/`: raw checkpoint hashes, initial/final policy bundles, every evaluation row, snapshots, progress and worker episode logs. Full planned training did not complete; all pilot evaluations did.
- `benchmark.json`, `replay-audit-linux.json`, `replay-audit-mac.json`: measured 1/2/4-worker workload and independent checks. Linux continuation passes; strict cross-host full-state parity remains failed, with separate accounting/same-host results retained.
- `balance-learning.png`, `balance-before-after.mp4`, `replay-preview.png`: inspected evidence, fixed scenario 30000; both policies fall. `artifact-manifest.json` hashes all published files except itself.

Policy checkpoints are evaluation/policy-optimizer bundles. Full training resume with replay buffer and all learner RNGs is not implemented. Hand targets are their prepared reset positions, not a raised boxing guard. No raw credentials or SSH configuration are included.
