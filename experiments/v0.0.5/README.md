# v0.0.5 experiment book

Read the [preregistered plan](../../docs/V0.0.5_PLAN.md) and [immutable machine protocol](protocol.json) first. The plan was committed as `09cbacb` before any wave outcomes. This iteration tests objective scaling and optimizer cost, preserving physical dynamics and the motor gate.

Every planned ID has its hypothesis and description in protocol.json. Execution writes a per-experiment description before work, an append-only event journal, raw initial/final evaluations (all attempts), progress, actor bundles and result receipts. Conditional replication remains explicitly not_run unless the registered selection rule passes. A negative screen is a useful completed experiment; incomplete training is reported separately.

The source is committed before shared-host execution. Formal outputs will be curated under `artifacts/v0.0.5`; final measured conclusions and exact reproduction commands will be linked here after completion. Development smoke tests use separate output directories and are never formal evidence. No competitive boxing/self-play is authorized by this motor wave.

## Resource amendment

The user explicitly requested increased host limits during S-A. [Resource addendum](../../docs/V0.0.5_RESOURCE_ADDENDUM.md), committed as `b3ec872` before H0, raises the aggregate worker allowance4→16. It retains the registered scientific trial settings and budgets. A standalone fixed-policy scaling job used4/8/12workers alongside the four-worker wave. Its nine samples cover14,400timed rollout transitions, zero invalids; medians220.48/436.01/646.79transitions/s. This excludes the optimizer and is shared-host timing. A16-worker standalone check follows after the formal wave exits if resource checks pass. Raw H0 descriptions/results are curated alongside the original wave without modifying its protocol or ledger.
