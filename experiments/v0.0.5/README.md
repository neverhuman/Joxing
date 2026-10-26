# v0.0.5 experiment book

Read the [preregistered plan](../../docs/V0.0.5_PLAN.md) and [immutable machine protocol](protocol.json) first. The plan was committed as `09cbacb` before any wave outcomes. This iteration tests objective scaling and optimizer cost, preserving physical dynamics and the motor gate.

Every planned ID has its hypothesis and description in protocol.json. Execution writes a per-experiment description before work, an append-only event journal, raw initial/final evaluations (all attempts), progress, actor bundles and result receipts. Conditional replication remains explicitly not_run unless the registered selection rule passes. A negative screen is a useful completed experiment; incomplete training is reported separately.

The source is committed before shared-host execution. Formal outputs will be curated under `artifacts/v0.0.5`; final measured conclusions and exact reproduction commands will be linked here after completion. Development smoke tests use separate output directories and are never formal evidence. No competitive boxing/self-play is authorized by this motor wave.
