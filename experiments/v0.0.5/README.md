# v0.0.5 experiment book

Read the [preregistered plan](../../docs/V0.0.5_PLAN.md) and [immutable machine protocol](protocol.json) first. The plan was committed as `09cbacb` before any wave outcomes. This iteration tests objective scaling and optimizer cost, preserving physical dynamics and the motor gate.

Every planned ID has its hypothesis and description in protocol.json. Execution writes a per-experiment description before work, an append-only event journal, raw initial/final evaluations (all attempts), progress, actor bundles and result receipts. Conditional replication remains explicitly not_run unless the registered selection rule passes. A negative screen is a useful completed experiment; incomplete training is reported separately.

The source is committed before shared-host execution. Formal outputs will be curated under `artifacts/v0.0.5`; final measured conclusions and exact reproduction commands will be linked here after completion. Development smoke tests use separate output directories and are never formal evidence. No competitive boxing/self-play is authorized by this motor wave.

## Resource amendment

The user explicitly requested increased host limits during S-A. [Resource addendum](../../docs/V0.0.5_RESOURCE_ADDENDUM.md), committed as `b3ec872` before H0, raises the aggregate worker allowance4→16. It retains the registered scientific trial settings and budgets. A standalone fixed-policy scaling job used4/8/12workers alongside the four-worker wave. Its nine samples cover14,400timed rollout transitions, zero invalids; medians220.48/436.01/646.79transitions/s. This excludes the optimizer and is shared-host timing. A16-worker standalone check follows after the formal wave exits if resource checks pass. Raw H0 descriptions/results are curated alongside the original wave without modifying its protocol or ledger.

## Completed original wave

Source `6ac2bad`: all four arms completed40,000transitions/9,500updates each;336evaluation attempts including baseline trials, zero invalids. Initial mean1.017750s; final A/B/C/D means1.011563/1.090938/0.802313/1.120156s. D won the screen but its0.102406s gain missed the0.25s gate. Four conditional confirmation slots remain not_run. Full raw rows, actor bundles, progress and every description are under `artifacts/v0.0.5`; Linux accounting/continuation pass, Mac accounting/repeated restore pass, strict cross-host closed-loop parity fails. Never discard the failed check or call these small gains boxing competence.

## Authorized longer-budget extension

The user explicitly answered “Increase workers and training budgets.” [Budget addendum](../../docs/V0.0.5_BUDGET_ADDENDUM.md) and [separate protocol](extension-protocol.json), committed as `76f6e5c` before execution, retain all four conditions with160,000transitions/2,700seconds each. Four independent trials run concurrently, four simulation workers each. Seed547 and starts64000–64031 form a new development screen. Conditional confirmation uses seeds761/983 and unopened starts65000–65031. Criteria are unchanged; reused diagnostic rows are excluded from new-attempt counts. This changes seed/budget/split and is not replication of the first screen. The screen completed 640,000 transitions and 288 new evaluations, zero invalids. B improved 1.059063 → 1.313469 seconds and qualified for confirmation; no screen policy reached five seconds. [The frozen screen receipt](../../artifacts/v0.0.5/extension/screen-freeze.json) records selection before confirmation outcomes. Matched A/B jobs for seeds 761/983 are running. No v0.0.5 release tag has been assigned yet.


## Final extension and release conclusion

All four confirmation jobs completed 160,000 transitions  / 39,500 updates each, with all 256 paired policy evaluations valid. Baseline confirmation adds 32 new attempts. Full extension: 1,280,000 transitions, 576 new evaluations, zero invalids; original+extension: 1,440,000 transitions, 912 new evaluations, 14,736 completed stochastic motor episodes. Eight reused diagnostic rows were excluded from new attempts. Both waves preserve every planned experiment and not_run decision.

| Confirmation seed | Initial mean (s) | Final A (s) | Final selected B (s) | B minus A, 95% reset CI |
|---|---:|---:|---:|---|
| 761 |1.022875|0.803000|1.293531|+0.490531 [0.452125,0.532286]|
| 983 |1.010531|1.165406|1.097188|−0.068219 [−0.096314,−0.040187]|

B's lower reset bound versus its initial policy is positive on both seeds, but versus A is negative on 983. The registered replicated-superiority criterion fails. Every final policy has 0/32 five-second survivors; no formal motor proof or self-play ran. Source remained unchanged; all owned jobs exited. Linux independent audits pass; Mac raw accounting/repeated restore pass while strict cross-host closed-loop state parity fails. Frozen constant confirmation evaluation occurred before learner training, unlike the broad unopened-bank wording; learner-policy evaluation was deferred. This was disclosed before outcomes in `eba0959`; do not call the bank completely sealed.

[Final results/interpretation](../../docs/V0.0.5_RESULTS.md) · [Machine-readable release summary](../../artifacts/v0.0.5/release-summary.json) · [Extension registry/descriptions/raw rows](../../artifacts/v0.0.5/extension/) · [Confirmation figure](../../artifacts/v0.0.5/extension/wave-confirmation.png) · [Source-host replay](../../artifacts/v0.0.5/extension/wave-before-after-linux.mp4). The earlier entries above are chronological records; their pending statuses are superseded by this completion entry. v0.0.5 is the completed release.

The user-requested state/history review is in [the code audit](../../docs/STATE_HISTORY_SELF_PLAY_AUDIT.md), with isolated actor timing, physical replay cost, command variation and full missing-game contracts. [The next teacher-transfer plan](../../docs/UPSTREAM_MOTOR_TRANSFER_PLAN.md) retains inspected release hashes and a possible decoder-load defect; no upstream teacher rollout is claimed. The small [diagnostic accounting correction](../../artifacts/v0.0.5/diagnostic-accounting-correction.json) preserves published raw counters and charges omitted constructor preparation separately.
