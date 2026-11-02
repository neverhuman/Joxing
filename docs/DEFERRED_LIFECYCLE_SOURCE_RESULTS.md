# H15 watchdog and deferred-lifecycle source controls

Two prerequisite packets concluded; native parity remains not_run. Both retain
the original8GiB/16process/one-thread limits and unchanged legacy default.

| Packet | Frozen source / protocol | Result |
|---|---|---|
| Independent fast RSS watcher |b468323 / 7c0f6778|20/20Linux cases pass,5.638537s whole|
| Opt-in deferred lifecycle source/fault controls |516f590 / a32da218|27/27Linux cases pass,18.834699s whole|

The watcher reads only bound PID/start RSS/status and children of owned threads.
It avoids PSS scans in its fast loop. The actual normal-exit/growing-grandchild
controls observed maximum21.36ms sample gap and1.23ms probe duration. Planned RSS
stop stayed below the registered hard threshold; the test-owned foreign process
survived. These are controlled tests, not a continuous memory bound or measured
watchdog behavior during native startup. Every future guarded run must satisfy
its own100ms maximum observation-gap/probe requirement.

Independent watcher audit verifies718source files/four history objects, all20JUnit
cases,12recorded identities exited/no writer, and35raw files/242919bytes transferred
exactly. Both raw watcher reports are included in its independent audit.

The new `deferred-evaluation-v1` keyword is opt-in. `wave.train_run` retains its
legacy default and exact frozen legacy AST body. The candidate takes metadata
from all four workers, freezes initial/final actors and training-end states,
closes workers, then allocates evaluation. Fault controls retain live states
before close, captured training-end states after close, and honest unavailable
states after partial construction. Cleanup errors retain the original fault.

All27candidate controls use fake constructors/learning, plus ownership/hash
fixtures. They establish allocation order and source/fault behavior; they do not
prove native trajectories, RNG, optimizer parity, memory saving, motor skill or
boxing. Independent audit verifies720source files/five history objects,27JUnit
cases,seven recorded exits/no writer and53raw files/250982bytes exact transfer.
No actual candidate actor, native environment, policy evaluation or training ran.

- [Watcher audit](../artifacts/v0.0.11/development/watchdog-controls-qualified-01-integrity.json),
  [watcher transfer](../artifacts/v0.0.11/development/watchdog-controls-qualified-01-transfer.json),
  [watcher protocol](../experiments/v0.0.11/watchdog-controls-protocol.json).
- [Candidate audit](../artifacts/v0.0.11/development/deferred-source-controls-qualified-01-integrity.json),
  [candidate transfer](../artifacts/v0.0.11/development/deferred-source-controls-qualified-01-transfer.json),
  [candidate protocol](../experiments/v0.0.11/deferred-source-controls-protocol.json).

Decision: prerequisites accepted within these scopes. No default adoption, full
source qualification or v11tag. [Next worker-ready native parity packet](DEFERRED_PARITY_EXECUTION_PACKET.md)
retains four sequential slots/both entropy arms and exact comparison gates before
eight-worker integration, then H10 native balance/contact recordings.
