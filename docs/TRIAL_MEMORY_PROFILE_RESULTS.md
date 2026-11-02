# H15 constructor allocation profile

Two attempts, both preserved. No training gain, memory optimization or complete
source qualification. Next: [deferred allocation parity](DEFERRED_EVALUATION_ALLOCATION_MISSION.md).

| Attempt | Frozen source / protocol | Outcome |
|---|---|---|
| First | efa4d41 / 409cc901 |24Linux controls pass; ownership inventory race stops profile before allocation events. Whole13.021967s. Two learner processes launched, initialization unconfirmed.|
| Ownership repair |7204702 / c617e6cd |28Linux controls pass. Capacity-limited constructor profile; both learners unconfirmed,2/10world constructors complete, neither SAC/learn-ready reached. Whole28.177220s.|

The second source only repairs binding of newborn PID/start identities through
actual supervisor ancestry; original model, trial settings and caps unchanged.
Original two learners each attempt four native training worlds and one evaluation
world; all700muscles remain. Policy inference, optimizer calls and env.step are
guarded. Native reset preparation is separate from training.

| Learner | RSS before evaluation constructor | RSS after | Observed delta |
|---|---:|---:|---:|
| A |397,631,488|1,013,399,552|615,768,064bytes|
| B |397,533,184|1,005,113,344|607,580,160bytes|
| Combined delta |||1,223,348,224bytes|

These are same-process boundary measurements, including model compilation,
preparation and retained allocator memory. They do not prove an equal recoverable
saving. Avoiding early evaluation construction is now a measured candidate.

The planned7.5GiB sampled stop was crossed during startup. Sampled aggregate RSS
reached8,536,518,656bytes, below the8,589,934,592hard limit by53,415,936bytes.
Fourteen owned processes were sampled, including guardian/manager/tracker; no
worker count was reduced. PSS at that sample was7,043,260,416bytes, explanatory
only.29raw samples independently sum exactly. Requested0.2s sampling produced
median0.473/max1.198s gaps, with a3,157,102,592byte largest RSS jump. No continuous
memory bound is demonstrated; a separate fast watchdog is a prerequisite next.

Two evaluation constructors each report200reset-preparation substeps, for a
known400-step lower bound. Eight world constructors were interrupted; their
preparation totals are unknown. Normal fixture-step lower bound0 is not a claim
of zero overall physics. No SAC actor was constructed or evaluated, and no
formal training run or boxing match occurred.

The profile orchestrator is stdlib and used26,537,984bytes at peak. The original
Torch-importing pytest parent is larger. This diagnostic therefore does not
establish that the unchanged parallel fixture fits its8GiB limit. Candidate
training/evaluation RNG parity, completed eight-worker startup and full-source
qualification remain open.

Independent audit verifies716source files, full history, all28JUnit identities,
bound raw samples, all19recorded identities exited/no writer.35raw files totaling
317,405bytes transferred byte-for-byte; fixture4728bytes/receipts312,677bytes.
Raw profiles remain outside Git; small result/audit summaries and hashes remain.
The earlier ownership stop separately verifies716files/11exits/20rawfiles229000bytes.

- [Second attempt](../artifacts/v0.0.11/trial-memory-capacity-limited-02/),
  [independent audit](../artifacts/v0.0.11/development/trial-memory-capacity-limited-02-integrity.json),
  [transfer](../artifacts/v0.0.11/development/trial-memory-capacity-limited-02-transfer.json).
- [First attempt](../artifacts/v0.0.11/trial-memory-ownership-failed-01/),
  [first audit](../artifacts/v0.0.11/development/trial-memory-ownership-failed-01-integrity.json).
- [Original protocol](../experiments/v0.0.11/trial-memory-profile-protocol.json),
  [ownership repair protocol](../experiments/v0.0.11/trial-memory-ownership-repair-protocol.json),
  [research memory](../research/H15.md).

Decision: keep ancestry repair; use allocation attribution to design the named
deferred lifecycle. Do not increase caps or claim balance/boxing/self-play progress.
