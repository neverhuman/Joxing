# H17 fails its two-second close deadline

Concluded2026-11-02. Opt-in serial transport controls pass; its four-worker native
constructor gate fails. This is not a training, capacity, balance or boxing win.

Frozen source `62b361bf3ec2086029af0e4537d24ddbb67b1c34`,
[protocol](../experiments/v0.0.11/serial-native-teardown-protocol.json), canonical
SHA256 `27dea5bd160a8a781001576978359d84b605d534da583c797bf13e2d04d1ff13`.
Dedicated xbabe1 root `/home/ubuntu/joxing/serial-native-teardown-20261102-01`.
[Deployment receipt](../artifacts/v0.0.11/development/serial-native-teardown-deployment-01.json)
retains anonymous public verification, capacity, hashes and exact commands.
All75 registered Linux controls passed with no skip, including actual four512MiB
serial heaps, positive retiring charges, foreign survival and old-default controls.
Earlier local checks/style faults are separately retained, not counted as Linux
qualification. Installed SB3 source hash matched the preregistered value.

## Fault chronology and complete denominator

One native constructor attempt: **1invalid,0valid**. All four workers retain700
muscles in actual reset traces. Exactly800preparation steps/0rollout,8credit events,
no pending reservation; no actor or optimizer constructed. Native learning remains
not_run, and the previous three invalid reference/nine not_run parity slots stay.

The FIRST recorded fault was `TimeoutError: Serial worker join deadline` after
2.001146459951997s from the first close send. Worker0 exited normally after
0.8505526464432478s, worker1 after0.9212638568133116s. Worker2 was sent close with
only the remaining budget; it did not join within it. Worker3 received no normal
close send. VectorOwner then terminated remaining owned processes as fault
cleanup. The reader subsequently faulted on a105.905134ms reserved interval at
release detection for worker3. That later error is the supervisor's headline;
it must not hide the earlier serial deadline failure.

Normal first-two-worker retirement reservations released at74.29/75.97ms. Cleanup
worker2 released at81.20ms; worker3 exceeded100ms. This does **not** establish that
ordinary serial exit alone violated the hold, nor that concurrent teardown was
the cause of earlier failures. It establishes that this declared serial lifecycle
does not meet its fixed overall close deadline. Do not enlarge either deadline
or repeat the unchanged packet to get a passing run.

Sample/probe maxima38.82/18.42ms stayed below100ms. Peak sampled capacity charge
4622938112bytes; peak retirement reservation2137624576bytes. No sampled RSS or
gap breach recorded; continuous memory is not bounded by these samples.

The first three wrapper close hashes equal
`8f65c491a4de3fd9e921f88483d3976709c8e2e2bf21679508874a36faf9bfff`.
Worker3 has no normal close trace. Parent initial hashes were computed in memory
but not persisted before shutdown; no constructor-result completed. Therefore
an independent four-way initial/close equality verdict is **unavailable**, even
though the first source checks ran. Do not repair this gap using an older run's
matching hash. Next runner must persist its before-close receipt first.

## Independent audit and accounting

[Audit](../artifacts/v0.0.11/development/serial-native-teardown-failed-01-integrity.json)
verifies735source files, five historical objects/full history, all75JUnit cases,
installed runtime file and input hashes,8credit events, four control-watch reports
and the failed native watch, all bound retirement proofs/charge arithmetic.
All34 recorded PID/start identities exited, no writer/source diff remains.
All446rawfiles/502591bytes transferred exactly; archive SHA256
`aa2e849ca5746b53b003ec66b2f02765087f667632c21936c44aae822b09f43d`.
[Chronology audit](../artifacts/v0.0.11/development/serial-native-teardown-failed-01-chronology.json)
independently checks every recorded close send/join/fault, worker identity and
partial native/close denominator. [Small parent receipts](../artifacts/v0.0.11/serial-native-teardown-failed-01/).

Exact supervisor command is recorded in deployment/result receipts:
`python3 scripts/qualify_deferred_trial.py --packet experiments/v0.0.11/serial-native-teardown-protocol.json --protocol-sha256 27dea5bd160a8a781001576978359d84b605d534da583c797bf13e2d04d1ff13 --source-sha 62b361bf3ec2086029af0e4537d24ddbb67b1c34`.
Independent audit command:
`rtk proxy python3 artifacts/v0.0.11/development/audit-serial-native-teardown-01.py experiments/v0.0.11/serial-native-teardown-protocol.json serial-native-teardown-failed-01`.
The original execution used the byte-identical helper under ignored runs; audit
and transfer outputs already exist, so reruns need a fresh output label.

This packet32.60769325681031whole seconds/204650fixture/297941receipt bytes.
All seven tracked packets:16258physics/211.86220482364297s/31955319fixture/
1872562receipt bytes. Original remainder223742steps/388.137795176357s/
1041786505fixture/2321742receipt bytes. No new physics during independent audit.

## Decision

Decline serial-close-v1 as a sufficient native prerequisite. Keep default
transport unchanged and preserve the qualified controls separately. No further
paired learning until a new close lifecycle qualifies. [H18mission](LIVE_RESOURCE_RELEASE_MISSION.md)
tests explicit graph release while the worker is still alive, starting with
ONE200-step constructor and persisted pre-close evidence. Native StanceEnv has
no close override; merely calling inherited close does not explicitly discard
its model/data graph. Whether graph release reduces exit time/RSS remains an
unmeasured hypothesis. Do not change model physics, use forced fast exit or
claim learned boxing from this infrastructure. v11 stays incomplete/untagged.
