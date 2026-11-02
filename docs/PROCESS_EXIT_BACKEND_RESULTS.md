# H19 first packet: clean-parent gate catches receipt-helper imports

Concluded first attempt2026-11-02; scientific exit comparison remains unmeasured.
Frozen source `7ee2bf3e0174f7be83bd59fac3c4ef3c51f6a1eb`,
[original protocol](../experiments/v0.0.11/process-exit-backend-protocol.json),
canonical `c83d92a42d5233c3da7407f6d79e677c4972957a86e11341cbf482def29a57a5`.
Immutable xbabe1 root `/home/ubuntu/joxing/process-exit-backend-20261102-01`.

All69Linux source/monitor controls pass. The first spawn1097root exits nonzero
before creating any payload worker: run() imported shared provenance.write_json,
whose module imports NumPy and MuJoCo. The clean-parent assertion then raises
`ValueError: Invalid probe root contract/imports`. The module-before-readiness
control passed, but missed this later parent-preparation path. This is our harness
error, not a measured spawn/forkserver failure or a latency result.

Formal rows:1INVALID/0VALID/5NOT_RUN. Payload workers0, physics/actors/updates0;
no payload/server proof/result or native physical outcome exists to recycle.
The guard completed and all recorded processes exited. The original code/root/
protocol/failed criteria are retained unchanged.

[Independent audit](../artifacts/v0.0.11/development/process-exit-backend-failed-01-integrity.json)
checks740source files, five historical objects/full history,69exact JUnit cases,
runtime and input hashes,23recorded exited identities/no writer. All153rawfiles/
311871bytes transferred exactly, archive SHA256
`1606f231ade7792b98e0d6f804110ff0f745782294ce0119323db11274ac0e7f`.
[Transfer](../artifacts/v0.0.11/development/process-exit-backend-failed-01-transfer.json),
[parent receipts](../artifacts/v0.0.11/process-exit-backend-failed-01/),
[deployment](../artifacts/v0.0.11/development/process-exit-backend-deployment-01.json).
Exact supervisor command is preserved in deployment/result: python3
scripts/qualify_deferred_trial.py with original packet, above protocol/source hashes.
Audit command was `rtk proxy python3 runs/v0.0.11/audit_process_exit_backend_01.py experiments/v0.0.11/process-exit-backend-protocol.json process-exit-backend-failed-01`.

Packet12.850743534043431s/50372fixture/261499receipt bytes,0physics. Nine-packet
carry including the prior toy charge:16458steps/258.5835353403473s/
32224454fixture/2434240receipt bytes. Original remainder223542steps/
341.4164646596527s/1041517370fixture/1760064receipt bytes.

Named stdlib-receipt-repair-v2 replaces only that helper with a local atomic
stdlib writer and adds fresh subprocess controls for both parent-preparation
paths after readiness. Local67pass/4Linux skips/Ruffpass. [New protocol](../experiments/v0.0.11/process-exit-backend-repair-protocol.json),
canonical `f94d81546acc3471dc5e6b06ba72269417b1fa74820cbb426c6eb60d3b3cecf9`,
requires71Linux controls before the identical six fresh probes. Same seeds,
payloads, order, resource/time limits, scientific criterion and old costs carried;
no gate loosened, default changed or failed row reused. Repair outcomes not_run
atfreeze; publish/anonymousverify/freshcapacity precede a dedicated new root.

## Second packet: parent repair works; monitor freshness fails

Frozen `1726ce2bb058ab033969626d55e10d1f180e3c9a`, repair protocolf94d8154,
dedicated `/home/ubuntu/joxing/process-exit-backend-repair-20261102-01`.
All71Linux controls pass. Spawn1097root proves clean single-thread parent/worker
and the payload worker reaches waiting_release then import_begin. It never
reaches payload-ready, close or a latency result. The watcher correctly stops
the job on a120.578872ms sampling gap>registered100ms. Maximum actual probe
duration1.706129ms; sampled peak379719680bytes/five processes, no retirement
records. This is not a memory-cap breach, an exit-time result or proof of any
backend benefit. One partial payload worker,0physics/actor/updates.

Formal second rows1INVALID/0VALID/5NOT_RUN; combined two packets2INVALID/0VALID/
10NOT_RUN. No failed row is recycled. The repaired parent-import path passes;
monitor gap failure remains failed, no limit changed. The interval before the
stale sample follows a quarter-second publication point in the synchronous
watcher save path. Because save/sleep/scheduling durations were not separately
recorded, **I/O attribution is a hypothesis**. Post-failure pressure snapshots
are not measurements during that gap and cannot prove its cause.

[Second independent audit](../artifacts/v0.0.11/development/process-exit-backend-failed-02-integrity.json)
verifies740source files,71JUnit cases, raw bound pre-import proc evidence and
25recorded exited identities/no writer; all165rawfiles368009bytes transferred
exactly, archive SHA256 `09378431ad6a4b9906922087a0708329b09c0a7c1b42fa95677a5efd34992403`.
[Transfer](../artifacts/v0.0.11/development/process-exit-backend-failed-02-transfer.json),
[parent receipts](../artifacts/v0.0.11/process-exit-backend-failed-02/),
[deployment](../artifacts/v0.0.11/development/process-exit-backend-repair-deployment-01.json),
[read-only post-observation](../artifacts/v0.0.11/development/process-exit-backend-failed-02-post-observation.json).
No additional physics/audit benchmark ran. Reproducible auditor is
artifacts/v0.0.11/development/audit-process-exit-backend-01.py; pass the appropriate
packet and a fresh output label because retained audit/transfer outputs exist.

Packet14.49564023502171s/87433fixture/280576receipt bytes. Ten-packet cumulative
including the toy charge:16458steps/273.079175575369s/32311887fixture/
2714816receipt bytes; original remainder223542steps/326.920824424631s/
1041429937fixture/1479488receipt bytes.

Decision: stop backend repeats until [H20monitor publication qualification](MONITOR_PUBLICATION_MISSION.md).
Keep the100ms bound and add phase timing to distinguish probing, persistence and
wake delays. Test a bounded private memory-backed report path with exact disk
copy after the task exits; disk remains default. This changes diagnostic I/O,
not numerical trajectories, caps or scientific acceptance. Native/model/backend
adoption, eight-world/full-source qualification and boxing remain blocked.
