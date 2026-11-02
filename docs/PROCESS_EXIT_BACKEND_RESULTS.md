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

## Third packet: paired import-only backend hypothesis supported

H20publication prerequisite qualified separately; this NEW comparison uses its
named RAM-backed path/timing, not the H20smoke as a sample. Frozen source
`5a9be0484f3842cca2251d19f61c7b7e4b9e7c67`,
[packet](../experiments/v0.0.11/process-exit-backend-publication-protocol.json)
`aefeeb4b57dd8a26b8b7cec4f93ae1ec75d1f7f4bd921175fec3f8b0a019e438`.
Fresh immutable root `/home/ubuntu/joxing/process-exit-backend-publication-20261102-01`.

All85Linux controls pass, then all6fresh slots are VALID/0INVALID/0NOT_RUN.
Fixed order and all paired seeds/payloads/RNG/runtime contracts match exactly.
Close-send→join (seconds):

|Seed|Spawn|Forkserver|
|---|---:|---:|
|1097|0.527060417458415|0.04012689180672169|
|1098|0.48309944942593575|0.03975098207592964|
|1099|0.5999017674475908|0.04120119288563728|

Median0.527060417458415 vs0.04012689180672169s (13.13484× faster exit in this
import-only diagnostic). All three candidate pairs faster, candidate/reference
median0.07613338<=0.5. Preregistered verdict **supported_import_only**.
This is n=3, one host, tiny tensor/array payload, not a native/RL throughput gain.
Including TWO prior packets, H19full comparison denominator is8ATTEMPTED/
6VALID/2INVALID plus10NOT_RUN (18registered slots); no failed row is relabeled.
H20single smoke remains a separate prerequisite.

Every formal slot stays within100ms sample/probe/retirement and2s close. Max
formal gap25.129117ms/probe4.962541ms; maximum formal capacity517738496bytes/six
processes. Six payload workers actually expose positively charged terminal exit,
all reservations release within47.318965ms. Clean single-thread stdlib-only server
has empty preload and actual bound ancestry before imports; server stays clean
and all payload workers1thread after imports. Child cleanup phases include pipe
close/before_return/managed finalizer. Python atexit observed for all spawn,
absent for all forkserver as separately registered; no application fast exit.
No model/data/physics/actor/inference/update occurred, no physics ledger exists.
All seven real private scopes copied exactly then removed; no IPC/live writer.
Default remains spawn/disk; this is only the explicit diagnostic variant.

[Independent audit](../artifacts/v0.0.11/development/process-exit-backend-qualified-01-integrity.json)
verifies743source/five historical objects/full history,85exact JUnit cases,
raw before/after stat/status/maps, all six payloads/RNG/phase ordering/close times,
independent pair medians/verdict, twelve real watcher reports, seven IPC receipts
and56recorded exited identities/no writer. All348rawfiles2330850bytes exact
transfer; archive SHA `5da0b0d0d398a8670b37e464b857d7e0ccad742b455307ddf35ec507f60cbea8`.
[Transfer](../artifacts/v0.0.11/development/process-exit-backend-qualified-01-transfer.json),
[parent receipts](../artifacts/v0.0.11/process-exit-backend-qualified-01/),
[deployment/source/capacity/command](../artifacts/v0.0.11/development/backend-publication-deployment-01.json).
Prelaunch load8.11/7.18/7.47,33.74GB available RAM/152.33GB disk, no other ownedjob;
anonymous public byte verification BEFORE launch. Runtime and all inputs match.
Auditor completes without a repair in this packet. Reproduce:
`rtk proxy python3 artifacts/v0.0.11/development/audit-backend-publication-01.py experiments/v0.0.11/process-exit-backend-publication-protocol.json FRESH_LABEL`.
Raw remains in ignored `runs/v0.0.11/process-exit-backend-qualified-01-evidence`.

Packet42.7743978574872s/1678701fixture/652149receipt bytes. Twelve-packet carry
16458physics/336.20004829830765s/34554362fixture/3641782receipt bytes; original
remainder223542steps/263.79995170169235s/1039187462fixture/552522receipt bytes.
Receipt budget is tight; no failed cap raised or previous bytes removed.

Decision: candidate backend merits ONE bounded native constructor/lifetime test,
not default adoption, four-worker teardown, paired-learning parity, native balance
or boxing qualification. [H21worker-ready mission](NATIVE_CONTEXT_TRANSFER_MISSION.md)
keeps all700muscles, same200prep/frozen state and H18release contract; new machine
protocol/source must be frozen before outcomes. H15three invalid native learning
attempts remain; no candidate learning parity or v11tag.
