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
