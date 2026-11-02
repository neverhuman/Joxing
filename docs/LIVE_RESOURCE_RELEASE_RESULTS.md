# H18 qualifies one-worker native resource release

Concluded2026-11-02. **A closed-world lifetime prerequisite passes.** This does
not qualify four-worker shutdown, training parity, eight-world capacity, balance
or boxing. H17's failed two-second serial shutdown remains failed.

Frozen source `e3d1272b4fb2c693f82440e81ca6c16877c06a91`,
[protocol](../experiments/v0.0.11/live-resource-release-protocol.json), canonical
SHA256 `94750277896099fff234bdfbc3f34455e3d027bdca7f59a72c0b4a3edbd173b5`.
Immutable xbabe1 root `/home/ubuntu/joxing/live-resource-release-20261102-01`.
[Deployment](../artifacts/v0.0.11/development/live-resource-release-deployment-01.json)
retains source/public/capacity verification, PID/start and exact commands.

## Registered outcome

All93 exact Linux controls passed,0skip. ONE native constructor **1valid,
0invalid,0not_run**: all700muscles,200exact preparation steps,0rollout/actor/
policy inference/optimizer updates. The initial parent receipt was written before
close. Parent initial, child before/after ordinary close and observed worker-close
state hashes all equal
`8f65c491a4de3fd9e921f88483d3976709c8e2e2bf21679508874a36faf9bfff`.
The actual model/action/observation hashes match the pinned specification;
normalizer none. No pose/model/rules/default change or application fast exit.

The worker's env, MjData and MjModel weak references all died after the closed
inner graph was discarded and GC collected29objects. Same live PID/start,
non-exiting flags and readable VmRSS verified before/after release. Observed RSS
fell from1018327040 to927662080bytes, a90664960byte decrease (~91MB). This does
not identify the remaining resident pages or establish a general memory saving.

Graph release phase144.64ms; close-send-to-join797.80ms; released-to-join tail
649.66ms. The native terminal reservation was actually observed, held positive
1102770176bytes and released96.47ms. The additional owned parent reservation
released21.08ms. All observed holds meet unchanged100ms; the high-water reservation
remained conservative after live RSS fell. Max sample/probe31.58/11.25ms,
peak sampled capacity1563746304bytes, no pending credits/retirements or guard faults.
Sampling bounds are not continuous physical-memory guarantees; the reserved
interval includes polling delay and is not exact kernel teardown duration.

The experiment meets its preregistered one-worker lifetime/resource criteria.
It has no controlled original/candidate speedup comparison. Comparing0.798s here
with0.851/0.921s in H17 cannot prove a gain. A serial four-worker extrapolation
is still around3.2s versus the fixed2s budget, an estimate rather than a new run.
Do not repeat the unchanged serial topology merely to seek a passing result.

## Independent audit and retained faults

[Audit](../artifacts/v0.0.11/development/live-resource-release-qualified-01-integrity.json)
verifies738source files, five historical executable objects/full history, exact
93JUnit denominator, installed Gym/SB3 sources, input header,200credit steps/
2events/no pending, all parent/child state hashes, graph lifetime/live RSS,
phase/close chronology and five watcher reports. All26 recorded PID/start
identities exited, no writer/source diff remains; vendor pins separately reread.
All479rawfiles/518942bytes transferred with exact names/sizes/hashes; archive
SHA256 `1d4532821bf74a526a3e1d60a2713c0fb072a72c3bef925b5bac0f3f299e3c5b`.
[Transfer](../artifacts/v0.0.11/development/live-resource-release-qualified-01-transfer.json),
[small receipts](../artifacts/v0.0.11/live-resource-release-qualified-01/).

The independent auditor initially failed a too-strict exclusivity assertion:
it expected only the native worker in retirement records, but the owned parent
also retired normally. The registered criterion requires the worker's positive
record, not absence of other owned records. The auditor was corrected to require
that worker and still check every record's bound proof/release/100ms limits.
[Audit-development fault/repair](../artifacts/v0.0.11/development/live-resource-release-audit-development-01.json).
No executed source/protocol/scientific gate changed, and no physics reran.

Exact supervisor command in deployment/result:
`python3 scripts/qualify_deferred_trial.py --packet experiments/v0.0.11/live-resource-release-protocol.json --protocol-sha256 94750277896099fff234bdfbc3f34455e3d027bdca7f59a72c0b4a3edbd173b5 --source-sha e3d1272b4fb2c693f82440e81ca6c16877c06a91`.
Independent audit:
`rtk proxy python3 artifacts/v0.0.11/development/audit-live-resource-release-01.py experiments/v0.0.11/live-resource-release-protocol.json live-resource-release-qualified-01`.
Original execution used its byte-identical ignored runs helper; outputs already
exist, so a re-audit/transfer needs a fresh label. Arrays/raw fixtures remain out
of Git. Local89pass/4Linux-only skips and toy binding check are development only.

## Budget and next measured bottleneck

Packet32.92965106666088s/218763fixture/300179receipt bytes. Eight tracked packets
plus the conservatively charged0.940935916s development command total:
16458physics/245.73279180630385s/32174082fixture/2172741receipt bytes. Original
remainder223542steps/354.26720819369615s/1041567742fixture/2021563receipt bytes.
Prior H15parity3invalid/0valid/9not_run and H17constructor1invalid remain separate.

The remaining649.66ms tail warrants a **zero-physics process-exit diagnostic**
before additional native constructors. Installed Python3.12.3 source shows spawn
uses sys.exit after bootstrap, while forkserver's child uses os._exit after its
managed worker path. This is a candidate mechanism, not proof of the measured
delay. Source hashes/excerpts and measurements are retained in the
[post-observation receipt](../artifacts/v0.0.11/development/live-resource-release-qualified-01-post-observation.json).
Python's [start-method documentation](https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods)
also requires attention to thread safety, context-specific IPC and resource
tracking. A different exit path changes cleanup semantics; it is not a silent
substitution for the reference backend.

[H19's worker-ready mission](PROCESS_EXIT_BACKEND_MISSION.md) proposes three small
paired import-only spawn/forkserver probes, same payload/seeds,0physics/learning,
clean single-thread server and all owned exits. No backend is adopted. Only a
measured supported result may motivate a separately frozen native/cleanup/state/
RNG/replay parity gate. Native walking remains distinct from stable boxing.
