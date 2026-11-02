# Four-worker teardown exceeds the registered reservation hold

Concluded2026-11-02. This is an infrastructure failure, not a boxing result.
The first two failed native packets and all original criteria remain unchanged.

## Registered attempt and denominator

Frozen source `e1a85212858d5b82656a7df133fc4a078fd670b0`,
[protocol](../experiments/v0.0.11/deferred-native-terminal-v3-protocol.json),
canonical protocol SHA256
`ca5ee668d4df04ea68b6745a82a1cca2fe791b3b98d77fccf89209dd87679706`.
Dedicated immutable xbabe1 root
`/home/ubuntu/joxing/deferred-native-terminal-v3-20261102-01`.
Source/runtime/input/header/history and anonymous public-file verification preceded
launch; capacity and ownership receipts are retained in the
[deployment receipt](../artifacts/v0.0.11/development/deferred-native-terminal-v3-deployment-01.json).

The23 registered Linux controls passed. LegacyA performed64real transitions,
12optimizer updates and5086exact physical steps:2000preparation,640training,
2446evaluation. The final actor changed, but shutdown failed before the semantic
result completed. The formal rows are **1invalid,0valid,3not_run**. Across all
three native packets: **3invalid,0valid,9not_run**;36updates occurred only in
invalid reference prefixes. Candidate learning, paired parity and memory saving
remain unmeasured. No failed baseline is reused as a valid result.

## What failed

Opt-in `terminal-mm-reserve-v3` retained positive high-water RSS charges for four
actual workers with matching PID/start, one exiting task, absent VmRSS/VmSize
and seven explicitly zero statm fields. This confirms the previously synthetic
exit mechanism also occurs in these actual native workers.

One reservation released after94.30ms. Another reached an observed reserved
interval of122.75ms at release detection, exceeding the registered100ms hold.
The watcher faulted, retained pending charges and stopped the trial. This is a
sampled reservation interval; it is not an exact measurement of kernel teardown
duration. The last three pending records reflect the fault-time reader state;
the independent later audit confirmed every recorded process had exited.

Peak sampled capacity charge was5165219840bytes, including separately labelled
RSS reservations; peak simultaneous retiring charge3203399680bytes. Maximum
sample gap46.54ms/probe26.32ms stayed below100ms. Thus the recorded stop concerns
retirement hold, not a recorded RSS or monitor-gap breach. Sampling cannot prove
a continuous physical memory bound. The100ms hold, RSS metric,7GiB planned/
8GiB hard limit and default strict reader were not relaxed.

Read-only inspection of the pinned SB3 2.7.1 installed source shows `close()`
sends all worker close messages before joining any process. That makes overlapping
teardown a plausible bottleneck, not a demonstrated cause of this fault. The
one-worker constructor previously qualified at70.99ms; comparing different runs
does not establish a causal speedup. [H17's next mission](SERIAL_RETIREMENT_MISSION.md)
tests a named serial-close alternative without another learning sweep.

## Independent audit and retained evidence

[Independent audit](../artifacts/v0.0.11/development/deferred-native-watch-failed-03-integrity.json)
verified731 source files, five historical executable objects, complete superproject
history, all23 exact JUnit cases,640physics-credit events, terminal proofs and
capacity arithmetic. Used credit5086/pending0. All18 recorded PID/start identities
exited, no writer remained, and source diff was empty. All306rawfiles/10854269bytes
transferred with matching names, sizes and hashes; archive SHA256
`86efdd615f3781dd1ff637d7752bfc526a8d10a0dab67dbf232cb19af857efef`.
[Transfer receipt](../artifacts/v0.0.11/development/deferred-native-watch-failed-03-transfer.json)
and [small parent receipts](../artifacts/v0.0.11/deferred-native-watch-failed-03/)
are checked in. Arrays, actor archives and full raw files remain outside Git.

A separate read-only audit loaded both saved actors on CPU, without inference,
physics or updates, and independently hashed sorted tensor names/shapes/dtypes/
bytes. Initial hash `8c5786c4cbed1667a814adef088719df6b8d0290a53fd85b2a07bc744022790f`;
final hash `84184743d0c7bf49139eed773bd67d0943409c0f5f4de5a7191afc64e6d0b486`.
All10 semantic histories match exactly across the three invalid reference
prefixes:31learner,249native-evaluation and18rows each for four wrapper and
four native-world files. This retrospective consistency is not formal parity.
[Read-only comparison receipt](../artifacts/v0.0.11/development/deferred-invalid-prefix-comparison-02.json).

Exact formal command is in result.json/deployment receipt:
`python3 scripts/qualify_deferred_trial.py --packet experiments/v0.0.11/deferred-native-terminal-v3-protocol.json --protocol-sha256 ca5ee668d4df04ea68b6745a82a1cca2fe791b3b98d77fccf89209dd87679706 --source-sha e1a85212858d5b82656a7df133fc4a078fd670b0`.
Local independent audit/transfer command was
`rtk proxy python3 runs/v0.0.11/audit_native_teardown_v3_01.py experiments/v0.0.11/deferred-native-terminal-v3-protocol.json deferred-native-watch-failed-03`.
That development helper is ignored; the retained integrity receipt exposes all
checked source/raw hashes, credit events' summary and actual identities. No
additional physics ran during the independent audit.

## Budget and decision

This packet consumed47.68762979283929whole seconds,10565687fixture bytes and
288582receipt bytes. Including all six tracked native/diagnostic/control packets:
15458physics,179.25451156683266active seconds,31750669fixture and1574621receipt
bytes. Remaining original envelope:224542steps,420.74548843316734s,
1041991155fixture bytes and2619683receipt bytes. Keep these cumulative costs;
do not reset the budget for the next variant.

Keep the qualified reader's narrow single-worker result and the failed four-worker
outcome separately. Do not repeat unchanged learning or increase the hold to make
this pass. Implement/freeze/test H17's opt-in serialized retirement, then register
one four-worker constructor-only test. Native balance, competitive learning,
eight-worker/full-source qualification and v11completion remain blocked.
