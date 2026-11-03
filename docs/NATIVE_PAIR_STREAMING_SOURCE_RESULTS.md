# Native motor evidence can now stream without duplicating observations

This is a source-development result within H31, not a new released-controller
training result. The earlier real-checkpoint32joint-step PPO witness and its 0–0
recordings remain the latest actual native performance evidence.

The new opt-in `CausalChunkWriter` stores each newest raw 1,970-feature frame,
bootstraps four-frame history at chunk boundaries, and reconstructs the full
7,901-feature actor observation exactly. Adjacent native integration endpoints are
stored once; real episode resets retain their own full initial state. Teacher
inputs/outputs, raw and applied actions, actual 700-muscle commands per body,
values/log probabilities, rewards and terminal masks all remain lossless.
Chunks use the already tested bounded ZIP_STORED format, at most 128 rows / 64 MiB.
A 3,072-byte per-row scalar ceiling and pre-write raw/text credits reject excess data.
This is an explicit zero-delay 20 ms profile; reaction/jitter remains future work.

The native paired VecEnv exposes raw histories only when `record_history=True`.
They are captured before terminal auto-reset. The default smoke runner and
recorder remain unchanged, and the frozen prototype checkout is untouched.
`StreamingNativeTransitions` connects the candidate writer to the PPO callback;
it does not accumulate the whole training tape in memory.

## Measured source results

- Final 55/55 paired motor/evidence checks pass, plus 39/39 history/rule/referee/
  stored-tape regressions. No skips or xfails. Ruff passes using its cached tool.
- The new native-engine fixture runs 16 joint transitions / 32 actor samples,
  two PPO updates and five 60 ms motor episodes, using a **synthetic teacher**.
  It exercises resets both inside chunks and at boundaries, including partial
  final chunks. It is not a released walking policy or a boxing-game result.
- Every reconstructed source/next observation, teacher input/output, applied/raw
  action, command, native before/after state, value/log probability, reward and
  terminal/truncation mask matches the original recorder byte-for-byte.
  Both before-update checkpoints independently recompute all 16 behavior rows with
  maximum value/logprob error 0. This is a source data comparison, not a fresh
  independent native physics replay or a replay speed measurement.
- Original transition archive 3,868,036 B; candidate **archive plus manifests**
  2,189,946 B: 56.6165% of original, **43.3835% fewer bytes**. Five-row chunks were
  intentionally small to exercise boundaries. No whole-pilot throughput or
  memory improvement is claimed from this fixture.
- Synthetic controls also cover 128-row chunks, partial final chunks, signedzero,
  dtype/shape, future/delayed/foreign samples, omitted history, wrong clocks,
  reset ownership, corrupt/truncated containers, resigned array faults,
  action clipping, text/raw credit failure and previous output/symlink refusal.

[Exact results, commands, source hashes, all development faults and raw hashes](../artifacts/v0.0.11/native-pair-streaming-source-01/summary.json).
The first 29-case run had 28 passes / 1 test-fixture failure: 300,000 B legitimately admitted
one row although the test expected rejection. The fixture was corrected to
200,000 B; production caps were not changed. Subsequent 52- and 55-case runs passed.
A receipt-collector relative-path error was fixed without native work. All
attempts remain in the small JUnit receipts. Static new source-physics count:
1,766 intervals across the two native source runs;0 new formal attempts or boxing games.

## Prospective training scope

[Worst-reset forecast02](../artifacts/v0.0.11/development/h31-stream-storage-forecast-02.json)
reserves every possible 20 ms reset, full checkpoints at 128-joint-step boundaries,
13 MiB/checkpoint + 1 MiB/continuation, and 128 MiB other raw / 8 MiB other text for source,
full fixed evaluations, replay, clips, faults and receipts. These are reservations,
not achieved write limits: all those remaining writers must enforce them before
any pilot can launch.

| Proposed scope | Raw upper bound | Text upper bound | Fits 1 GiB / 16 MiB? |
|---|---:|---:|---|
|2 seeds × 2,048 joint steps|1,249,335,552B|22,020,096B|No|
|2 seeds × 1,024 joint steps|706,456,704B|15,204,352B|Yes, subject to remaining guards|

The earlier forecast 01 used 64 MiB / 2 MiB other evidence allowances and is retained;
02 increases those *reservations*, keeping the raw/text hard ceilings unchanged.
No formal 2,048-step pilot was registered or started. Next implement and preregister
2 × 1,024 joint steps (4,096 actor samples / 40,960 training physics steps) with the same proposed
three-second / four-start / guard-reach-return behavior gates. A changed actor alone cannot pass
that behavioral experiment. Preserve falls and negative outcomes.

The fresh four-host inventory finds no owned Joxing simulations on xbabe1–4.
xbabe2 and xbabe3 have substantial shared I/O pressure; xbabe1 has about 16.7 GB available, while
xbabe4 Mac load is about 5.39. Local source development had no overlapping native job.
No host services, packages, drivers, unrelated jobs or upstream models changed.
Runtime remains the prototype's explicit local Mac profile, not Linux parity.
No new release tag; whole v0.0.11 and the larger H31behavioral pilot remain pending.

[Worker-ready next steps](NATIVE_PAIR_MOTOR_TRAINING_CONTINUATION.md).
