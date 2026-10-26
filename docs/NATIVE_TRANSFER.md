# Native motor transfer contract

This is a frozen single-athlete locomotion task, separate from the production boxing game. v0.0.6 qualifies the upstream controller; v0.0.7 adds explicit equipment/timestep variants and a longer diagnostic. Formal outcomes belong in the experiment book and results, never inferred from this design document.

Read `src/joxing/native_transfer.py`, `src/joxing/upstream_motor.py` and `tests/test_native_transfer.py` alongside the immutable protocol. `ReferenceBodyMap` stores canonical trajectory indices separately from actual variant body IDs. Both `mapped_observation()` and `MappedEnv.step()` use that map. Adding two glove child bodies can shift later native body IDs; indexing a saved reference array with those IDs selects the wrong target even if observation dimensions still match.

The adapter preserves 3,601 feature values in upstream order. Actual keypoints use variant IDs; target keypoints use canonical indices. New gloves do not require invented reference features. The native no-equipment oracle compares every feature block, full integration state, rewards and termination against the unmodified environment. Source-host launch additionally continues the actual v6 first-seed snapshot and commands exactly. Glove variants retain all native joint/actuator/tendon/equality ordering, 700 muscles and native body inertias; separate glove bodies add exactly 0.566kg. MjSpec's compiled model is passed directly to Gym to avoid parameter rounding from an XML pretty-print/reparse path. Generated XML is a receipt; the canonical assets and versioned builder reproduce the measured model.

Physics is 2ms or 1ms, with 10 or 20 substeps per fixed 20ms controller action. The frozen DynSyn decoder amplitude is explicitly 0.075 and normalization statistics are immutable. No optimizer, action smoothing, external support or per-step pose writes are admitted. The reference trajectory's current/future self-targets are known task goals. They are not temporal opponent memory or future opponent state.

Snapshots bind variant/model/reference mapping/observation hashes and retain the actual cached last actor observation, engine integration state, reference arrays/phase, RNG and sampled-support timers. Reference position arrays retain canonical body count even when the live model has two extra bodies. A cross-variant restore fails its contract. Exact source-host continuation is required; the earlier Mac compiled-contract portability failure is retained.

`run_transfer()` is the only shared registry/stop writer and owns the exclusive mission lease. Four worker processes write disjoint variant directories. A worker fault produces its own receipt; the parent requests peer stop. Atomic receipt renames are tolerated during disk accounting. All 80 formal slots exist before dependencies run. Completed native tracking failures are valid unfavorable outcomes; resource cuts are incomplete; numerical/application errors are invalid; unopened slots stay not_run. Benchmark/prerequisite attempts are retained separately and their physics counters enter the runner total.

The protocol fixes 1,800 seconds, 1.7 million runner physics steps, 1GiB raw output and at most four simulation workers/one thread each. Another 0.3 million physics steps are reserved for audit/video, keeping the mission below two million. The auditor and renderer enforce remaining wall/physics budget. The benchmark separates full worker-loop throughput from process-startup throughput and excludes optimizer/referee work; it cannot promise boxing throughput.

The existing source-host command is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --project experiments/v0.0.7/runtime python scripts/run_native_transfer.py --protocol experiments/v0.0.7/protocol.json --archive /home/ubuntu/joxing/downloads/LocomotionFull-2b2f0f10.zip --out runs/v0.0.7/transfer-01
uv run --project experiments/v0.0.7/runtime python scripts/verify_native_transfer.py --run runs/v0.0.7/transfer-01 --out runs/v0.0.7/audit-linux.json
MUJOCO_GL=egl uv run --project experiments/v0.0.7/runtime python scripts/report_native_transfer.py --run runs/v0.0.7/transfer-01 --audit runs/v0.0.7/audit-linux.json --out runs/v0.0.7/report --video
```

Use fresh output paths. The registered prerequisite points to the preserved v6 run/audit on xbabe1. Reproduction elsewhere must supply that verified prerequisite at its registered path or explicitly register a new location/protocol; never silently bypass source-host equivalence. Large weights remain acquired from the exact upstream archive. Every outcome and first-scenario replay stays in Git; full long trajectories belong in a public release archive with a committed manifest/hash, avoiding large raw runs in Git.
