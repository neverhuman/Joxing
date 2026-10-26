# Training sequence and test gates

Repository: [neverhuman/Joxing](https://github.com/neverhuman/Joxing), public. **Current implemented iteration: v0.0.5; both waves are complete.** It extends v0.0.4’s native-gravity feedback platform with controlled objective/throughput experiments. The paragraph below and later formal targets preserve the earlier pilot context. Read the [v0.0.4 execution plan](V0.0.4_PLAN.md) for exact delivered scope and [STATUS.md](STATUS.md) for measurements. The [historical v0.0.3 plan](V0.0.3_PLAN.md) and later stages below describe the larger formal motor proof, which remains gated.

**Historical v0.0.4 outcome:** the pilot stopped at 39,040 transitions under its wall cap. All 64 development evaluations completed with zero invalids; mean survival changed 1.018 -> 1.023 seconds, with an interval including zero. Five-second survivors were 0/32, so the larger proof remains blocked. See [full results](V0.0.4_RESULTS.md). The v0.0.5 controlled wave below addresses those throughput/objective questions; its motor result still blocks the formal budget.

**Latest outcome:** 1.44 million training transitions and 912 new evaluations, zero invalids. Candidate B improved versus its initial policy on both fresh seeds but lost to A on seed 983, so registered superiority did not replicate. Every final policy had 0/32 five-second survivors. The formal proof remains gated. All owned jobs exited; [complete v0.0.5 results](V0.0.5_RESULTS.md).

## v0.0.5 controlled motor wave

The [v0.0.5 plan](V0.0.5_PLAN.md) and [experiment book](../experiments/v0.0.5/README.md) replace another undirected pilot with a registered experiment: constant muscle controls, real optimizer profiling, a matched reward-scale/temperature factorial and conditional fresh-seed confirmation. [Results](V0.0.5_RESULTS.md) retain all outcomes and the reason each conditional run did or did not execute. The earlier formal stance/recovery targets below remain gated.

All four screen arms use the same seed149, initial actor, 128×128 network, batch64, 2,000-transition warmup and one optimizer update per four transitions. Temperature is auto or0.0001; reward multiplier is1 or1000. Each run ends at40,000transitions or900seconds. Raw physical survival on32paired starts is the common outcome. Altering training reward units does not alter boxing points or physics.

Replication requires a complete valid screen, at least0.25s gain over the selected actor’s initial policy and, for a non-reference selection, at least0.10s over reference A. Confirmation uses fresh seeds263/431 and unseen reset IDs61000–61031, opened only after final actors are frozen. Scenario bootstrap intervals do not capture uncertainty across training seeds. The unchanged five-second continuation gate still requires16/32 survivors, zero invalids; it does not establish the full motor proof.

```bash
uv sync --extra dev --extra viz --extra motor
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python scripts/run_wave.py --protocol experiments/v0.0.5/protocol.json --out runs/v0.0.5/reproduction
uv run python scripts/verify_wave.py --run runs/v0.0.5/reproduction --out runs/v0.0.5/reproduction-audit.json
uv run python scripts/report_wave.py --run runs/v0.0.5/reproduction --out artifacts/generated/v0.0.5-reproduction --video
```

Output directories must be fresh. The source checkout stays frozen during execution. The journal, registry, per-experiment descriptions, every evaluation row, actor bundles and raw/scaled reward counters are retained. Actor-only bundles support inference, not exact training resume. The legacy `train-stance` command retains v0.0.4 settings and is not the v0.0.5 factorial runner. CPU Torch is intentional; no GPU speedup is claimed.

## Authorized larger-budget extension

A [separately registered extension](V0.0.5_BUDGET_ADDENDUM.md) raises each trial to 160,000 transitions / 2,700 seconds, first limit. Four independent learners run concurrently with four simulation workers each, 16 total. It retains the common learner/update ratio and all physical/observation/action contracts. Screen seed 547 uses starts 64000–64031; a qualified candidate is compared with reference A on fresh seeds 761/983 and starts 65000–65031. Read the results for actual outcomes and the constant-baseline timing limitation. Initial/final learner-policy confirmation evaluations are deferred until final actor save.

To reproduce the two-stage workflow, first complete the original runner into a **fresh** parent directory as above, then immediately run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python scripts/run_wave_extension.py --protocol experiments/v0.0.5/extension-protocol.json --original runs/v0.0.5/reproduction --out runs/v0.0.5/reproduction-extension
uv run python scripts/verify_wave_extension.py --run runs/v0.0.5/reproduction-extension --original runs/v0.0.5/reproduction --out runs/v0.0.5/reproduction-extension-audit.json
uv run python scripts/report_wave.py --run runs/v0.0.5/reproduction-extension --out artifacts/generated/v0.0.5-reproduction-extension --video
```

The extension intentionally inherits a four-hour deadline from the parent's provenance timestamp. Using the archived October 26 parent for a new training run after that deadline fails with `Combined wall cap already reached`; do not edit the old timestamp to bypass it. Offline verification/reporting against archived evidence remains valid. Both runner directories must be fresh and their executing source stays frozen. The released evidence retains the original source commits; a reproduction from another source revision records its own provenance.

The [state/history audit](STATE_HISTORY_SELF_PLAY_AUDIT.md) explains the difference between present feedback and the missing temporal opponent/game loop. The [upstream motor-transfer plan](UPSTREAM_MOTOR_TRANSFER_PLAN.md) is proposed development work, not an additional completed trial or a new self-play command.

## What the next proof establishes

Train one MS-Human-700 athlete to maintain a local standing posture under normal gravity, then qualify recovery from small, measured horizontal pushes. Keep all 700 muscles, physical floor contact and explicit glove inertia. The policy observes the moving body and updates muscle excitation at 100 Hz; CPU physics advances at 1 kHz. v0.0.4 implements this platform and the development balance pilot; the three-seed stance/recovery proof follows only if the pilot gate passes.

The v0.0.2 arm controller had a fixed pelvis and zero gravity. Its successful contact/withdrawal result is preserved, but its parameters cannot be loaded as a whole-body stance policy. v0.0.4 evaluates an upright starting state after explicit preparation/projection; standing up from the floor is outside this proof. Hand targets are their reset positions, not a raised boxing guard.

The objective is a reproducible test platform plus an honest measurement of learning. Passing the platform tests, passing the motor thresholds and showing improvement are separate outcomes. A physically valid failed balance trial is retained; a numerical fault invalidates the trial and stops the affected run.

## Build and training order

| Order | Work | What to test before continuing | Output |
|---|---|---|---|
| A | Build the named gravity rig and root probes | Native dimensions/muscles/equalities, segment inertia, root frames, floor force signs; no hidden support | Model/assistance audit, four-direction fall traces |
| B | Build support monitor and procedural reset bank | Foot versus hand/knee/trunk support, persistence at different dt, limits/equalities, penetration, repeatable preparation | Reset bank with rejection log and support fixtures |
| C | Implement feedback environment and learner | 138-group exact muscle coverage, measured observations, 10-substep action hold, correct failure/truncation handling, restore during push/contact | Passing API/replay tests, locked learner dependencies |
| D | Benchmark and run a development pilot | 1/2/4-worker throughput, memory, loss/fault counters; final pilot policy completes 5 seconds on at least 16/32 development starts, zero invalids | Benchmark and pilot receipts, initial/final development policies |
| E | Freeze the formal protocol | Separate seed namespaces, sealed final reset bank, initial baseline hashes, checkpoint selection rule, budgets, success and improvement metrics | Committed protocol and baseline/reset artifacts |
| F | Train three independent seeds, sequentially | Stable action/observation contracts, finite transitions, no assistance, budgeted progress/checkpoints | Frozen final policy per seed, all training history and faults |
| G | Evaluate paired policies and publish | Exact attempt coverage, denominators, stance/recovery targets, independent replay, fixed-scenario video | Results JSON/rows, plots, video, audit and versioned docs |

Read-first files, module/class names, editable paths and forbidden/generated zones are specified in the [detailed plan](V0.0.3_PLAN.md). Execute with one integration owner. Third-party submodules, supplied reference plans and v0.0.2 evidence remain read-only. If the native root is defective, a separately hashed free-root diagnostic must pass equivalence tests before being selected on development evidence.

## Historical v0.0.4 pilot defaults and formal-proof budgets

Use SAC for feedback muscle control, through the pinned upstream 138-group muscle partition converted to names. Preserve the upstream normalized-action excitation transform. Begin with two 256-unit hidden layers, learning rate 3e-4, batch 256, gamma .99, tau .005, automatic entropy, 10,000-transition warmup and one optimizer update per admitted control transition after warmup. Lock dependency versions and measure replay storage before launch. These are starting pilot settings, not established optimal parameters.

The motor reward integrates upright posture, position and guard-target tracking, with nonnegative excitation-effort and action-change penalties. A physical fall terminates; the 20-second task horizon truncates with a valid final observation for bootstrap. Invalid transitions are quarantined and never trained as ordinary terminal experiences. The precise reward and failure contracts are in the detailed plan and must be frozen before formal proof.

| Budget | Limit |
|---|---|
| Development pilot | Seed 73; 100,000 control transitions, at most 1 million physics substeps or 30 minutes |
| Formal training | Seeds 103, 211, 307; at most 500,000 control transitions / 5 million physics substeps / 60 minutes per seed |
| Entire compute experiment | At most 40 million physics substeps, four wall hours and 10 GiB raw artifacts, including pilot, benchmarks, diagnostics, failures and evaluation |
| Original v0.0.4 shared CPU resources | Four simulation workers total, one OpenMP/BLAS thread per process; seeds sequential. The authorized v0.0.5 extension above uses 16 workers across four independent trials. |

Stop at the first applicable cap. A control transition is one world's 10 ms policy step; four workers do not make one transition count as four or multiply the global budget. Formal training uses the scheduled final checkpoint; an earlier resource stop is reported as incomplete, with remaining planned attempts marked not_run. Never select a checkpoint using final evaluation scores.

Start on xbabe1 CPU MuJoCo after a fresh capacity check. Use xbabe4 native CPU for independent replay/render; xbabe2/3 are fallback CPU resources after checking availability. No SSH access is needed to use the public source locally. A GPU neural learner is a measured optional variant; Warp GPU physics remains unqualified. Do not claim GPU acceleration from CPU simulation or install drivers/change shared services for this iteration.

Report end-to-end transitions/s, physics substeps/s, simulated time/wall time, optimizer updates/s, reset cost, peak RAM/VRAM, invalid rates and motor success gained per wall hour. Separate compile/warmup and rendering. The compute inventory in [COMPUTE.md](COMPUTE.md) is historical; it does not reserve any host.

## Held-out tests and acceptance

Freeze all rules before formal training. Development uses IDs 30000–30031, training uses 100000–199999, final quiet stance uses 40000–40063 and final recovery uses 50000–50063. Persist purpose-separated RNG keys and complete reset/activation state. Candidate and initial baseline receive identical starts and push schedules for each pair.

For each of three seeds, evaluate both the frozen initial and final policies on 64 quiet and 64 recovery scenarios: `3 * 2 * (64 + 64) = 768` planned attempts, each up to 20 seconds.

| Test | Scenario | Required motor result, separately for every seed |
|---|---|---|
| Quiet stance | Normal gravity, randomized accepted upright start, no external push | At least 61/64 successes; finish 20 seconds without motor failure, with torso tilt <20 degrees and pelvis horizontal drift <15 cm throughout the last 2 seconds |
| Push recovery | One 50 ms trunk-COM push at t=5 s; 2 or 4 N s in four horizontal directions, eight repetitions per combination | At least 58/64 successes; meet stance bounds within 2 seconds after push end and keep them through episode end |
| Numerical validity | All planned attempts and deterministic stress/replay checks | Zero invalids for a passed proof; no unattempted scenario hidden from the denominator |
| Assistance and replay | Audit model, applied forces, state writes and full snapshots | No fixed root/hidden support/gravity compensation during evaluation; reproduce state and application history on restore |

Measure delivered push impulse independently; prescribed and delivered values must agree within 1e-6 N s. Pushes are test stimuli, not empirical punch-force calibration. A brief glove-floor brush does not automatically mean a fall; the support monitor tests force/persistence. Final sporting foul/fall rules remain unresolved.

Preregister **paired recovery-success improvement over the initial policy** as the primary learning metric. Report its paired-scenario bootstrap 95% interval per seed using 10,000 resamples and bootstrap RNG seed 61003. A positive lower bound for every seed supports a limited learning claim. Quiet-stance improvement and stable seconds are secondary measurements; do not switch the primary metric after viewing results. Passing absolute motor targets can establish competence even when improvement over a strong baseline is inconclusive. Three seeds do not establish broad algorithm superiority.

All attempts remain in raw accounting. Store success, physical/task failure, invalid and not_run with reason, model/protocol/policy/reset hashes, timing, force and snapshot paths. A numerical fault produces a last-valid-state receipt and halts the run. A missed target produces a failed motor result. Any retuned proof uses a new protocol and fresh final scenarios; preserve the original failure report.

## Regression tests and visible proof

Keep the existing contact/scoring/withdrawal tests and published checkpoint audit passing. Add tests that can detect wrong root axes, missing muscles, inertia changes, mis-signed/double-counted support, inconsistent persistence at 2/1/0.5 ms, bad resets, fake observations, action leakage, wrong terminal bootstrap, injected numerical faults, duplicate attempts and incomplete snapshot state. Restore between motor ticks, during a push, at foot liftoff and during fall-persistence accumulation. These tests validate mechanisms rather than copying reward arithmetic.

Published proof includes the model/reset/assistance audits, frozen protocol, all evaluation rows, per-seed policies/results, resource/throughput receipts, Mac/Linux test logs and independent replay audit. Show all seeds against physics budget and wall time. Use preselected scenarios 40000 and 50000 for seed 103 in the before/after video; also show the first failure by scenario order. Display gravity, assistance, support force, tilt, delivered push, outcome and policy identity. Label any missing artifact as not_run with its blocking reason.

## What runs now, and what comes later

The [README Start commands](../README.md#start) execute the current package; existing strike/reach commands retain their v0.0.2 behavior. v0.0.4 implements `balance-probe`, `train-stance`, the `motor` extra and `scripts/{verify_balance,report_balance,benchmark_balance}.py`. Install with `uv sync --extra dev --extra viz --extra motor`, then use the [v0.0.4 commands](V0.0.4_PLAN.md). The current `train-stance` CLI runs a development pilot; it does not silently execute the proposed formal proof. `freeze-stance`, a standalone `evaluate-stance`, configuration-file dispatch and arbitrary-substep snapshotting remain future interfaces. Source is committed before compute; evidence follows separately. Policy/optimizer checkpoints support evaluation, not full training resume with replay/RNG restoration.

After the development pilot passes, freeze and run the full three-seed stance/recovery proof above. If it fails, publish that failure and fix the observed balance/learning bottleneck first. Once formal balance passes, propose a separate next version for unassisted reach/return under gravity while keeping balance and contralateral guard. Then qualify fitted two-human collision/zone/guard behavior, complete foul/fall/boundary attribution and simultaneous environment replay. Only after those gates should historical-opponent self-play begin. Later release numbers are not assigned by this guide.

The boxing contract remains head 2 points, torso 1, guard 0 per qualified encounter, with win/loss/draw tactical utility. Balance reward is preparation for that game; it does not change match points. See [GAME.md](GAME.md) and [IMPLEMENTATION.md](IMPLEMENTATION.md) for the complete progression.
