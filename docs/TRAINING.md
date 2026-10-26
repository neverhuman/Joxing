# Training sequence and test gates

Repository: [neverhuman/Joxing](https://github.com/neverhuman/Joxing), public. Current implemented release: v0.0.2. **Next implementation and training iteration: v0.0.3.** Its stance code and training runs are pending. This guide summarizes the execution sequence; [V0.0.3_PLAN.md](V0.0.3_PLAN.md) is the detailed worker handoff and [STATUS.md](STATUS.md) records actual execution.

## What the next proof establishes

Train one MS-Human-700 athlete to maintain a local standing posture and recover from small, measured horizontal pushes under normal gravity. Keep all 700 muscles, physical floor contact and explicit glove inertia. The policy observes the moving body and updates muscle excitation at 100 Hz; CPU physics advances at 1 kHz.

The v0.0.2 arm controller had a fixed pelvis and zero gravity. Its successful contact/withdrawal result is preserved, but its parameters cannot be loaded as a whole-body stance policy. v0.0.3 evaluates an upright starting state; standing up from the floor is outside this proof.

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

## Learner and budgets

Use SAC for feedback muscle control, through the pinned upstream 138-group muscle partition converted to names. Preserve the upstream normalized-action excitation transform. Begin with two 256-unit hidden layers, learning rate 3e-4, batch 256, gamma .99, tau .005, automatic entropy, 10,000-transition warmup and one optimizer update per admitted control transition after warmup. Lock dependency versions and measure replay storage before launch. These are starting pilot settings, not established optimal parameters.

The motor reward integrates upright posture, position and guard-target tracking, with nonnegative excitation-effort and action-change penalties. A physical fall terminates; the 20-second task horizon truncates with a valid final observation for bootstrap. Invalid transitions are quarantined and never trained as ordinary terminal experiences. The precise reward and failure contracts are in the detailed plan and must be frozen before formal proof.

| Budget | Limit |
|---|---|
| Development pilot | Seed 73; 100,000 control transitions, at most 1 million physics substeps or 30 minutes |
| Formal training | Seeds 103, 211, 307; at most 500,000 control transitions / 5 million physics substeps / 60 minutes per seed |
| Entire compute experiment | At most 40 million physics substeps, four wall hours and 10 GiB raw artifacts, including pilot, benchmarks, diagnostics, failures and evaluation |
| Shared CPU resources | At most four simulation workers total, one OpenMP/BLAS thread per process; seeds run sequentially |

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

The [README Start commands](../README.md#start) execute v0.0.2. `balance-probe`, `benchmark-balance`, `freeze-stance`, `train-stance`, `evaluate-stance`, the `motor` extra and the balance report/audit scripts are **planned interfaces**. Implement them in the detailed plan's order before running its commands. Source is committed before compute; frozen protocol/bank/baselines are committed before formal training; evidence follows in a separate result commit. Update STATUS after each gate and tag a release only with its implemented scope and measured outcomes.

After v0.0.3 passes, propose a separate next version for unassisted reach/return under gravity while keeping balance and contralateral guard. Then qualify fitted two-human collision/zone/guard behavior, complete foul/fall/boundary attribution and simultaneous environment replay. Only after those gates should historical-opponent self-play begin. Later release numbers and success thresholds are not assigned by this plan.

The boxing contract remains head 2 points, torso 1, guard 0 per qualified encounter, with win/loss/draw tactical utility. Balance reward is preparation for that game; it does not change match points. See [GAME.md](GAME.md) and [IMPLEMENTATION.md](IMPLEMENTATION.md) for the complete progression.
