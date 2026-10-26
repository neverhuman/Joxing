# Compute inventory and HPC approach

Public repository: [neverhuman/Joxing](https://github.com/neverhuman/Joxing). The latest workload is [v0.0.7 transfer qualification](V0.0.7_RESULTS.md): all 80 attempts in 141.44 s,256,920 runner physics steps (including prerequisites/benchmark), four one-thread CPU workers; no optimizer. Exact Linux replay passes but every longer tracking condition fails. All owned jobs exited. One/four-worker loop medians 91.24/256.50 control TPS exclude startup; short-case startup-inclusive medians 4.64/15.53 TPS. [v8](V0.0.8_MISSION.md) proposes two workers after fresh capacity inspection. The earlier [v0.0.6 native motor qualification](V0.0.6_RESULTS.md): one CPU world/one thread,40 valid attempts,70,400 formal physics steps in 77.2294 seconds. It used the checkpoint-compatible locked runtime on xbabe1; exact replay passes. No optimizer or GPU physics ran. All owned jobs exited. The completed v7 run used at most four simulation workers after fresh capacity/benchmarking. The prior [v0.0.5 controlled motor wave and expanded-budget extension](V0.0.5_RESULTS.md): 1.44 million training transitions, 912 new evaluations, zero invalids. The user authorized 16 aggregate simulation workers and 160,000 transitions / 45 minutes per extension trial; four independent learners retained four worlds each. The global 40-million-step/four-hour/10-GiB cap remained unchanged. All owned jobs have exited; shared-host availability must be inspected again before another launch.

Completed v0.0.4: the pilot hit its wall cap at 39,040 transitions; total wall with final evaluation was 1,848.367 s. All 64 evaluations finished with zero invalids, but the motor gate failed. Separate inference/support/reset medians were 65.71 / 131.81 / 240.52 control transitions/s at 1/2/4 workers (three repeats); optimizer work was excluded. SAC's gradient phase was roughly 17–18 transitions/s. CPU rollout scaling is useful, but more rollout workers alone do not resolve the learner bottleneck. [Results](V0.0.4_RESULTS.md), [benchmark](../artifacts/v0.0.4/benchmark.json). No v0.0.4 job remains running.

Read-only SSH inventory taken October 26, 2026 at approximately 05:13 America/Denver. These are shared hosts: availability is transient, not a resource reservation. No unrelated process or service was changed. No credentials or raw SSH config are stored here.

| Host | Hardware / availability at inspection | Initial role |
|---|---|---|
| xbabe1 | Threadripper PRO 3995 WX, 64 physical / 128 logical cores; 62 GiB RAM / 40 GiB available; RTX 4090 24 GiB, idle; 334 GB free disk; load ~15 | First bounded CPU physics/motor experiments; preferred first GPU learner/feature spike |
| xbabe2 | Threadripper PRO 3995 WX, 64 physical / 128 logical cores; 125 GiB RAM / 74 GiB available; RTX 3090 24 GiB, idle; 487 GB free disk but filesystem 94% used; load ~44 | Later capped rollout/evaluation workers; credentials available if needed, never copied into repository |
| xbabe3 | Threadripper PRO 3995 WX, 64 physical / 128 logical cores; 125 GiB RAM / 99 GiB available; RTX 3090 24 GiB, idle; 63 GB free disk, 94% used; load ~9 | Later capped CPU rollout/seed experiments; keep artifacts small |
| xbabe4 | Apple M 3 Ultra Mac Studio, 32 logical CPUs, 512 GiB unified RAM; 667 GiB free disk; load ~28 | CPU reference, large-memory analysis, replay/render; native Python first |
| Local laptop | Apple Silicon, 14 logical CPUs, 36 GiB RAM, 75 GiB disk available | Development, tests, inspection and reporting |

Linux hosts have Docker. xbabe1 and xbabe4 have uv; xbabe2/3 have Python. Docker was not on xbabe4's inspected PATH. macOS Docker cannot provide an NVIDIA CUDA device; it is not the route to Warp GPU physics. Native CPU MuJoCo on Apple Silicon avoids VM overhead. Recheck GPU support before considering Metal-specific alternatives.

## First actual workload

Use `/home/ubuntu/joxing/bootstrap-20261026` on xbabe1. Start at four CPU workers, one thread per process (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`). This is intentionally small relative to host count and leaves existing services alone. Simulation is CPU MuJoCo; do not claim RTX utilization from this workload. Store raw output under `runs`, copy small receipts to `artifacts/receipts` and stop the job after its explicit episode budget.

## Fast learning path

1. Establish contact correctness and CPU performance on the actual two-human model. Report compile time separately; benchmark medians and invalid runs. Current benchmark is a short-reset simulator diagnostic, not boxing throughput.
2. Use a low-cost constrained muscle-reach experiment to verify optimization, worker orchestration, checkpoint persistence, held-out paired evaluation and replay. It is not the motor teacher or tactical learner.
3. Build native stance/reach teacher with fixed muscle groups first. Use measured replay-memory cost `4*C*(2*O+A)` bytes plus metadata. Do not allocate giant replay buffers from an assumed observation dimension.
4. Audit exact-model Warp support: spatial tendon wrapping, all muscle dynamics, equalities, collider primitives, staging, integrator and capacities. Lock compatible engine/Warp versions together. CPU full-anatomy parity precedes any GPU replacement. Generic GPU installation is not feature validation.
5. Sweep CPU workers 1/2/4/8 before larger counts. Separately sweep GPU worlds 8/32/128/512 up to the measured 24 GiB budget, retaining 20% memory headroom for policy/optimizer/league. Never assume 2,048 or 4,096 worlds fit.
6. Keep simulation state, inference and referee on GPU once parity passes; avoid Python contact loops and per-substep host sync. Keep video/render on separate evaluation workers. Benchmark with real contact/referee/reset/inference load.
7. Use xbabe1 as first learner and xbabe2/3 for independent evaluation and additional seeds only after measuring LAN transfer and queue latency. Transfer compact transitions/bundles, not full engine state every substep. Freeze normalizers and bundle identity during each matchup.

Report physics substeps/s, simultaneous world transitions/s, admitted learner samples/s, simulated seconds/wall second, optimizer updates/s, memory, GPU utilization, reset/contact invalid rates, and evaluation strength per wall hour. Include compile/warmup, transport and optimization costs in end-to-end runs. A two-player world step is not automatically two on-policy learner samples.

Stop on unexpected warnings/engine resets, nonfinite state, contact loss/overflow, unexplained score parity failures, disk below the run's reserved budget, or sustained interference with shared-host capacity. Preserve last valid engine state, held controls and referee state before investigation. No cloud GPU purchase or host service changes are required for this first slice.

## Measured GPU spike

Actual probes are in `artifacts/receipts/gpu-probe-*.json`; see RESULTS.md. `mujoco-warp==3.8.0` with `warp-lang==1.12.0` detects xbabe1's 4090 on driver 570.153.02. Auto-resolved Warp 1.18 requires a newer CUDA driver, so the Linux `gpu` extra now pins 1.12. The exact reference scene is rejected due to mesh-margin/MULTICCD support. An explicit MULTICCD-disabled variant executes eight worlds but fails the declared short-state parity tolerance. GPU physics remains unqualified; no speedup is claimed.

Driver follow-up: xbabe2 reports 595.99.02 and xbabe3 580.126.09, both newer than xbabe1. These hosts are alternative candidates for newer CUDA/Warp investigations without touching xbabe1's driver. Driver compatibility alone cannot resolve scene support or physics parity. They were inventoried, not used for the first training job.

Reproduce on Linux with `uv sync --extra dev --extra viz --extra gpu`, then `uv run python scripts/gpu_probe.py --worlds 8 --out runs/gpu-exact.json`. The deliberate variant uses `--disable-multiccd` and receives a distinct physics hash. These are diagnostic commands, not a production backend. [Official MuJoCo 3.8 Warp documentation](https://mujoco.readthedocs.io/en/3.8.0/mjwarp/index.html) distinguishes high-throughput batching from single-world latency and requires explicit contact/constraint capacity tuning.

## v0.0.5 authorized resource increase

The original controlled wave finished with all 336 evaluations valid. Its real optimizer profile measured 91.99 vs 19.62 updates/s for the smaller versus prior network/batch,4.69×; this excludes simulation. The update ratio also changed from 1 to 0.25 per transition, so old/new learner quality is not a controlled comparison.

The user explicitly increased worker AND training budgets. Aggregate allowance is now 16 simulation workers on xbabe1, with one thread per process. [H 0 resource addendum](V0.0.5_RESOURCE_ADDENDUM.md) measured 4/8/12 worker rollout medians 220.48/436.01/646.79 transitions/s alongside the four-worker wave, then 844.67 at 16 workers after it exited; 24,000 timed transitions, zero invalids. Physics/support/reset/inference are included; optimization is excluded. Results are shared-host timing, not a reservation or GPU measurement.

[The larger-budget extension](V0.0.5_BUDGET_ADDENDUM.md) uses four independent learner jobs concurrently, each retaining four worlds, the same network and 0.25 update ratio. Each gets 160,000 transitions/45 minutes, first limit. It ran from a separate frozen checkout `/home/ubuntu/joxing/v005-extension-20261026` under source 76 f6e5c. Preflight confirmed identical physical/action/observation contracts, load 4.43, about 47 GiB available RAM and 323 GiB free disk. Retain the four-hour deadline from original launch,40 million total physics steps and 10 GiB raw-artifact ceiling; disk/memory floors remain 10/4 GiB. No other owned simulation workload shares its 16-worker allocation. CPU Torch is intentional; GPU physics is still unqualified. See STATUS for live/completed PIDs and final measurements.


## Final v0.0.5 compute conclusion

The extension completed eight 160,000-transition trials (screen + two paired confirmation seeds), 316,000 optimizer updates and 576 new evaluations in 3,555.388 wall seconds: about 360 end-to-end training transitions/s including setup/baselines/evaluation. This differs from the 844.67 TPS fixed-policy benchmark and from per-world latency. Original plus extended formal physics/preparation totaled 15,989,627 steps; with the conservative one-million supplementary reserve, 16,989,627 remains below 40 million. The larger budgets finished without a resource stop; all registered PIDs and dedicated-checkout Python jobs exited, including audits/rendering (`artifacts/v0.0.5/process-exit.json`).

A tiny policy network was not a measured inference bottleneck on Mac: isolated batch-one medians 0.062–0.064 ms, p 99 up to 0.124 ms in the recorded samples. A short recorded-action continuation spent about 76%of profiled time in native simulation steps. Neither measurement qualifies complete two-player game deadlines or forecasts the upstream larger teacher network. Read the [state/control audit](STATE_HISTORY_SELF_PLAY_AUDIT.md) for methodology, scope and the preparation-counter correction.

The extra compute produced small motor gains but no five-second survivor and no replicated superiority over reference A. The [upstream motor teacher](UPSTREAM_MOTOR_TRANSFER_PLAN.md) now passes native qualification; next work is equipment/timestep transfer and feasible resets, then benchmark the full state/history/referee/league workload. Do not extrapolate single-athlete rollout TPS into a 1,400-muscle boxing throughput promise. CPU physics remains the qualified reference; no GPU-physics improvement is claimed.
