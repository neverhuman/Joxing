# v0.0.5 evidence

Both waves are complete: **1,440,000 training transitions, 354,000 scheduled optimizer updates, 912 new evaluation attempts, zero invalids**. Every final policy had 0/32 five-second survivors. Extended candidate B improved versus its initial policy on both fresh seeds but lost to reference A on 983, failing registered replicated superiority. [Full interpretation](../../docs/V0.0.5_RESULTS.md), [experiment book](../../experiments/v0.0.5/README.md), [release summary](release-summary.json).

The root retains the original 40,000-transition-per-arm wave under source `6ac2bad`: protocol/descriptions, registry/events, all 336 attempts, actor-only bundles, progress/monitor/loss/resource records, constant baseline, real optimizer profile, both passing and failing audits, plots/video. Its best arm D missed the 0.25-second gate; four conditional slots remain not_run.

`extension/` contains the separately preregistered 160,000-transition-per-trial wave, source `76f6e5c`. All four screen arms and four fresh-seed reference/candidate jobs completed; 576 new attempts. `screen-freeze.json` hashes the selected screen before confirmation outcomes. `conclusion.json` retains every comparison; `wave-confirmation.png` shows both seeds; `wave-before-after-linux.mp4` renders the first registered scenario on the source host. The source/report hashes and CPU-physics/EGL distinction are in `render-receipt.json`. Eight reused diagnostic rows are retained but excluded from new attempts.

`host-scaling-concurrent/` and `host-scaling-16/` record the user-authorized 4→16 allowance benchmark, 24,000 timed rollout transitions, not learner throughput. `development/` retains tests and startup failures. Partial/full failed cross-host checks remain visible. `process-exit.json` confirms all owned training, benchmark, audit and render jobs exited.

State/history audit receipts and isolated latency/profile diagnostics are included. `diagnostic-accounting-correction.json` additively charges constructor preparation omitted by early diagnostic counters; timings/raw rows remain unchanged. Upstream checkpoint acquisition/metadata receipts record hashes/interfaces only; large downloaded weights stay in ignored `runs/`, and no teacher-transfer rollout is claimed.

`artifact-manifest.json` hashes all curated files, including the nested extension manifest. Actor-only `.pt` bundles support inference, not exact training resume; large replay/optimizer run directories remain ignored. All scientific raw evidence is preserved; derived reports may be regenerated without rewriting raw inputs.
