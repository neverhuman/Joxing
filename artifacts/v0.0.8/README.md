# v0.0.8 evidence

[Results](../../docs/V0.0.8_RESULTS.md), [experiment book](../../experiments/v0.0.8/README.md), [release summary](release-summary.json). All40 valid physical episodes fail the20s gate; all12 replays are exact. No training or boxing claim.

`physical/` contains every outcome/start/specification, preregistration, benchmark/oracles/progress, firstseed full commands/substeps/replay and plot/video. `audit-linux.json` independently checks all40 outcomes and12 continuations. `development/` preserves passing/failing tests; `deployment/` preserves original ownership/driver snapshots. Final `supervisor.json` and `process-exit.json` record completion and actual exits. Original captured whitespace is retained.

All184 raw files are available in the47,279,704byte [release archive](https://github.com/neverhuman/Joxing/releases/download/v0.0.8/joxing-v0.0.8-raw.tar.gz), SHA256 `5b5dbacf4f113f04551616eb4c2498bd5357c86fc49f340981ba4a440a5b60ab`. Raw/action histories outside firstseed stay there to avoid large Git runs. Original model/normalizer weights are reconstructible from the pinned upstream archive, not committed here.
