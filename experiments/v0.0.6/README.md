# v0.0.6 experiment book

Mission: qualify a frozen upstream motor controller and repair reproducibility of its decoder/normalizer bundle. [Execution plan](../../docs/V0.0.6_PLAN.md), [protocol](protocol.json), [mission state](../../docs/MISSION.md). Protocol was committed as `75e9a18` before checkpoint loading or native motor outcomes. No optimization or boxing/transfer experiment is implied.

L0 validates archive/source/assets/interfaces, reproduces default amplitude loss and corrected save/load, and freezes normalization. N0 runs twenty paired native starts (40 attempts), default versus restored schedule, one CPU world under 900 s/1 Msteps/1 GiB raw. R0 uses the first registered seed 71000 and step 10 snapshot for independent native action/policy replay and fixed-scenario video. Native diagnostic threshold: >=16/20 restored three-cycle completions, all 40 attempts valid, strict source-host replay. It does not establish Joxing quiet stance or boxing.

Status: **completed and independently audited on Linux**. Frozen source `97565e90012b59bb16db8c45e4aa7f095a905c3c`; native diagnostic passes. [Complete results and reproduction](../../docs/V0.0.6_RESULTS.md), [machine summary](../../artifacts/v0.0.6/release-summary.json), [evidence index](../../artifacts/v0.0.6/README.md).

| Experiment | Result | Evidence |
|---|---|---|
| L0 load/normalizer | Default amplitudeNone vs scheduled 0.075; probe mean absolute action change 0.030902829; corrected save/load error 0; immutable stats | [Oracle](../../artifacts/v0.0.6/native/loader-oracle.json) |
| N0 frozen native task | Default 20/20,restored 20/20 three-cycle completions; 40 valid, 0 invalid/incomplete/not_run; 3.52 simseconds each | [Every attempt](../../artifacts/v0.0.6/native/evaluations.json), [raw summary](../../artifacts/v0.0.6/native/result.json) |
| R0 source-host replay | Firstseed 71000 step 10, both modes, fixed commands/closed-loop/repeat: zero state/observation error | [Linux audit](../../artifacts/v0.0.6/audit-linux.json), [fixed video](../../artifacts/v0.0.6/native/native-controller.mp4) |
| Portability | Mac raw accounting/hashes pass; strict compiled-contract check fails before reference replay; no tolerance relaxed | [Failure](../../artifacts/v0.0.6/audit-mac-failure.json) |

N0:77.2294 wall seconds/70,400 physics steps; one CPU world/one thread within 900 s/1 Msteps/1 GiB raw. R0 audit 9,960 steps; video 3,520 steps. All owned processes exited. Full 80 tests pass Mac/Linux, final 8 target Mac tests pass. Every raw attempt, start, command, trace, progress/event, source/actor/normalizer hash and development failure remains in Git; known-hash large weights are reacquired from upstream.

Both modes hit the completion ceiling: no schedule-restoration performance advantage established. Zero optimization occurred; upstream 45 Mtraining steps are provenance. Native 2 ms physics/50 Hzcontrol walking is separate from stationary guard, boxing points, opponent memory and self-play. The raw summary's pending-replay marker is historical; the independent R0 audit supplies the final qualification. Next measured mission is [v0.0.7 transfer](../../docs/V0.0.7_MISSION.md), ready but unexecuted.
