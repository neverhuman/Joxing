# Physical motor continuation contract

The registered [v0.0.8 mission](V0.0.8_MISSION.md) separates a walking controller's native reference-tracking failure from persistent physical motor failure. The implementation is `src/joxing/physical_motor.py`, with bounded execution/accounting shared with `native_transfer.py`. This is a named diagnostic, `tracking-independent-motor-continuation`; it does not repair the old tracking task, qualify stationary stance or enable competitive boxing. Current execution status is in [MISSION](MISSION.md) and the [experiment book](../experiments/v0.0.8/README.md).

## Unchanged inputs and distinct outcomes

Keep the pinned original anatomy and all 700 muscles, native excitation transform, 20 ms control cadence, decoder amplitude 0.075, immutable checkpoint/normalizer and original reference/reward arithmetic. Native 2 ms and gloves 1 ms are separate registered conditions. Record first native tracking failure at the old pre-reference-update check (`mean(abs(qpos[3:]-reference[3:])) > 0.06`). It remains a failed tracking gate even if the new physical task continues.

The diagnostic latches persistent nonfoot support, excess tilt or pelvis drop using the existing motor `SupportMonitor`. It finishes the held control interval, retains exact latch time and final boundary time separately, and ends at the 20 s horizon only if no physical failure has latched. A physical failure at the horizon is a failure, not a horizon completion. Neither outcome defines sporting fall/foul adjudication.

## Physics and measurement timing

Gym assigns controls, executes `mj_step(..., nstep=frame_skip)` and calls `mj_rnePostConstraint` once. The instrumentation replaces that batch with identical individual Euler steps and keeps the one final RNE call. No extra live `mj_forward`, pose write, control filter or external force is introduced.

After each step, the contact geometry/frame/solved force cache describes that engine interval's start pose. Convert each contact-frame force to world coordinates and orient it onto the owned body. Preserve signed xyz forces and every contributing floor manifold point. Sum positive world-z force by right foot, left foot and nonfoot role; unknown identities fail. Each interval explicitly stores start/end times. Compute end posture using current qpos with `mj_kinematics` on separate `MjData`; never label a live cached start pose as the end measurement. Control-level end spatial velocities also use scratch kinematics/comPos/comVel and `mj_objectVelocity`, with angular xyz followed by linear xyz in world orientation.

The native actor retains its original cached feature semantics. Raw history in the files is telemetry, not newly supplied recurrent memory. Exact action/state/feature/reward equivalence before the old tracking failure must pass synthetic and actual source-host v7 oracles before new outcomes are admitted.

## Snapshot and raw evidence

Each snapshot contains native integration/reference/RNG/cache state plus `physical_control`: pending flag, completed controls, held normalized command, decision qpos, control start, partial substep index, current interval evidence, tracking onset and physical latch. The actor's cached observation at a partial step intentionally belongs to the previous decision. Validate that decision pose and exact excitation separately from current qpos. Resume a pending command without another policy decision.

Restore validates contracts, dimensions, finite values, clock, command/control agreement, contact roles/load sums, monitor durations, RNG and application fields before committing a separate restored `MjData` to the live task. Base reference restoration and cache rebuilding are isolated from the live state. Physics work counters never rewind on restore. Full episode evidence is output, not required acting state; snapshots only carry the current held interval.

`substeps-SEED.npz` stores interval times, loads, tilt/drop, contact offsets and packed geometry/role/force/position arrays. `actions-SEED.npz` and `trace-SEED.json` retain commands and control telemetry. Starts, first-scenario boundary/mid-command snapshots, final state, every attempt/fault and process/progress receipts are separate files. Parent alone writes shared stop/registry files and flushes final progress from the completed registry. Partial physics counts include an attempted failing substep.

## Execution, validation and audit

Run only from a new dedicated checkout/output directory after source freeze, full regressions and fresh shared-host inspection. Prefix local shell commands with `rtk` per RTK. The exact commands, ownership and completed outcomes must be recorded before a release claim.

```bash
uv sync --project experiments/v0.0.8/runtime --locked
uv run --project experiments/v0.0.8/runtime pytest -q
uvx ruff check src scripts tests
uv run --project experiments/v0.0.8/runtime python scripts/run_physical_motor.py --protocol experiments/v0.0.8/protocol.json --archive /home/ubuntu/joxing/downloads/LocomotionFull-2b2f0f10.zip --out runs/v0.0.8/physical-01
uv run --project experiments/v0.0.8/runtime python scripts/verify_physical_motor.py --run runs/v0.0.8/physical-01 --out runs/v0.0.8/audit-linux.json
MUJOCO_GL=egl uv run --project experiments/v0.0.8/runtime python scripts/report_physical_motor.py --run runs/v0.0.8/physical-01 --audit runs/v0.0.8/audit-linux.json --out runs/v0.0.8/report --video
```

The runner first validates archive/runtime/pins, exact synthetic equivalence and both actual first-seed v7 trajectories, then benchmarks one/two workers before opening the 40 fresh registered scenarios. Its limits are two one-thread workers, 900 s, 700,000 runner physics attempts and 1 GiB raw, first reached; mission total with audit/video is at most 950,000 physics attempts. Resource floors and all other frozen fields remain in the [protocol](../experiments/v0.0.8/protocol.json).

The independent auditor checks all 40 attempt identities, pairing, controls, counters, clipping/variation, tracking onsets and signed contact reconstruction/persistence. It then runs all 12 predeclared continuations (two conditions × boundary/mid-command × fixed-action/closed-loop/repeat), retaining completed/invalid/not_run findings and work counts. Original source-host commands and final engine, cached observation, application/RNG/monitor/latch state must reproduce. The report plots tracking onset beside physical lifetime and uses the unfavorable first seed if necessary. Audit validity and the >=16/20 physical-completion criterion remain separate: a valid negative diagnostic is publishable, not a motor win.
