# Current mission: v0.0.11 connected two-player loop and recording

**Active implementation, formal v11physics0.** v0.0.10is concluded with a negative
motor result:16,384transitions,896optimizerupdates,32/32valid evaluations and2/2
exact replays; both learning gates failed (−0.06625s/+0.01625s paired survival,
1/8five-second survivors each). Independent all-row/raw audit passed; all35
recorded processes exited, owned scanempty/sourceunchanged10:58:23UTC.
[Results](V0.0.10_RESULTS.md). Public raw/tag publication is the remaining v10
administrative step until artifacts/receipts/v0.0.10-publication.json exists.
Do not rerun or enlarge the walking pilot.

Integration owner: this chat, no agents/concurrent writers/other-chat messages.
Goal: first reproducible shared-world diagnostic footage, then qualified guard,
punch/miss/block/recovery exchanges and historical self-play. Native valid bouts0.
[Recorded-bout milestone](BOUT_RECORDING_MILESTONE.md),
[worker-ready v11mission](V0.0.11_MISSION.md),
[integrated route](SELF_PLAY_EXECUTION_PLAN.md).

## Read-first and exact next work

1. AGENTS and /Users/bentaylor/.codex/RTK.md; README/GAME/SPEC_REVIEW;
   STATUS/IMPLEMENTATION; V0.0.10_RESULTS; V0.0.11_MISSION;
   BOUT_RECORDING_MILESTONE; experiments/v0.0.11/README.md.
2. src/joxing/{scene,physics,rules,bout_recording,history,provenance}.py and
   bodies/{base,native}.py plus their tests. Supplied references are proposals.
3. Implement observations.py before policy use: named hashed own proprioception,
   opponent visible body pose/velocity, ego yaw frame preserving gravity/height,
   public clock/points/ring and previous applied command. Opponent activation,
   commands, goals, RNG, hidden state and bundle IDs are forbidden actor inputs.
   BodyMeasurements is private readout, not a safe ActorObservation. Extend the
   adapter explicitly for needed rotation/angular/torso fields; test real
   asymmetric states, yaw/translation and no alias/private leakage.
4. Wire existing TimedHistory into causal samples. Implement bounded timestamped
   action queues with decision<=apply, game/player ownership, zero-delay oracle,
   one/two-tick delay/reorder/drop/stale/reset tests and complete snapshot state.
5. Implement env.py/referee.py/replay.py in the order and APIs of V0.0.11_MISSION:
   freeze both observations, validate both700commands, apply legacy sigmoid once,
   advance one shared MjData, collect every substep, stage both sides' contacts,
   commit all events then any shared ending. No pose/root assistance in playback
   generation. Keep unresolved scene/endings/motor gates explicit; no competition.
6. Connect BoutRecorder to completed shared states and retain first registered
   scenario tape; full contact/application evidence separately. Render with
   render_bout_recording.py, decode/check duration/HUD/checksums/0rendersteps.
   Never present a static frame or assisted/scripted diagnostic as learned boxing.
7. Implement tests then finite qualify_boxing_env.py/audit_boxing_env.py protocol
   enumerating exact trajectories/replay tails. Freeze/public-push before formal
   outcomes. Target<=20kplannedphysics; hardmax120kphysics/900s/256MiB/2simulation
   workers/1thread, firstlimit, subject to a new registered machine protocol.
   Inspect fresh capacity, no unknown owned job;8GiBRAM/10GiBdisk floors twiceatstart.
   Preserve all700muscles/corner, no third_party or old source/protocol edits.

## Implemented foundations and validation

NativeBodyAdapter measures85joint/700activation states and six body positions/
linear velocities per corner from shared state; owned immutable copies. Passive
recording stores typed complete dual engine state, hashes and clock/counts;
first format diagnostic-only/pointsunavailable. Static video smoke attempt01
failed an image-format boundary, fixedattempt02decoded one1024x576frame with
both athletes and proper diagnostic label,0physics; it is not a bout.

TimedHistory now isolates game/player/schema, accepts delay/reordering and returns
only received past samples with validity and age masks; snapshot/restore rejects
foreign/corrupt packets. Threehistory+body+recording fixtures6passed5.64s; Ruffpass.
It stores filtered features only. Observation assembly, action queues, shared loop,
contact referee and application replay remain unfinished.

Current development runtime: experiments/v0.0.11/runtime, CPUlocked versions
inherited from v10 with new package metadata. Full current development regression passed201tests188.09s;
receipt artifacts/v0.0.11/development/full-tests.json.

```text
rtk proxy uv run --project experiments/v0.0.11/runtime --locked pytest -q
rtk proxy uvx ruff check src scripts tests
```

Source-frozen v10reproduction uses b1d1690 and its own v10runtime/protocol;
current main intentionally fails that old source guard. No old runtime changed.
Compact v10outcomes/rawmanifest/exit/model/optimizer receipts in artifacts/v0.0.10;
all668D2files independently SHA-verified after archive copy. Historical mission
entries preserved in docs/history/MISSION_BEFORE_V11_2026-10-27.md.

## Heartbeat and handoff

Inspect actual owned PIDs/startticks, fresh events/counters, capacity, faults,
tests and Git. No active native simulation after v10; the next concrete task is
observation schema and action queues above. Healthy future job: independent
work only, no executing-source mutation/duplicate. Stop on first drift, unknown
owner, nonfinite, assistance, leakage, contact/replay mismatch or cap; retain every
invalid/incomplete/not_run row and lower/upper physics bounds. Finish independent
all-attempt audit, source/process exits, README/docs/book/public push and tag
only concluded scope. Max16aggregate owned simulation workers globally, tighter
v11cap2. Stay quiet for unchanged healthy state; notify meaningful changes only.
