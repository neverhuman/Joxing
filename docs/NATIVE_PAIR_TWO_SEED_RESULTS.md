# Native paired PPO runs and replays; the boxing motor objective failed

Two intact700-muscle humanoids trained on xbabe4 with causal four-frame own/
opponent observations and simultaneous muscle actions. Two seeds each completed
1,024joint transitions: **4,096actor samples,16PPOupdates,256optimizer minibatch
steps and22completed motor-training episodes** in total. These are motor episodes,
not qualified boxing games. The frozen pretrained walking teacher did not change.

The independent post-task audit reconstructed every700-channel command per body,
replayed all2,048training transitions and all16fixed evaluations bit-for-bit,
and recomputed PPOvalues/log probabilities with maximum error0.0. That establishes
actual native learning execution and reproducibility. It does not establish skill.

Every evaluation scored **0–0**. None completed a guard/reach/return cycle or
survived the full three-second horizon. The figures show the athletes walking
into close body contact with their hands low, then leaning/falling. Both trained
policies remain unsuitable for competitive boxing.

| Training seed/corner | Target-error improvement | Mean survival change | Cycle starts /4 |3s starts /4 | Original motor gate |
|---|---:|---:|---:|---:|---|
|761/A|−12.62mm|+23.00ms|0|0|Failed|
|761/B|+3.13mm|+9.25ms|0|0|Failed|
|983/A|−4.91mm|+12.00ms|0|0|Failed|
|983/B|+1.48mm|+15.50ms|0|0|Failed|

The registered gate required at least50mm target-error improvement, no more than
100ms mean survival loss, a full cycle on at least2/4starts and three-second
survival on at least3/4starts, for **each** seed/corner. Small mean survival
changes above are not evidence of a robust learning gain. No thresholds changed.

## Complete fixed recordings

All frames of the first registered reset71000are retained, including failures.
These are motor evaluations, not working bouts. No pose assistance was applied
during physics. Export only reads archived states into a scratch renderer.

| Seed | Before training | After training |
|---|---|---|
|761|[Before](../artifacts/v0.0.11/native-pilot-posttask-01/761-before/before.mp4)|[After](../artifacts/v0.0.11/native-pilot-posttask-01/761-after/after.mp4)|
|983|[Before](../artifacts/v0.0.11/native-pilot-posttask-01/983-before/before.mp4)|[After](../artifacts/v0.0.11/native-pilot-posttask-01/983-after/after.mp4)|

The clips contain80/82/80/66frames at50fps; encoded durations1.60/1.64/1.60/1.32s
include the initial frame. Actual terminal physics times are1.58/1.62/1.58/1.30s.
First/middle/last frames of all four were inspected; container frame counts and
file hashes were independently checked. No convincing punch or guard is shown.

## Keep the failed attempt and the successful audit separate

Original native attempt03remains **invalid**: its first800×448recording hit the
30s stage budget, stopping the pipeline before three other clips and its audit.
The owner also recorded a final ffmpeg retirement tail; later PID/start checks
confirmed every owned process exited. No lifecycle promotion follows from that.

H32was a separately registered read-only audit/export task, with no new training
or old-clock extension. Its original-data manifest remained unchanged. All five
stages passed in181.135s /540s; replay140.706s /300s,61700NEWphysics intervals;
peak sampled owned RSS2,425,913,344B /8GiB. Four compact512×288exports took
5.64–5.78s each under the SAME30s clip budget. Tendons, shadows and reflections
were hidden for VISUAL rendering only; all1400physical muscles remained present.
All11observed post-task identities exited. The full audit/media raw output was
under1MiB; weights and native tapes remain outside Git.

Native primary61937 + post-task replay61700 =123637entered intervals, plus the
separate2048source-fixture reserve(not a measured count). Original source gate:
99fresh cases pass. Two earlier local launches deferred on CPUload BEFOREsource/
physics; both sealed roots and both seeds' not_run rows remain in Git. The spare
Mac runtime was copied into an isolated project environment:22554regular files
matched exactly, with the same teacher/normalizer/reference. No host services,
drivers, system packages, unrelated jobs or upstream sources changed.

[Original full counts/faults/hash manifest](../artifacts/v0.0.11/native-pair-motor-pilot-03/summary.json) ·
[H32audit/export receipt](../artifacts/v0.0.11/native-pilot-posttask-01/summary.json) ·
[Independent replay](../artifacts/v0.0.11/native-pilot-posttask-01/audit/independent-audit.json).

The runtime profile is explicitly ARM64Mac/Python3.12.3/NumPy2.5.4/Torch2.14.1/
SB32.9.2/MuJoCo3.8.0. No Linux or cross-host arithmetic, native100ms retirement,
RFjitter, historical-opponent league, force calibration or competitive gate is
promoted. The next question is [pretrained stopping/stance and interaction control](NATIVE_STANCE_REFERENCE_MISSION.md),
not a larger unchanged PPOsweep. Wholev0.0.11 remains untagged.
