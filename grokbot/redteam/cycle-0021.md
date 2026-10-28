# Grokbot red-team — cycle 0021 / v0.0.16 (2026-10-28): first real motor progress, gate still missed

Reviewed `origin/main` at `77956c5`. jekko has made no commits since
`6b6ba04`. The v0.0.16 preregistration landed on main in #20 before this
run.

## v0.0.16 result (protocol `1ff40c3d…`): preregistered gate NOT passed, not tagged

The candidate was `GroupPolicy`: 138 per-DynSyn-group tones, plus regional
posture feedback, plus a punch, for 284 parameters. It was initialized from
gen4 and trained with 32 iterations of 12 antithetic ES pairs on the solo
fitness, stand fraction plus strike. The gate, seeds, opponent and horizon
were identical to v0.0.15. The run took 2,297 s on 4 nice-10 workers.

| | Stance pass | Reach pass | Median stand | Stand range |
|---|---:|---:|---:|---|
| gen4 baseline | 0/24 | 0/24 | 1.464 s | 1.44–1.63 s |
| v0.0.15 candidate (`PunchPolicy`) | 0/24 | 0/24 | 1.349 s | 1.33–1.64 s |
| **v0.0.16 candidate (`GroupPolicy`)** | **0/24** | **0/24** | **2.344 s** | **2.14–2.77 s** |

- **Primary gate:** stance ≥ 0.5 and reach ≥ 0.5. **Failed.** No trial
  stood the full 3 s, and no gate trial landed a strike.
- **Preregistered secondary:** Hoeffding lower bound on the paired gain over
  gen4. The mean gain is **+0.870 s**, but the lower bound is −0.629, so
  this is **not met**. The bound assumes a ±3 s range over 24 pairs and is
  very conservative.
- **Exploratory, not preregistered:** the candidate stands longer than gen4
  in **24 of 24** paired trials (one-sided exact sign test p = 6.0×10⁻⁸),
  and longer than the v0.0.15 candidate in 24 of 24.
- **Training curve:** mean stand time rose steadily, from 1.31 s at
  iteration 0 to 1.91–2.07 s over iterations 25–31.
- **Training strikes:** in 5 iterations (13, 23, 25, 29 and 30), one of the
  24 trials landed a glove strike. **These are the first strikes produced by
  any policy in this project.**
- **Engine faults:** 1, quarantined.

**v0.0.16 is not a level-up and is not tagged.** Pre-declared reading 2
applies: *capacity helps, but stance still falls short of 3 s; scale up*.

## Finding

### F1 — High (positive) — The stance ceiling was controller capacity, not physics

Splitting 10 regional tones into 138 group tones, from the same starting
commands and with the same fitness, raised standing time by about 0.9 s,
from a plateau that four league versions and one solo gate had never moved.
This confirms the cycle-0019 root cause from the other side. The bottleneck
is controller capacity on the stance, and the next version should buy more
of it rather than change rules or spawns.

## v0.0.17 preregistration (this PR, before any run)

The protocol is `grokbot/redteam/protocols/v0.0.17.json`. This is a bigger,
specific change rather than a rerun (owner 04:25), and the level-up
definition follows owner 03:13 and 03:14.

| Item | Registered value |
|---|---|
| Controller | **`GroupFeedbackPolicy`**: the v0.0.16 `GroupPolicy` plus **per-group posture feedback**, i.e. gains on the 6 pelvis and trunk tilt features for each of the 138 groups. That adds 828 parameters, all initialized to 0, so iteration 0 is bit-identical to the v0.0.16 candidate. |
| Training | 48 iterations of 12 ES pairs, seed 7, on the same solo fitness |
| Level-up | Tagged only if **both** criteria pass |
| Criterion 1 | The **primary motor gate**, unchanged: stance ≥ 0.5 and reach ≥ 0.5 over the same 24 seeds |
| Criterion 2 | **Head-to-head against the frozen v0.0.13 champion** at 0.50 m under anti-stalling, 12 side-swapped noisy seed pairs: **valid games > 0 and a paired 95 % lower bound > 0** |
| Secondary | Exact sign test against the v0.0.16 candidate, p < 0.05 |

The policy class is implemented in the PR that runs it, and must match this
protocol.

## What this PR adds

The v0.0.16 summary data, `data/cycle-0020-group-gate-summary.json`, holds
the per-trial stand and strike ticks, the training history, the final
parameters, the exploratory sign test, and the full-report SHA-256. The PR
also adds the v0.0.17 protocol (sha256 `24fdd85b…`), `grokbot/STATUS.md`,
and this report. The v0.0.17 probe and `GroupFeedbackPolicy` land in the
next PR, before the run.
