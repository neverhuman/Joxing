# Grokbot lane status (joxing_claude)

Updated 2026-10-28 ~06:10 UTC. The full history is in `~/joxing_grokbot/LOG.md`
and `grokbot/redteam/cycle-*.md`.

## Where things stand

- **Main:** green at `9c22f2c`, plus this PR. **Latest tag: `v0.0.13`.**
  Tags `v0.0.11`, `v0.0.12` and `v0.0.13` have GitHub pre-releases that say
  plainly which criteria failed. Tagging rule (owner 01:50): from v0.0.14 on,
  a version is tagged **only if its preregistered criterion passes**.
- **Infrastructure on main:** CI; the league/Elo/Nash oracle; the
  side-swapped arena with common random numbers; the reference ending
  (attribution v2 and anti-stalling); the promotion gate; positive controls;
  the referee fuzzes; the P4 sampler; the self-play league; the
  degenerate-league guard (`DegenerateLeagueError`, with CI tests); and the
  solo motor gate (`grokbot/motor_gate.py`).
- **Root cause of invalid games** (cycle 0019): athletes fall at about
  1.3–1.65 s, backward; reach is about 0.2 m; bodies collide at ≤ 0.5 m. That
  gave 0/2,772 valid league games in v0.0.12–v0.0.14.

## Gate history (all preregistered)

| Version | Change | Stance | Reach | Median stand | Result |
|---|---|---|---|---|---|
| v0.0.15 | `PunchPolicy`, 10 regional tones | 0/24 | 0/24 | 1.349 s (gen4 1.464 s) | failed; the rerun reproduced it exactly |
| v0.0.16 (gate-0020/0021) | **`GroupPolicy`, 138 group tones** | 0/24 | 0/24 | **2.344 s** | failed, not tagged. Stand time better than gen4 in 24/24 trials (exploratory sign test p = 6e-8). First training strikes ever, in 5 iterations. |

## v0.0.16 preregistered criteria (protocol `1ff40c3d…`)

- **Primary:** stance pass ≥ 0.5 **and** reach pass ≥ 0.5 over 24 noisy solo
  trials (seeds 1000–1023), at the facing 0.5 m spawn, against the frozen
  gen4 posture, over 3 s. Not met.
- **Secondary:** Hoeffding lower bound on the paired stand gain over gen4.
  Not met: the mean is +0.87 s, the lower bound −0.63.

## Next step: v0.0.17 (protocol `850ec810…`, on main since #21)

1. Done in cycle 0022: `GroupFeedbackPolicy` (`GroupPolicy` plus 138 × 6
   per-group tilt gains, 828 parameters, zero-initialized on the v0.0.16
   final parameters), with tests and the probe
   `probes/cycle-0022-group-feedback-gate.py`.
2. Run 48 × 12 ES with seed 7 on the same gate (in progress or next).
3. **Level-up** requires the motor gate to pass **and** the head-to-head
   against the frozen v0.0.13 champion (seeds 2000–2011, 0.5 m,
   anti-stalling) to have valid games > 0 and a paired lower bound > 0.
   Note: the champion falls at about 1.7 s, so a valid game needs a strike
   before then (cycle 0022, F1). Only a level-up gets the tag `v0.0.17`.
4. If there is no stand gain, the next lever is recurrent control or jekko's
   pretrained bridge.

## Rules this lane follows

- Only one grokbot PR open at a time. Each PR branches off fresh main.
- Landing: neverhumanbot approves the exact head via `gh-role`, then
  `gh pr merge --rebase --match-head-commit`. Re-verify main immediately
  before approving and before merging.
- Preregister and push the protocol before reading any result.
- At most 4 one-thread workers, at `nice -n 10`.
- Write only under `grokbot/` and `.github/`. Edit `CHANGELOG.md` only when
  the owner asks.
