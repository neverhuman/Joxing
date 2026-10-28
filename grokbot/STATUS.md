# Grokbot lane status (joxing_claude)

Updated 2026-10-28 ~07:00 UTC. The full history is in `~/joxing_grokbot/LOG.md`
and `grokbot/redteam/cycle-*.md`.

## Where things stand

- **Main:** green at `1dc96e9`, plus this PR. **Latest tag: `v0.0.17`**, a pre-release labelled "criterion not met" (owner request). No tag so far marks a passed preregistered level-up.
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

| Version | Change | Stance | Reach | Head/torso hits | Median stand | Result |
|---|---|---|---|---|---|---|
| v0.0.15 | `PunchPolicy`, 10 regional tones | 0/24 | 0/24 | — | 1.349 s (gen4 1.464 s) | failed; the rerun reproduced it exactly |
| v0.0.16 | `GroupPolicy`, 138 group tones | 0/24 | 0/24 | — | 2.344 s | failed; not tagged |
| v0.0.17 | + per-group tilt gains | **7/24** | 24/24 | 0/24 | **2.682 s** | criterion not met; pre-release at owner request |
| v0.0.18 | + per-group velocity and height gains, with a hit-only bonus | 1/24 | 24/24 | **24/24** | 2.314 s | registered level-up not met; not tagged |

**v0.0.18 head-to-heads** (48 games each, 0.5 m, anti-stalling):

| Opponent | Status | Valid | W–L | Elo | Elo lower95 | Gate |
|---|---|---|---|---|---|---|
| frozen v0.0.16 | registered | 44/48 | 13–31 | −151 | −579 | failed |
| frozen v0.0.13 | addendum (owner 06:28), preregistered | 48/48 | 48–0, all with hits | **+669** | **+191** | passed |

Cycle 0024: hitting costs balance, so v0.0.18 falls first against a standing
opponent.

## Next step: v0.0.19 (recommended, not yet preregistered)

- **One change:** a "hit and stay up" fitness, where the hit bonus counts
  only if the athlete is still up 0.5 s after the hit. Train from v0.0.18
  with the same policy class.
- **Report:** the motor gate, the Elo gate vs v0.0.16 and the Elo gate vs
  v0.0.13. Preregister and push before running, and tag only on the
  registered level-up.
- If stance does not recover, the next lever is recurrent control or jekko's
  pretrained bridge.

## Rules this lane follows

- Only one grokbot PR open at a time. Each PR branches off fresh main.
- Landing: neverhumanbot approves the exact head via `gh-role`, then
  `gh pr merge --rebase --match-head-commit`. Re-verify main immediately
  before approving and before merging.
- Preregister and push the protocol before reading any result.
- At most 4 one-thread workers, at `nice -n 19`. Yield if loadavg ≥ 86.
- Write only under `grokbot/` and `.github/`. Edit `CHANGELOG.md` only when
  the owner asks.
