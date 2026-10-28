# Grokbot lane status (joxing_claude)

Updated 2026-10-28 ~06:45 UTC. The full history is in `~/joxing_grokbot/LOG.md`
and `grokbot/redteam/cycle-*.md`.

## Where things stand

- **Main:** green at `508cd94`, plus this PR. **Latest tag: `v0.0.13`.**
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
| v0.0.16 | `GroupPolicy`, 138 group tones | 0/24 | 0/24 | 2.344 s | failed, not tagged |
| v0.0.17 | `GroupFeedbackPolicy`, per-group tilt gains | **7/24** | **24/24** | **2.682 s** | **criterion not met**: stance 0.29 < 0.5, so the head-to-head did not run. Not tagged. Sign test vs v0.0.16 passed (p = 1.5e-6). |

**Exploratory, not counted:** v0.0.17 vs v0.0.13 on the pinned seeds gave 14
of 24 games valid and 14 of 14 won. These are the first valid games, but
every one was a glove tap followed by the opponent's own fall (cycle 0023,
F1/F2).

## Next step: v0.0.18 (protocol `protocols/v0.0.18.json`)

- **Changes:** `GroupPDPolicy` (per-group velocity and height gains, 966
  parameters, from v0.0.17) and a hit-only strike bonus. 48 × 12 ES, seed 8,
  at nice 19.
- **Level-up** requires both the **motor gate** and the **Elo gate**. The Elo
  gate is against the frozen v0.0.16 champion, on seeds 3000–3023 with both
  sides, and always runs. It needs all of:
  - valid games ≥ 50 %;
  - a paired lower bound > 0, which is the same as an Elo lower bound > 0;
  - at least half of the candidate's wins including head or torso hits.

  Only a level-up gets the tag `v0.0.18`.
- **Probe:** `probes/cycle-0023-group-pd-gate.py`. The run's tree is checked
  against the merged tree.

## Rules this lane follows

- Only one grokbot PR open at a time. Each PR branches off fresh main.
- Landing: neverhumanbot approves the exact head via `gh-role`, then
  `gh pr merge --rebase --match-head-commit`. Re-verify main immediately
  before approving and before merging.
- Preregister and push the protocol before reading any result.
- At most 4 one-thread workers, at `nice -n 19`. Yield if loadavg ≥ 86.
- Write only under `grokbot/` and `.github/`. Edit `CHANGELOG.md` only when
  the owner asks.
