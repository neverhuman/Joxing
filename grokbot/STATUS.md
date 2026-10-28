# Grokbot lane status (joxing_claude)

Updated 2026-10-28 ~09:50 UTC. The full history is in `~/joxing_grokbot/LOG.md`
and `grokbot/redteam/cycle-*.md`.

## Where things stand

- **Main:** green at `4ffbd91`, plus this PR. **`v0.0.19` will be tagged on this PR's merge: the first passed preregistered level-up.** The previous tag, `v0.0.17`, is labelled "criterion not met".
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

| Version | Change | Stance | Head/torso hits | Median stand | Result |
|---|---|---|---|---|---|
| v0.0.15 | `PunchPolicy`, 10 regional tones | 0/24 | — | 1.349 s | failed |
| v0.0.16 | `GroupPolicy`, 138 group tones | 0/24 | — | 2.344 s | failed |
| v0.0.17 | + per-group tilt gains | 7/24 | 0/24 (glove taps) | 2.682 s | criterion not met; pre-release |
| v0.0.18 | + velocity and height gains, hit-only bonus | 1/24 | 24/24 | 2.314 s | not met; untagged |
| **v0.0.19** | **hit-and-stay-up fitness** | **16/24** | **24/24** | **3.000 s** | **PASSED: level-up, tagged** |

**v0.0.19 head-to-heads** (48 games each):

| Opponent | Status | W–L | Wins with hits | Elo | Elo lower95 |
|---|---|---|---|---|---|
| frozen v0.0.13 | **primary** | 48–0 | 48 | +669 | +191 |
| frozen v0.0.16 | secondary | 47–1 | 21 | +669 | +172 |

v0.0.18 went 15–29 against v0.0.16 (Elo −115). The plateau diagnosis is in
cycle-0026.md.

## Next step: v0.0.20 — one lever, more self-play games

- A two-player self-play league seeded from v0.0.19, with the same model and
  fitness. Frozen-opponent sampling uses the P4 sampler; the degenerate
  guard is on.
- **Primary:** the a/b/c gate against the frozen v0.0.19 on new seeds.
- Preregister and land the protocol before running. Run at nice 19 with
  loadavg below 86.

## Rules this lane follows

- Only one grokbot PR open at a time. Each PR branches off fresh main.
- Landing: neverhumanbot approves the exact head via `gh-role`, then
  `gh pr merge --rebase --match-head-commit`. Re-verify main immediately
  before approving and before merging.
- Preregister and push the protocol before reading any result.
- At most 4 one-thread workers, at `nice -n 19`. Yield if loadavg ≥ 86.
- Write only under `grokbot/` and `.github/`. Edit `CHANGELOG.md` only when
  the owner asks.
