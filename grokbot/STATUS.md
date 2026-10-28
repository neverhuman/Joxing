# Grokbot lane status (joxing_claude)

Updated 2026-10-28 ~19:20 UTC. The full history is in `~/joxing_grokbot/LOG.md`
and `grokbot/redteam/cycle-*.md`.

## Where things stand

- **Main:** green at `2f3c9ec`, plus this PR. **Latest tag: `v0.0.19`**, the first passed preregistered level-up, released as a pre-release diagnostic.
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
| v0.0.20 | two-player self-play league (outcome only) | 16/24 | 24/24 | 3.000 s | **honest null**: no promotion, champion = v0.0.19; primary 3/48 valid; not tagged |
| v0.0.21 | training opponent = frozen v0.0.19 | 1/24 | 10/24 | 1.856 s | **CRITERION_NOT_MET**: 48/48 valid but 4–2–42 draws (referee points 0–0); NOT monotone (@48 lost 0–20 to @16); vs v0.0.13 46/48 blocked; not tagged |
| v0.0.22 | knockdown objective | 0/24 | 24/24 | 1.816 s | **primary PASSED** (46–0–0 vs v0.0.19), but the hard gate failed (36/48 valid vs v0.0.13, 12 blocked) and monotonicity was unmeasurable (@16 vs v0.0.19 38/48 valid); level-up NOT met; not tagged |
| v0.0.23 | clean-knockdown objective (bonus never fired) | 2/24 | 23/24 | 1.923 s | **VALIDITY_FAILED** (primary 5/48 valid); hard gate 14/48; not tagged. Knockdowns of v0.0.19 are attributed `self` (fall 1–2 s after the hit), not `knockdown` |

**v0.0.19 head-to-heads** (48 games each):

| Opponent | Status | W–L | Wins with hits | Elo | Elo lower95 |
|---|---|---|---|---|---|
| frozen v0.0.13 | **primary** | 48–0 | 48 | +669 | +191 |
| frozen v0.0.16 | secondary | 47–1 | 21 | +669 | +172 |

v0.0.18 went 15–29 against v0.0.16 (Elo −115). The plateau diagnosis is in
cycle-0026.md.

## Ladder, v0.0.11 to v0.0.20 (cycle 0028; Elo with v0.0.11 = 0)

| Version | Elo |
|---|---:|
| v0.0.13 / v0.0.14 | +25 |
| v0.0.15 | +25 |
| v0.0.16 | +586 |
| v0.0.17 | +520 |
| v0.0.18 | +810 |
| **v0.0.19 / v0.0.20** | **+1,012** (+885 to +1,164) |

- **0 intransitive triads.**
- **Degeneracy:** the v0.0.19 mirror has 0 strikes (no-contests at the
  bell), and v0.0.18 vs v0.0.19 is 92 % blocked (body collisions at 0.5 m).
  The self-play league therefore had a valid share of only 23–31 %, which
  fell over training.

## Next step: v0.0.24 (its own preregistration PR)

- **Smoke gate first:** a short run at the base policy must produce the
  bonus event in at least 10 % of trials before the full run is allowed
  (owner 18:40).
- **One lever: a valid-knockdown bonus.** It pays `knocked_down` when the
  fall is not `unresolved`, so it pays for valid `self` falls and excludes
  only blocked ones. It would have fired in 22/24 replayed trials.
- **Gates as in v0.0.22/23:**
  - primary against frozen v0.0.19: at least 40/48 valid, then at least
    24/48 decisive, then b/c;
  - hard gate: at least 40/48 valid against v0.0.13;
  - monotonicity;
  - the per-seat split.

## Rules this lane follows

- Only one grokbot PR open at a time. Each PR branches off fresh main.
- Landing: neverhumanbot approves the exact head via `gh-role`, then
  `gh pr merge --rebase --match-head-commit`. Re-verify main immediately
  before approving and before merging.
- Preregister and push the protocol before reading any result.
- At most 4 one-thread workers, at `nice -n 19`. Yield if loadavg ≥ 86.
- Write only under `grokbot/` and `.github/`. Edit `CHANGELOG.md` only when
  the owner asks.
