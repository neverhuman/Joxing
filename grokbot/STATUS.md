# Grokbot lane status (joxing_claude)

Updated 2026-10-29 ~10:20 UTC. The full history is in `~/joxing_grokbot/LOG.md`
and `grokbot/redteam/cycle-*.md`.

## Where things stand

- **Main:** `aeaced5` (jekko H10 work), plus this PR. **Latest tag: `v0.0.19`**, the first passed preregistered level-up, released as a pre-release diagnostic.
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
| v0.0.29 | v0.0.28 recipe with the base on half of the ES pairs | 2/24 | 24/24 | 2.32 s | primary (44–0) and hard gate (46/48 valid, 46–0) PASSED; @48 beat @16 47–1; @16 lost **16–32** to base (v0.0.28: 2–46) → monotonicity FAILED; not tagged |
| v0.0.28 | v0.0.26 recipe + frozen v0.0.22 base as a third training opponent | 1/24 | 24/24 | 1.98 s | primary (44–0) and hard gate (48–0) PASSED; @48 beat @16 46–2; @16 lost **2–46** to base → monotonicity FAILED; not tagged (post hoc @48 beat base 43–5, non-gating) |
| v0.0.27 | v0.0.26 recipe with lr 0.02 | 8/24 | 24/24 | 2.68 s | primary and hard gate PASSED (48–0 each); monotonicity FAILED (@16 lost **0–48** to base; @48 lost 11–37 to @16); not tagged |
| v0.0.26 | contact penalty trained vs both v0.0.19 and v0.0.13 | 11/24 | 24/24 | 2.82 s | primary PASSED (46–0), **hard gate PASSED (48/48 vs v0.0.13)**, monotonicity FAILED (@16 lost 2–40 to base; @48 vs @16 24–24); not tagged |
| v0.0.25 | graded contact penalty from v0.0.22 (smoke passed) | 0/24 | 22/24 | 1.435 s | **primary PASSED** (48–0 vs v0.0.19, contact 380 → 113), but the hard gate scored 0/48 valid vs v0.0.13 (all blocked) and monotonicity failed (@16 lost 4–44 to base); not tagged |
| v0.0.24 | valid-knockdown bonus from v0.0.22 (smoke-gated) | — | — | — | **SMOKE_FAILED**: bonus fired 23/24, but base head-to-head 10/12 valid (< 90%); no training; not tagged |
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

## Next step: v0.0.30 (its own preregistration PR)

- **One lever: a curriculum.** The frozen v0.0.22 base is the only training
  opponent for iterations 0–15, then v0.0.29's 4-slot rotation runs for
  iterations 16–47.
- **Why:** @16 vs base follows the base's update share. A share of 0.25
  gave 2–46; 0.44 gave 16–32.

## Rules this lane follows

- Only one grokbot PR open at a time. Each PR branches off fresh main.
- Landing: neverhumanbot approves the exact head via `gh-role`, then
  `gh pr merge --rebase --match-head-commit`. Re-verify main immediately
  before approving and before merging.
- Preregister and push the protocol before reading any result.
- At most 4 one-thread workers, at `nice -n 19`. Yield if loadavg ≥ 86.
- Write only under `grokbot/` and `.github/`. Edit `CHANGELOG.md` only when
  the owner asks.
