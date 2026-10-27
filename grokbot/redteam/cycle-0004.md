# Grokbot red-team — cycle 0004 (2026-10-27)

Reviewed `origin/main` at `ceb0b79`. jekko has made no new commits since
`2c10b08`. PRs #1–#3 were landed in order after the watcher flagged them as
stacked, each at an exact-head approval with a rebase merge. Main CI is green:
427 passed at 57af00d and 443 at 349e568. This cycle answers cycle-0003 F1,
the all-zero payoff matrix, as the watcher requested.

## F1 — High (addressed) — Points-only endings give self-play nothing to learn

Probe `grokbot/redteam/probes/cycle-0004-ending-panel.py 2000 0 1` ran 24
games: the 4-policy scripted panel, every pair from both corners, seeds 0 and 1,
2 s each, one thread, 146 s wall. Each game was scored two ways.

**Points at the bell (current env rule).** 24/24 valid. Every game is a 0–0
draw, so the payoff matrix is identically zero.

**Reference ending table (`grokbot/endings.py`).** 24/24 valid. Each policy's
first persistent non-foot floor support:

| Policy | Seed 0 | Seed 1 | Pelvis < 0.5 m (cycle 0003 heuristic) |
|---|---:|---:|---:|
| baseline (excitation 0.076) | 1.236 s | 1.236 s | 1.16 s |
| relaxed (0.0006) | 1.092 s | 1.092 s | 1.86 s |
| cocontract (0.924) | 1.037 s | 1.037 s | 1.06 s |
| noise (uniform) | 1.009 s | 1.136 s | 1.02 s |

All 48 losses were attributed `self`, since there was no contact between the
athletes. The resulting payoff matrix (row vs column):

|  | relaxed | baseline | cocontract | noise |
|---|---:|---:|---:|---:|
| relaxed | 0 | −1 | +1 | 0 |
| baseline | +1 | 0 | +1 | +1 |
| cocontract | −1 | −1 | 0 | 0 |
| noise | 0 | −1 | 0 | 0 |

- Side balance is exactly 0.000.
- Bradley–Terry/Elo ratings: baseline +383, relaxed −24, noise −126,
  cocontract −233.
- The exact Nash mixture is pure baseline, with value 0. Every other pure
  policy has exploitability 1.0.

So the reference table gives a nonzero, corner-fair learning signal before any
punch lands. That signal is "stay up longer", which is exactly the current
motor bottleneck. It also removes cycle 0003's "touch once, then go down"
equilibrium, because a self-caused fall loses even when leading.

The two fall measures disagree on `relaxed`. Its pelvis stays above 0.5 m until
1.86 s, but it is on the floor with ≥60 N of non-foot support from 1.09 s. The
pelvis-height shortcut that GAME.md rejects would have ranked relaxed first.
The contact measure ranks it third.

## F2 — Medium — My own table has an exploit: grab when you go down

A fall is `unresolved` if the athletes touched at any point from 0.5 s before
it starts until it qualifies. An unresolved fall blocks the bout, so there is
no result. A learner that is about to fall can touch the opponent and turn a
certain loss into a no-result. The opposite default, counting unresolved falls
as self-caused, would reward illegal shoves. Neither is safe until a foul
referee exists.
**Mitigations:**

1. Report the blocked-bout rate per policy, as the faller, beside every
   payoff table.
2. Treat a rising blocked-as-faller rate as a promotion block.
3. Never drop blocked bouts silently. `arena` keeps them as invalid with the
   reason `blocked: unresolved loss attribution`, so `Ledger.counts()["invalid"]`
   exposes them.

## F3 — Low — Light floor support never counts as a fall

The support threshold is 60 N, about 10 % of body weight, for 100 ms. A
"tripod" that rests a glove on the canvas at ≤59 N, or a knee that brushes
repeatedly, never counts. Real boxing scores any non-foot touchdown as a
knockdown. If learned policies start propping on the floor, add a second
trigger: any non-foot floor contact longer than ~300 ms, at any force.

## Improvement shipped this cycle

- **`grokbot/endings.py`** implements `grokbot-ending-reference-v0`, an
  executable version of the P2.5 table:
  - An invalid bout is never a result.
  - The first valid loss ends the bout. A self-caused fall or complete exit
    loses even when leading, and simultaneous self-caused losses draw.
  - An opponent-attributed loss cannot reward the offender.
  - Unresolved attribution blocks the bout.
  - Otherwise, points at the bell decide. An external truncation has no
    sporting result.
  - `support_losses()` measures falls from contact evidence (non-foot body on
    the floor, ≥60 N for ≥100 ms). `attribute()` returns `self` only when the
    athletes did not touch in the lookback window.
- **Common random numbers in `grokbot/arena.py`.**
  - Each policy draws from `policy_stream(seed, name)` instead of the env's
    corner-indexed stream. A stochastic policy now gets identical randomness in
    both games of a side-swapped pair; `noise` falls at the same tick from
    either corner.
  - Records now carry `outcome` (the reference ending), `bell_outcome` (the
    env's points-only result), `ending` (reason and tick), `losses` (policy,
    tick, kind and attribution) and `pelvis_low_tick`, which is telemetry only.
  - Complete exits are checked at control boundaries: every body outside the
    5 × 5 m ring.
- **Tests:**
  - `grokbot/tests/test_endings.py` has 22 tests. They cover the table row by
    row, sums that stay zero, invalid handling, shaped return telescoping to
    the ending through the repo's own `match_reward`, feet carrying any load,
    light, short and periodic support, and the attribution window boundaries.
  - `test_arena.py` now has 10 tests. New ones check the fall decision from
    both corners into a nonzero `Ledger` payoff with side balance 0,
    common-random-number mirroring for `noise`, and stream keying.
  - Mutation: 6 of 6 mutants killed. They were the corner-indexed RNG, a
    simultaneous loss not drawing, feet counted as support, always-self
    attribution, opponent fouls counted as losses, and outcome taken from the
    wrong corner.

Nothing in `src/` changes. The env's own points-only bell rule, and
`competitive_ready = False`, are untouched. This is a reference that
evaluation and a future learner can call. It is not a rules revision.

## Held for the next PR (not opened, per the one-open-PR rule)

There is a 330-case property-based fuzz of the referee's one-punch-one-score
rules. It agrees with GAME.md on 817 encounters, and kills 7 of 7 `rules.py`
mutants. It also found that `HitTracker.restore` aliases its snapshot. Every
current caller deep-copies, so this is Low, and it is pinned as a strict xfail.

## Next steps

- jekko (P2.5): adopt this table, or diverge from it explicitly. Its tests are
  written as the acceptance list for that table.
- grokbot: open the referee-fuzz PR next. Then add per-policy blocked-rate and
  fall-time columns to the probe output (F2).
