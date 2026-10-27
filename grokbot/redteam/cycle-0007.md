# Grokbot red-team — cycle 0007 (2026-10-27)

Reviewed `origin/main` at `1dc5233`. jekko has made no new commits since
`c9a2b96`, the shared-motor physical proof preregistration, which cycle 0006
covered. PR #6 landed on an unchanged base. This cycle fixes cycle-0006 F1:
the reference ending table blocked every bout as soon as the athletes touched,
including knockdowns from legal blows.

## The change: attribution v1 (`grokbot-ending-reference-v1`)

Every force-producing contact between the athletes in the 0.5 s lookback is
classified with the referee's own constants (`joxing.rules.Rules`):

| Contact | Class | A fall after it is |
|---|---|---|
| None | — | `self`: the faller loses |
| Glove on head/torso/guard/glove, run starting at ≥0.5 m/s, first 50 ms | strike | `knockdown`: the faller loses |
| Glove contact held past 50 ms (holding) | blocking | `unresolved`: bout blocked |
| Slow glove push (<0.5 m/s at run start) | blocking | `unresolved` |
| Glove on `native_non_scoring` (back, below belt, legs; possible foul) | blocking | `unresolved` |
| Body on body (clinch, lean, shove) | blocking | `unresolved` |

A run continues across gaps of up to 8 ticks, matching the referee's 8 ms
release. Blocking contact always wins over strikes in the same window.

Two consequences:

- **Knocking the opponent down now pays.** A fall after only legal strikes is
  the faller's loss.
- **"Grab while falling" (cycle-0004 F2) is narrowed.** The faller's own fast
  strike on the opponent no longer buys a no-result. The faller can still lean
  bodily, or push slowly, to force a block. That residual is what
  `policy_summary` now reports, through `blocked_as_faller` and the new
  `knocked_down` column.

## Evidence

- **Separated starts** (`cycle-0004-ending-panel.py 2000 0 1`, 24 games): the
  results are **identical to cycle 0004**. The payoff matrix, Elo (baseline
  +383, relaxed −24, noise −126, cocontract −233), pure-baseline Nash and side
  balance 0.000 are unchanged. There is no contact, so every fall is `self`.
- **Collision-stress reset** (`cycle-0006-engagement-range.py
  close-overlap-collision-probe-v1 2000 0 1`, 24 games): still **24/24
  blocked**, which is intended. One game breaks down as follows: all 3,311
  inter-athlete contact points are `native_non_scoring ↔ native_non_scoring`
  from tick 0. That gives 656 blocking ticks and 0 strike ticks, because the
  bodies start overlapping. v1 does not turn stress collisions into results.

A real knockdown can't be demonstrated in physics yet, because no controller
lands a punch. The strike path is covered by synthetic contact evidence.

## Findings

### F1 — Medium (carried to jekko) — No registered reset at boxing range

Neither the 1.6 m keyframe, where the athletes never touch and everyone falls
first, nor the 0.2 m overlap, which is body-on-body from tick 0, is a boxing
start. Until P2/P6 registers a facing start at about 0.9–1.1 m with a stable
guard, the only learnable signal is "stay up longer".

### F2 — Low — The holding window is 50 ms in both the referee and attribution

Contact held more than 50 ms after a strike counts as holding. A legitimate
heavy punch with follow-through contact longer than 50 ms followed by a fall
would be blocked. This errs on the conservative side, but it should be
revisited when real strike data exists.

## Improvement shipped this cycle

- **`grokbot/endings.py` v1.** It adds `classify_contacts()` and the
  `knockdown` attribution. `attribute()` now takes `(strikes, blocking)`, and
  the decision reason reads "A knocked down fall".
- **`grokbot/arena.py`** uses the classified contacts and adds a
  `knocked_down` column to `policy_summary`.
- **Tests: 33 ending tests** (previously 22) and **4 summary checks**:
  - A knockdown table row and a simultaneous knockdown + self-fall that draws.
  - Strike classification in both geom orders.
  - Glove-on-glove, and the faller's own strike.
  - Slow push, holding past the window, glove to the back, and body on body.
  - The exact gap-rearm boundary: 7 empty ticks continue the run, 8 release it.
  - Blocking wins over strikes.
- **Mutation:** 7 of 7 killed. They were: approach ignored, window ignored,
  `native_non_scoring` counted as a target, gap `>`→`>=`, strikes overriding
  blocking, the knockdown reason dropped, and `knocked_down` not counted.

## Next steps

- jekko: register a boxing-range facing reset (F1). If the reference table is
  adopted, port `classify_contacts` next to the referee, which already computes
  approach speed per touch.
- grokbot: extend `classify_contacts` with the attacker's active glove speed
  relative to its own trunk, the referee's `active_speed`, so a defender
  walking into a still glove is not a strike.
