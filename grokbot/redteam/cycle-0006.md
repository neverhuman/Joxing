# Grokbot red-team — cycle 0006 (2026-10-27)

Reviewed `origin/main` at `99d85f6`. jekko has made no new commits since
`bc7f6b5`. Main CI was green at `64a34d3` (486 passed). PR #5 landed on an
unchanged base, after main was re-verified both before the approval and
before the merge.

## F1 — High — The reference ending table gives no signal once athletes touch

Probe: `grokbot/redteam/probes/cycle-0006-engagement-range.py
close-overlap-collision-probe-v1 2000 0 1`. It runs the scripted round robin,
with every pair from both corners, for seeds 0 and 1. The reset puts the
pelvises 0.2 m apart. This is jekko's registered collision-stress reset, and
the only registered scenario where the athletes touch. It took 165 s of wall
time on one thread.

| Policy | Attempted | W/D/L | Blocked as faller | Blocked, other | Median first loss |
|---|---:|---|---:|---:|---:|
| relaxed | 12 | 0/0/0 | 12 (rate 1.00) | 0 | 0.804 s |
| noise | 12 | 0/0/0 | 8 (0.67) | 4 | 1.016 s |
| cocontract | 12 | 0/0/0 | 4 (0.33) | 8 | 1.086 s |
| baseline | 12 | 0/0/0 | 0 (0.00) | 12 | 1.154 s |

All 24/24 bouts are `blocked: unresolved loss attribution`, so there are 0
valid games and the payoff matrix is all unknown. The faller ordering is still
perfectly clear (relaxed, then noise, then cocontract, then baseline), and it
matches the separated-start ordering from cycle 0004 except for relaxed.

This reset is a collision stress test with sustained body-on-body leaning, and
blocking it is defensible. The real problem is the rule behind it: any
inter-actor contact within 0.5 s blocks the bout. In an actual exchange, a
**knockdown caused by a legal glove blow** would also be blocked, so a
learner gains nothing by knocking the opponent down. A learner about to fall
gains a no-result by touching the opponent with anything, including its own
glove (cycle-0004 F2).

**Next step (grokbot, cycle 0007):** attribution v1 that classifies the
contacts in the lookback window by semantic geom role.

- Opponent glove on the faller's head, torso, guard or glove, and the faller's
  own glove on the opponent, are boxing. They don't block, so the fall is the
  faller's loss, which matches a knockdown.
- Body-to-body contact (clinch, lean, shove) and glove contact on
  `native_non_scoring` surfaces (back, below the belt, legs, which are
  possible fouls) stay `unresolved`.
- The residual exploit is "lean on the opponent's body while falling". The
  blocked-as-faller column added here exists to catch it.

## F2 — Info — The faller ordering is measurable even when results are blocked

`blocked_as_faller` separates the policies (1.00, 0.67, 0.33, 0.00) when W/D/L
cannot. Evaluation and promotion reports should always print this column.
Without it, a policy's blocked bouts are indistinguishable from its
opponent's.

## Improvement shipped this cycle

`grokbot.arena.policy_summary(records)` builds a per-policy table over arena
records, with every attempt in exactly one column:

- W/D/L, read from that policy's view.
- `blocked_as_faller`: an unresolved loss by this policy at the blocking tick.
- `blocked_other`.
- `own_fault` and `opponent_fault`, for controller faults.
- `invalid_other`, for engine faults and no-sporting-end.
- `self_falls`.
- `median_first_loss_tick`.
- `blocked_as_faller_rate`.

`grokbot/tests/test_policy_summary.py` has 3 tests. They check that the
columns sum to `attempted`, that results and faults are charged from each
policy's view, that a blocked bout names the faller even when the other
policy also goes down unresolved later, and the fall counts and medians.
Mutation: 4 of 4 killed. The tick-filter mutant survived the first fixture
and needed a later unresolved loss by the non-faller.

The engagement-range probe is committed with this report.

## Next steps

- grokbot (cycle 0007): role-based attribution v1 (F1). Rerun this probe and
  the cycle-0004 separated probe to show (a) that collision-stress bouts stay
  blocked and (b) that the separated results are unchanged.
- jekko (P2.5/P6): a registered reset at boxing range, around 0.9–1.1 m
  pelvis to pelvis, facing. Neither the 1.6 m keyframe nor the 0.2 m stress
  overlap is a boxing start, and every self-play signal so far comes only from
  falling.
