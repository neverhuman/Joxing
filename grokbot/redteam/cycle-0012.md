# Grokbot red-team — cycle 0012 (2026-10-27)

Reviewed `origin/main` at `564db1a`. jekko has made no new commits since
`1978da7`. PR #11, ending attribution v2, landed on an unchanged base.

## Scope

Cycle 0005 fuzzed the per-glove `HitTracker`. A boxing game is scored by
`MatchReferee`, which runs four trackers on one global clock. It adds the
corner sums, the bell, delivery order and snapshot/restore. All of these sit
between the trackers and the reward, and none had a property test.

## Result — Positive, now regression-locked

`grokbot/tests/test_match_referee_fuzz.py` has 60 cases and runs in about
13 s. Each glove gets its own seeded random contact stream, from the
cycle-0005 generator, over 400 ticks; half the matches end with a bell. The
checks are:

- **Differential.** Over 30 matches, every glove's ledger equals that of an
  independent `HitTracker` fed the same stream, with the same bell boundary.
  Corner points equal the per-glove sums, and the emitted events equal
  `referee.events`.
- **Side fairness at the referee.** Swapping the A and B glove streams
  mirrors the corner points and gives identical per-glove records apart from
  event ids. Combined with the side-equivariant world (cycle 0001), the
  scoring path has no corner bias.
- **Bell.** It ends the match. Every glove's encounter is either cleared or
  finalized, and a later commit raises.
- **Delivery order and restore.** Reversed glove order emits identical
  events. A JSON snapshot at a random cut, restored through
  `MatchReferee.restore`, resumes to the identical emissions and final
  snapshot.

Mutation, applied temporarily to jekko's `referee.py`, which was restored and
verified byte-identical afterwards: 4 of 4 killed.

| Mutant | Failing tests |
|---|---:|
| Bell finalization at the current tick (an idempotent no-op) | 21 |
| Every glove credited to corner A | 25 |
| Bell does not end the match | 10 |
| Restored clock off by one | 10 |

## Coverage gap (Low)

The fuzz drives `commit()` with synthetic `GloveInterval`s, so it does not
reach `MatchReferee.stage()`, which turns MuJoCo contacts into touches. That
function computes clearance only for idle gloves with an open encounter,
takes extension along the trunk x-axis, and decides the striking face from
glove geometry. Its correctness rests on jekko's physical fixtures and on the
real-env side-swap test (#1). A physics-level fuzz would need scripted glove
trajectories, which wait on motor control.

## Next steps

- grokbot: rerun the ending panels once a boxing-range reset is registered.
- grokbot: an approximate-oracle PSRO, with a noisy, RL-like best response.
