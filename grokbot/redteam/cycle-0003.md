# Grokbot red-team — cycle 0003 (2026-10-27)

Reviewed `origin/main` at `2c10b08`; jekko has made no new commits since cycle
0001. CI health: PR #1 is green with 427 passed in 477.5 s, and PR #2 is green
with 438 passed in 484.6 s. Both run on the same `ci.yml`. This cycle puts the
first real games through the shared world.

## F1 — High — The current game gives no learning signal, and falling is free

Probe: `grokbot/redteam/probes/cycle-0003-scripted-panel.py 5000`. It runs a
round robin over four scripted controllers, playing both corners for seed 0
from the native keyframe, with 5 s per game and one thread. The controllers
are:

- `relaxed`: excitation 0.0006.
- `baseline`: excitation 0.076.
- `cocontract`: excitation 0.924.
- `noise`: uniform normalized commands, resampled every 20 ms.

| Pair (first vs second) | Games | Valid | Result | Referee events | First falls at | Second falls at |
|---|---:|---:|---|---:|---:|---:|
| relaxed vs baseline | 2 | 2 | 0–0 draw ×2 | 0 | 1.86 s | 1.16 s |
| relaxed vs cocontract | 2 | 2 | 0–0 draw ×2 | 0 | 1.86 s | 1.06 s |
| relaxed vs noise | 2 | 2 | 0–0 draw ×2 | 0 | 1.86 s | 1.06 / 1.08 s |
| baseline vs cocontract | 2 | 2 | 0–0 draw ×2 | 0 | 1.16 s | 1.06 s |
| baseline vs noise | 2 | 2 | 0–0 draw ×2 | 0 | 1.16 s | 1.06 / 1.08 s |
| cocontract vs noise | 2 | 2 | 0–0 draw ×2 | 0 | 1.06 s | 1.06 / 1.08 s |

"Falls" means own pelvis height below 0.5 m. Twelve of twelve attempts were
valid. Every athlete fell within 1.9 s, and the athletes never touched: they
start 1.6 m apart (pelvis to pelvis), with heads 1.76 m apart. The empirical
payoff matrix is identically 0, so every controller ties with every other, and
falling 0.8 s earlier changes nothing. Under the canonical win/draw/loss
utility, once motors improve, the cheapest dominant line is to score one touch
and then go to the floor. 0001-F5 is therefore not hypothetical. It is the
equilibrium of the current rules for any policy that can land one touch.
**Next steps:**

1. P2.5's ending table must be in the env before any learner runs.
2. Report fall-rate and time-to-fall per policy beside every payoff table. The
   arena records `fall_tick` per policy for this purpose.
3. For first-signal learning, consider a reset scenario at engagement range,
   such as the existing close-overlap variants or a reviewed ~0.9–1.1 m
   variant. Otherwise contact needs footwork the motors don't have.

## F2 — Low — The env's random streams follow the corner, not the policy

`BoxingEnv.reset` assigns `SeedSequence(seed).spawn(2)` to corners A and B. In
a side-swapped pair, a stochastic policy therefore gets a different stream in
each corner. In F1, `noise` fell at 1.06 s as corner B and 1.08 s as corner A.
This is not a bias, because the spawn is symmetric. But paired evaluation loses
common random numbers, so variance per paired game is higher than it needs to
be.
**Next step:** in the league harness, key the stochastic stream on
`(seed, policy)`, or swap streams with corners. This is an evaluation-protocol
choice, not an env change.

## Improvement shipped this cycle

`grokbot/arena.py` is a paired side-swapped match runner for frozen
controllers, built on the unchanged `BoxingEnv.act` contract:

- **`play()`** returns one record per attempted game and never raises for a
  game-level failure. The record carries `first`, `second`, `first_side`,
  `outcome` and `invalid_reason`, which are `grokbot.league.Game`'s fields
  (PR #2). It also carries points from first's view, seed, contract hash,
  ticks, referee events and `fall_tick` per policy.
- **Fault attribution.** Commands are validated before physics: shape,
  float32, finite and within [−1, 1]. A malformed command is charged to its
  policy (`fault_player`). Engine guards, such as `External assistance
  forbidden`, quarantine the game with no fault player. An invalid game has
  `outcome = None`; it is never a draw or a win.
- **`paired()`** plays both corner assignments per seed. `SCRIPTED_PANEL` is
  the plumbing and baseline panel above. It is not boxing competence and not a
  promotion anchor.

`grokbot/tests/test_arena.py` has 8 tests and runs in about 26 s. It covers:

- The corner-to-first-view result mapping.
- Identical controllers draw 0–0 from both corners.
- A side-swapped deterministic pair mirrors fall ticks exactly.
- NaN and float64 commands are charged to the right policy, and the env is
  reusable and deterministic afterwards.
- An `xfrc_applied` push mid-game is quarantined, not scored.
- Same seed gives the same record.
- Self-pairing by name and unknown corners are rejected.

Mutation: 4 of 4 mutants were killed. They were the result orientation,
dropping the dtype check, keying `fall_tick` by corner, and charging faults to
the corner instead of the policy.

This PR is independent of #1 and #2. It carries a byte-identical `ci.yml`,
`grokbot/__init__.py` and `grokbot/tests/__init__.py`, and does not import
`grokbot.league`.

## Next steps

- After #2 merges, use `arena.paired` → `league.Ledger` as an integration test:
  a scripted payoff matrix of all zeros, `side_balance` = 0, and invalid
  denominators.
- Still highest leverage: 0001-F2, an adequately budgeted motor run after the
  bridge, and the P2.5 ending table.
