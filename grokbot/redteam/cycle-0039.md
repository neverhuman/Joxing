# Grokbot red-team — cycle 0039 (2026-10-28): blocked games checked by hand; v0.0.26 preregistration (train against both opponents)

Branches from `origin/main` at `0d467da`. jekko pushed native-tape work
(`f4a5b03`, `870f480` and the merge `0d467da`) **directly to main**,
adding `src/` and `tests/`. The v0.0.25 report PR (#38) was rebased onto
it. This PR is **the v0.0.26 preregistration on its own**. It changes no
STATUS or CHANGELOG text, so it cannot conflict with #38. The run starts
only after merge, through one command gated on main CI being green and load
below 70.

## Hand check: are the v0.0.13 blocks real? (owner 23:05)

I replayed v0.0.25's hard-gate game on seed 29000 deterministically. Both
seats were identical, as common random numbers make mirrored games.

| | Value |
|---|---|
| Faller | The opponent, v0.0.13 |
| Support onset | Tick 1,519 |
| Attribution | `unresolved`, so the game is blocked |
| Attribution window | [onset − 500, onset + 99] = [1,019, 1,618] |
| Blocking contact ticks in the window | **58: 50 immediately before onset (1,469–1,518), 8 during the fall** |
| Contact type | All positive-force **torso-to-torso** contacts (`native_non_scoring` on both bodies) |

**Verdict: a real block, not a harness bug.** The candidate is pressed
body-to-body against v0.0.13 for the 50 ms before it topples. The rule
correctly declines to attribute that fall to a strike.

A correction to my earlier blocked-game diagnostics (cycles 0033 and 0036):
they looked at [fall − 500, fall] and missed the 100-tick persistence period
**after** onset, which is also part of the window. Their verdicts stand
(`unresolved` falls with body contact), but contact during the fall itself
was not counted.

## The one lever: train against both opponents (`protocols/v0.0.26.json`)

v0.0.25's contact penalty, trained only against v0.0.19, worked there:
contact fell from 380 to 113, with 48–0 and no blocks. Against v0.0.13,
which training never saw, all 48 games were blocked. Its registered reading
for that outcome was "alternate both training opponents".

| | v0.0.25 | **v0.0.26** |
|---|---|---|
| Training opponent | Frozen v0.0.19 | **Alternating per ES pair:** frozen v0.0.19, then the frozen v0.0.13 champion. Both twins of a pair face the same opponent. |
| Unchanged | | The fitness (stand + valid knockdown − 0.5 · min(1, contact/500)), the v0.0.22 start, the model, ES (seed 16), the reset and ending v2 |

**Smoke gate first:**

1. The penalty term must vary against **each** opponent: 24 base trials
   per opponent, alternating seeds 31000–31047.
2. A 48-game base head-to-head against v0.0.19 must be at least 90 %
   valid (seeds 31100–31123).

**Gates, in order, on new seeds:**

| Gate | Seeds |
|---|---|
| Primary against frozen v0.0.19 | 32000–32023 |
| Hard gate against v0.0.13 | 33000–33023 |
| Monotonicity: @48 vs @16, and @16 vs the base | 34000–34023 |

**Tag only if every gate passes.** The contact rate is reported for both
opponents.

**Registered red-team note:** half the pairs now face v0.0.13, so each
opponent gets 12 trials per iteration, and early-training collapse (v0.0.25
@16 lost 4–44 to its base) may recur.

## Code

| Change | Detail |
|---|---|
| `motor_gate.train` | Accepts a list of opponents: pair k faces `opponent[k % n]`, and a single spec behaves exactly as before |
| Test | `RecordingPool` checks that the twins are paired, that opponents alternate, and that single-opponent tasks are unchanged |
| Probe | `probes/cycle-0039-two-opponent-gate.py` |

While writing the probe I caught an alignment bug in the per-opponent smoke
split. Invalid rows were dropped *before* splitting into alternating halves,
so one engine fault would have shifted every later trial into the wrong
opponent's half. It is fixed by splitting the rows first.

The smoke test ran on seeds 99920 and above. With 2 seeds per opponent,
v0.0.13 gave identical contact (80 and 80), so the spread check correctly
failed. The pass path was verified with the spread floor relaxed in the
smoke copy only. **No game has been played on seeds 31000–34023.**
