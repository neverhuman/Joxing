# Grokbot red-team — cycle 0010 (2026-10-27)

Reviewed `origin/main` at `1aecf1e`. jekko has made no new commits since
`1978da7`. PR #9, the known-payoff positive control, landed on an unchanged
base. This cycle tests the one league measurement the plan cannot compute
exactly in boxing: exploitability.

## F1 — High — A fixed-budget exploiter can certify a cycling learner as safe

P4/P6 of `docs/SELF_PLAY_EXECUTION_PLAN.md` plan "payoff coverage and
fixed-budget exploiters" as the exploitability measurement. In boxing, the
true best response is unknown, so an exploiter trained for a fixed budget is
the only estimate available. It is a lower bound. The question is how loose
that bound is.

Probe: `grokbot/redteam/probes/cycle-0010-exploiter-recall.py`. The snapshots
are the first 60 latest-only self-play policies of each game. Every one is
truly exploitable: 1.0 in the cyclic games, 0.6–1.0 in the random ones. An
exploiter with a budget of *b* searches *b* random strategies.

| Game | Budget | Recall (reported/true) | Exploitable snapshots reported safe |
|---|---:|---:|---:|
| cyclic 20 | 5 | 0.183 | **81.7 %** |
| cyclic 20 | 10 | 0.450 | 55.0 % |
| cyclic 100 | 10 | 0.117 | **88.3 %** |
| cyclic 100 | 50 | 0.567 | 43.3 % |
| random 20 (3 seeds) | 1 | 0.21–0.36 | 47–55 % |
| random 20 (3 seeds) | 5 | 0.70–0.78 | 1.7–5.0 % |
| random 20 (3 seeds) | 10 | 0.89–0.93 | 0 % |

Recall depends on how many counters exist, not just on the budget. Random
games are full of counters, so a modest exploiter finds one. When each
strategy has a single counter (cyclic games), even an exploiter searching half
the strategy space reports nearly half of the fully exploitable snapshots as
unexploitable. Boxing styles are more likely to be counter-sparse than
counter-rich: a specific defence beats a specific attack.

**Requirement for the P6 exploiter, before its numbers gate anything:**

1. **Calibrate it on a positive control with known exploitability.** For
   example, train the same exploiter recipe against a `cyclic_game` or an RPS
   population of frozen boxing-like policies with an injected known counter.
   Report its recall.
2. **Report every exploitability number with the exploiter's budget and its
   calibrated recall.** A reported value of 0 means "not found within budget",
   not "safe".
3. **Never use a low exploiter score as a promotion criterion on its own.** The
   grokbot promotion gate (#8) relies on panel coverage and no-regression,
   which do not depend on exploiter strength.

## Confirmed — Latest-only self-play cannot level up in counter-sparse games

In `cyclic_game(20)`, latest-only self-play is fully exploitable (1.0) at all
60 iterations. The Nash league is unsolved until its population holds all 20
strategies, and is solved from iteration 19 onwards. The league has to keep
the whole archive. That is the plan's "archive predeclared snapshots,
including weaker policies" requirement, now with a test behind it.

## Improvement shipped this cycle

`grokbot/positive_control.py` gains:

- **`cyclic_game(n)`**: each strategy beats only its predecessor, and every
  other pairing is a draw. It has a single counter per strategy.
- **`exploiter_proxy(game, mixture, budget, rng)`**: the best utility among
  `budget` random strategies, floored at 0. It never overstates, and it is
  exact at full budget.
- **`exploiter_recall(game, mixtures, budget, rng)`**: returns the
  reported/true ratio and the share of exploitable mixtures reported as safe.

There are 5 new tests, 27 in all for the module:

- In the cyclic game, latest-only stays at 1.0, and Nash is solved exactly
  when all 20 strategies are present.
- The proxy never exceeds the truth and is exact at full budget, checked on
  both game types.
- A weak exploiter calls cyclic snapshots safe (recall below 0.4, more than
  60 % missed), while the same budget recalls more than 0.6 on a random game.
- The exploiter reports 0, never a negative value, when only a dominated
  strategy was tried.

Mutation: 3 of 3 killed. They were "ignore budget", "no floor at 0" and
"missed share fixed at 0". The floor mutant needed the dominant-strategy test.

## Next steps

- jekko (P6): adopt the calibration requirement (F1). `exploiter_recall` is
  the measurement to report.
- grokbot: the `classify_contacts` active-speed check (carried since cycle
  0007).
