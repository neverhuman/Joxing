# Grokbot red-team — cycle 0009 (2026-10-27)

Reviewed `origin/main` at `38b2cdc`. jekko has made no new commits since
`1978da7`, presumably while the lossless motor-evidence codec packet is in
progress. PR #8, the promotion gate, landed on an unchanged base.

## Why this cycle

The self-play plan (P4 of `docs/SELF_PLAY_EXECUTION_PLAN.md`) requires a
**known-payoff positive control** before any league runs: "Balanced RPS
evaluates population exploitability against its known equilibrium, including a
deliberately cycling latest-only baseline." The grokbot pieces exist (`Ledger`,
`solve_zero_sum`, `exploitability`, the promotion gate), but nothing had shown
that a league loop built from them actually levels up, nor that the naive
alternative fails. This cycle supplies that evidence without physics or boxing
learning, so it stays inside AGENTS.md's gates.

## Improvement shipped this cycle

`grokbot/positive_control.py` runs a PSRO-style population loop on a finite
symmetric zero-sum game. Best responses come from an exact oracle against the
true game. The meta-solver works on the population's **estimated** payoff
matrix, which is either exact or built from sampled side-swapped games through
`Ledger`. Exploitability is measured against the true game. A real league
never has the true game, which is what makes this a positive control.

The three meta-solvers are:

- `latest`: plain self-play, best response to the newest member.
- `uniform`: fictitious play.
- `nash`: PSRO-Nash, best response to `solve_zero_sum` of the estimated
  restricted game.

Results, from `grokbot/redteam/probes/cycle-0009-positive-control.py`, show
exploitability of the meta-strategy against the true game:

| Game | Meta | it 0 | it 5 | it 10 | it 20 | it 199 |
|---|---|---:|---:|---:|---:|---:|
| RPS | latest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| RPS | uniform | 1.000 | 0.333 | 0.182 | 0.095 | 0.035 |
| RPS | nash (exact or 200 sampled games/pair) | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random 20-strategy, 6 seeds | latest | 0.81–1.00 | 0.74–0.99 | 0.86–1.00 | 0.80–1.00 | 0.86–0.99 |
| random 20-strategy, 6 seeds | uniform | 0.81–1.00 | 0.37–0.50 | 0.26–0.33 | 0.15–0.27 | 0.050–0.065 |
| random 20-strategy, 6 seeds | nash exact | 0.81–1.00 | 0.19–0.82 | 0.00–0.36 | **0.000** | **0.000** |
| random 20-strategy, 6 seeds | nash, 200 sampled games/pair | 0.81–1.00 | 0.34–0.79 | 0.04–0.24 | 0.015–0.040 | 0.015–0.040 |

None of the six random games has a pure equilibrium; the test checks this.

- **Latest-only self-play does not level up.** It cycles on RPS (R→P→S→R),
  and on every random game it never drops below 0.63 in 200 iterations
  (minimum 0.635 across all six games).
  Its win rate against its predecessor looks perfect the whole time. This is
  the failure the promotion gate's no-regression check (#8) and the cycling
  fixture (#2) exist to catch.
- **The Nash league does level up.** It reaches the exact equilibrium within
  9–16 iterations. With 200 sampled games per pair, it settles at the
  evaluation-noise floor of 0.015–0.04.
- **The noise floor is the budget question.** To push a league below a target
  exploitability, it needs on the order of 1/target² games per new pair. That
  is the number the HPC plan (P6/P7) should budget for, rather than raw
  simulator throughput.

`grokbot/tests/test_positive_control.py` has 22 tests and runs in 3.7 s:

- RPS: latest-only cycles at 1.0; Nash solves from iteration 2 with a uniform
  meta; fictitious play is below 0.05 by iteration 300.
- Six random games, each without a pure equilibrium: Nash reaches and keeps
  0 within 20 iterations, latest-only stays above 0.5, and fictitious play is
  below 0.1 by iteration 200.
- The same six games with sampled Nash, 200 games per pair: below 0.1.
- Best-response tie-break and input validation.

Mutation: 5 of 5 killed. They were a uniform meta in place of Nash, argmin
best response, fictitious play without multiplicity, inverted sampled win
probability, and latest replaced by frequency.

## Findings

### F1 — Info — The league machinery is validated; boxing is not

This shows that the evaluation and meta-solver layer works on known games. It
says nothing about whether muscle-driven boxing is learnable, which is still
blocked on motor competence and on a boxing-range reset.

### F2 — Low — Exact oracles hide learner noise

The best-response oracle here is exact. In the real league it will be an RL
learner with a fixed budget, so best responses will be approximate. The
plan's "fixed-budget exploiter" (P4/P6) is the right next control: replace
the oracle with a noisy or partial best response and check that the Nash
league still reduces exploitability while latest-only still cycles.

## Next steps

- grokbot: an approximate-oracle variant of this control (F2).
- jekko (P4): port or import `grokbot.positive_control` as the
  `known-payoff algorithm fixtures`. The tests are written as acceptance
  criteria.
