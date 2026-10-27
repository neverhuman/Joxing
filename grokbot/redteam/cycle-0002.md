# Grokbot red-team — cycle 0002 (2026-10-27)

Reviewed `origin/main` at `2c10b08`. jekko has made no new commits since
cycle 0001, so this cycle closes the highest-ranked open gap that can be
addressed without touching jekko's paths: cycle 0001 F3, "leveling up is
unmeasurable".

## Status of open findings

| ID | Finding | Status |
|---|---|---|
| 0001-F1 | No CI | PR #1 open; the first hosted run failed at collection because `MUJOCO_GL=egl` and hosted runners have no EGL. Fixed with `MUJOCO_GL=disable` in 499533d. The second run failed 3 history-dependent tests; fixed with `fetch-depth: 0` in cb2aad6 (see F4). This PR carries the identical `ci.yml`, so it can be tested without stacking on #1 (an identical add/add merges cleanly). |
| 0001-F2 | Motor budgets 3–4 orders below upstream | Open; jekko is on the pretrained bridge, input oracle running on xbabe1 |
| 0001-F3 | No league/payoff/exploitability substrate | **Addressed here** with a reference oracle, `grokbot/league.py` |
| 0001-F4 | Feather-tap ceiling, 64.5 pts/s per glove | Open; P2.4 is planned, no rules revision yet |
| 0001-F5 | Falls/exits carry no utility | Open; P2.5 is planned |

## New findings

### F1 — Medium — Sensor delays are per-corner and asymmetric by construction

`SensedBoxingEnv(sensor_schedule={"A": [...], "B": [...]})` accepts independent
schedules, and nothing ties a schedule to a policy rather than a corner. A
league that keeps one schedule per corner, and doesn't swap it with sides,
hands the lower-latency corner a structural edge. That edge would show up as
"skill" in Elo.
**Requirement for P4/P6:** a schedule belongs to the match, and is either
symmetric or swapped with the corners in each paired game. Report the
corner-A mean utility, `Ledger.side_balance()`, next to every rating table.
It is 0 for a fair panel.

### F2 — Low — A zero-latency default is fine as an oracle but must not be the evaluation world

The default `BoxingEnv` gives instantaneous opponent kinematics, including
velocities. A policy trained there can learn reaction-based counters that no
delayed-sensing evaluation would allow. The plan already calls this world a
zero-delay oracle (`SELF_PLAY_EXECUTION_PLAN.md` P3). The red-team ask is
narrower: freeze the evaluation panel's sensing contract before the first
learner, so that ratings from the two worlds are never pooled.

### F4 — Low — The gate only runs from a full clone on a host with EGL

The first two hosted CI runs found two hidden host assumptions in the 422-test
gate:

- **`MUJOCO_GL=egl`.** This is what every supervisor exports. With it set,
  `import mujoco` itself fails where libEGL is absent, and 51 modules error at
  collection. No test renders, so CI uses `MUJOCO_GL=disable`.
- **Full git history.** Three tests `git show` frozen historical script
  sources at old SHAs:
  - `test_motor_residual_audit.py::test_saved_evidence_corruption_rejected[reward|contact]`
    reads `7acd207`.
  - `test_scaling_audit.py::test_actual_capacity_stop_retains_all_seventy_eight_cases`
    reads `a8f47d8`.

  A shallow clone or a release tarball cannot pass the gate.

With `fetch-depth: 0` and GL disabled, the hosted runner otherwise passed 424
tests in 310 s. That is faster than the 516 s on xbabe2 under load.
**Next step (jekko, optional):** skip or `xfail` those three tests when
`git cat-file -e <sha>` fails, with a clear reason, so a tarball reproduction
is honest about what it cannot check.

### F3 — Checked, no defect — Opponent history is causal

`TimedHistory.at()` admits only samples with `received_tick <= decision_tick`.
`observe()` reverses the points per corner, and opponent features are
kinematics only: no commands, activations or bundle identity. The pretrained
bridge `NativeMotorMap` rotates keypoints by an explicit, compile-checked spawn
yaw, and the root slide joints live in each actor's own spawn frame. No
corner-frame leakage was found.

## Improvement shipped this cycle

`grokbot/league.py` is a pure-numpy, known-payoff evaluation oracle that P4's
`src/joxing/rl/league.py` and `evaluation.py` can adopt or cross-check. It
contains:

- **`Game` / `Ledger`.** Each attempted game is kept with its side.
  - An invalid game needs a reason and no outcome. It stays in the `attempted`
    denominator and never scores.
  - `counts()` gives win/draw/loss/invalid per ordered pair.
  - `payoff()` returns an antisymmetric mean-utility matrix, with NaN for
    unplayed pairs (never a silent 0).
  - `side_balance()` detects corner bias.
- **`ratings()`.** Bradley–Terry ratings on the Elo scale, using Hunter's MM
  algorithm, with draws counted as half wins.
  - Virtual-draw prior; unplayed players are NaN.
  - A disconnected comparison graph is rejected, because it has no common scale.
- **`bootstrap_ratings()`.** Seeded percentile CIs; resamples that disconnect
  the graph are skipped and counted.
- **`solve_zero_sum()`.** Exact maximin mixture via simplex with Bland's rule,
  with a duality-gap certificate (it raises if the gap is above 1e-7).
- **`exploitability()`.** Best pure-response utility against a population
  mixture. It is 0 exactly at Nash.

Tests (`grokbot/tests/test_league.py`, 16 tests, 0.5 s) cover these cases:

- Malformed and invalid games.
- Denominators and antisymmetry.
- Corner bias.
- A transitive ladder recovered within ±40 Elo, with bootstrap CIs excluding 0.
- The exact Elo expected-score identity.
- Four known matrix games and 25 random games certified.
- RPS, which is exploitable unless mixed.
- A dominant versus a dominated policy.
- A **latest-only cycling learner**: every snapshot beats its predecessor 100 %,
  yet the newest is exploitable by 1.0 and the Nash mixture by 0.

Validation:

- Stress: 3,000 random matrices, a quarter each of Gaussian, {−1,0,1} ties,
  constant and antisymmetric. 0 failures; worst duality gap 4.9e-15. A
  200×200 antisymmetric matrix solves in 1.2 s.
- Mutation: 5 of 5 mutants killed. They were the exploitability sign, mirrored
  invalid count, side sign, payoff antisymmetry and rating centring.
- `-W error`: no warnings.

## Next steps

1. **jekko (P4):** when `src/joxing/rl/league.py` lands, assert agreement with
   `grokbot.league` on the RPS, cycling and ladder fixtures. Or import it.
2. **grokbot, next cycle:** an end-to-end evaluation harness that plays
   side-swapped paired games of frozen scripted controllers in the real
   `BoxingEnv` and feeds `Ledger`. Use the zero-command, random and fixed-pose
   controllers, since no learned policy exists. That proves the plumbing and
   gives an invalid-rate baseline.
3. Still highest leverage for the project: 0001-F2, an adequately budgeted
   motor run after the bridge.
