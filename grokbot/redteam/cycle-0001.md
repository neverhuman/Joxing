# Grokbot red-team — cycle 0001 (2026-10-27)

Lane: `joxing_claude` (grokbot red-team + level-up). Reviewed `origin/main` at
`2c10b08`. This is the first cycle, so the review window is the last 24 h:
82 commits by jekko, from v0.0.10 conclusion through v0.0.11 D0/D1 and the start
of the pretrained shared-motor bridge.

## Verdict

**There is no two-player self-play yet, so nothing is "leveling up".** Native
boxing games remain 0 (`docs/STATUS.md`). `BoxingEnv.competitive_ready` is
`False`, and no learner runs in the shared world. No league, payoff matrix, Elo
or exploiter code exists anywhere in `src/`, `scripts/` or `tests/`. The
two-player plumbing (shared world, simultaneous commands, egocentric
observations, referee, zero-sum shaped reward) exists. It is side-symmetric to
machine precision (F7). The blocker is still motor competence: both athletes
fall within about 1–5 s.

## Findings (most severe first)

### F1 — High — No CI: the 402-test gate never runs on a clean checkout

`origin/main` has no `.github/`. Tests run only inside hand-launched supervisor
scripts on xbabe hosts, at the frozen commits those scripts choose. Commits
between gates, including doc commits that also touch `src/`, are never tested.
External PRs can't be validated either.
**Fixed in this cycle's PR** (see "Improvement" below).

### F2 — High — Motor budgets are 3–4 orders of magnitude below the upstream recipe

Upstream MS-Human-700 full-body DynSyn-SAC locomotion uses
`total_timesteps: 5e7` (`third_party/msgym/DynSyn-SAC/configs/locomotionFull.json:46`).
Joxing's motor pilots used far less:

| Run | Transitions per seed/condition | Fraction of upstream 5e7 |
|---|---:|---:|
| v0.0.10 D2 residual pilot | 8,192 | 1/6,100 |
| v0.0.5 extension | 160,000 | 1/312 |
| v0.0.4 pilot | 39,040 | 1/1,280 |

Every motor gate failed, for example v0.0.10 with −0.066 s and +0.016 s paired
change and 1/8 five-second survivors per seed. These failures say almost
nothing about learnability at these budgets. Throughput is not the limit: v0.0.9
measured 520.7 native control transitions/s at 16 workers. At that rate, 1M
transitions take about 32 min and 5M take about 2.7 h.
**Next step:** once the pretrained bridge lands, preregister one adequately
budgeted residual fine-tune (≥1M transitions × 3 seeds, the same
gates/denominators), instead of more 8k-transition pilots.

### F3 — High — "Leveling up" is unmeasurable: no evaluation/league substrate

P4 of `docs/SELF_PLAY_EXECUTION_PLAN.md` plans `league.py`, `evaluation.py` and
known-payoff fixtures, but nothing exists yet. Without a frozen panel, payoff
matrix, rating confidence intervals and an exploitability proxy, the first
self-play run can't show improvement, and it can't tell improvement apart from
cycling or forgetting.
**Next step (grokbot lane, disjoint path `grokbot/`):** build a pure, tested
known-payoff oracle. It covers the payoff matrix from side-swapped paired
games, a Bradley–Terry/Elo fit with bootstrap CIs, exact Nash and exploitability
of a mixture on the empirical matrix, and RPS/transitive fixtures with
invalid-game denominators. jekko's `src/joxing/rl/*` can adopt or cross-check
it.

### F4 — Medium (High before any learning) — The referee pays up to 64.5 points/s per glove for feather taps

Evidence: drive one `HitTracker` with the minimum qualifying touch (0.5 m/s
approach and active speed, face cos 0.5), then clear for the 30 ms minimum with
5 cm retraction and ≥2 cm clearance. Repeat for 60 s.

| Tap force × contact | Impulse | Clear | Events/60 s/glove | Points/60 s/glove |
|---|---:|---:|---:|---:|
| 251 N × 1 ms | 0.251 N·s | 30 ms | 1,936 | 3,872 (64.5/s) |
| 26 N × 10 ms | 0.260 N·s | 30 ms | 1,500 | 3,000 |
| 6 N × 50 ms | 0.300 N·s | 30 ms | 750 | 1,500 |
| 26 N × 10 ms | 0.260 N·s | 90 ms | 600 | 1,200 |
| 251 N × 1 ms | 0.251 N·s | 29 ms | 1 | 2 |

The 30 ms clear is the only refractory period. The qualifying event (0.25 N·s,
0.5 m/s) is about 1 % of a real punch's momentum (~2.9 kg effective mass ×
~9 m/s ≈ 26 N·s, Walilko et al. 2005). Under win = higher points, the cheapest
reliable tap rate dominates. Motor bandwidth, not the rules, would be the only
brake. P2.4 of the execution plan already schedules "high-frequency
peck/rearm" probes. These numbers give the referee-level ceiling those probes
should report.
**Next step:** before learning, freeze a named rules revision with
(a) a punch-scale impulse/approach threshold derived from scripted-strike data,
(b) a per-glove refractory or ≥10 cm chambered-retraction rearm, and (c) a
regression probe asserting the ceiling. These are referee-schema changes, so
they stay jekko's to make.
Reproduce: `grokbot/redteam/probes/cycle-0001-tap-rate.py`.

### F5 — Medium (High before any learning) — Falls and exits carry no utility

The env ends only at the bell, and the outcome is `sign(points)`. A policy that
leads and then goes to the floor ("turtle") cannot lose. A falling attacker
pays nothing either. This is acknowledged (competition disabled, P2.5
ending-utility table), but it must be a hard gate: no learner may run against a
bell-only ending, even a "positive-control" one.
**Next step:** add the planned table-driven ending tests. Include an adversarial
"lead then lie down" scripted scenario whose expected utility is a loss.

### F6 — Medium — Effort skew and repository growth

The 24 h churn on main breaks down as follows:

- `artifacts/` +3.15M lines.
- `scripts/` +13.3k, mostly 46 qualify/supervise/register/audit/verify scripts.
- `src/` +12.8k, `tests/` +6.1k, `docs/` +9.8k.

58 of 82 commit subjects are record/retain/register/publish/freeze. The pack is
76 MiB, and 171 MB of artifacts are checked out. Evidence discipline is a
strength. But the motor bottleneck has been unchanged since v0.0.4, and the
proof overhead per physics step is very high.
**Next step:** move raw evidence (NPZ, ZIP, multi-MB JSON) to release assets, as
v0.0.9/v0.0.10 raw tarballs already do. Cap per-commit artifact bytes. Spend
the saved cycles on F2.

### F7 — Positive, now regression-locked — The world is side-equivariant

Swapping corners (A←c1, B←c2 versus A←c2, B←c1, 25 random control steps)
gives mirrored egocentric observations to ≤1e-14 and mirrored points, with
identical A/B contact counts. This holds even in `close-overlap-collision-probe-v2`,
which has 10,469 inter-actor contacts. Rewards are exactly zero-sum at every
step. A 0.1 % mass change on `B_humerus_r` is caught at step 0, so the test has
teeth. Paired side-swapped evaluation and a shared two-corner policy are
therefore sound on the current scene.

### F8 — Low — Threshold boundaries depend on float accumulation

A nominal 25 N × 10 ticks (0.25 N·s) does not qualify, because the summed
impulse is below 0.25, while 250 N × 1 tick does qualify. A retraction of
exactly 3 cm (0.30 → 0.27) does not rearm. This doesn't matter for learning,
but it does matter for exact-threshold fixtures. Compare against a documented
epsilon or integer micro-units.

## Improvement shipped this cycle

1. `.github/workflows/ci.yml` mirrors the canonical supervisor gate:
   `uv sync/run --project experiments/v0.0.11/runtime --locked`, with pinned
   submodules, all BLAS/OpenMP pools at 1 thread, and actions pinned by commit
   SHA. It runs on PRs and on pushes to main.
2. `grokbot/tests/test_side_symmetry.py` (5 tests, ~13 s) locks in F7.

Local validation (xbabe2, uv 0.11.28, one thread, `nice 10`):

- Full `tests/` on unchanged main `2c10b08`: 422 passed in 516.24 s. The
  slowest test is `test_wave_learner.py::test_parallel_extension_trials_defer_initial_evaluation`
  at 50.65 s.
- `grokbot/tests`: 5 passed, 12.60 s.
- Mutation: B-side mass ×1.001 is killed in both scenarios.

## Next cycle candidates

- F3 known-payoff league/evaluation oracle under `grokbot/` (pure numpy, tested).
- Watch the pretrained bridge commits for observation/feature leakage across
  corners. The 3601-feature teacher layout must stay egocentric per corner.
