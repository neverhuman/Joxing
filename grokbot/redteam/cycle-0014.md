# Grokbot red-team — cycle 0014 (2026-10-27): first two-player self-play league

Reviewed `origin/main` at `934cb01`. jekko has made no commits since `1978da7`,
and main CI for #13 is green. This cycle carries out the owner requests of
17:05, 20:45 and 21:42: a real two-player self-play RL loop on the reference
ending table with common random numbers, played over a league of frozen
checkpoints chosen by the P4 sampler. Each generation is reported with Ledger
Elo, its gate decision, whether it beats **every** earlier checkpoint, and its
side balance.

> **Classification.** This is diagnostic self-play, not boxing. The AGENTS.md
> contact, rules, stable-motor and replay gates are unmet. From the native
> keyframe the athletes start 1.6 m apart and never touch, so under the
> reference ending the learnable objective is "fall later than the opponent".
> Every result below is learned **balance**.

## What was built

| Module | Role |
|---|---|
| `grokbot/endings.py` | Refactored into incremental `ContactClassifier` and `BoutMonitor`, O(new rows) per control step. The batch functions are thin wrappers, so there is one implementation. |
| `grokbot/reference_env.py` | Wraps the unchanged `BoxingEnv`. The episode ends at the first decisive reference event. Rewards are the repo's own `match_reward`: zero-sum per step and telescoping to the ending utility. A blocked bout returns 0 and sets `info["blocked"]`. |
| `grokbot/selfplay.py` | The league, described below. |
| `grokbot/promotion.py` | Side-swapped pairs that share a seed now count as **one** paired sample (F2). |

**`grokbot/selfplay.py` in detail:**

- **Policy.** A linear policy over 10 muscle regions, derived from tendon
  attachment bodies, drives all 700 muscles. Each region gets a tone plus
  feedback gains on 13 features of the actor's **own** pelvis and trunk:
  tilt, velocity and height. That is 140 parameters. It never reads opponent
  state.
- **Learner.** Antithetic evolution strategies with centred ranks. Each
  antithetic pair shares opponent, seed and corner (common random numbers).
- **Opponents.** Drawn from `grokbot.opponents`: latest (self-play),
  PFSP-weighted archived checkpoints, and scripted anchors.
- **Archive and promotion.** Every generation is frozen and archived. Each is
  evaluated with seeded action noise in side-swapped pairs against **all
  earlier checkpoints** and the scripted panel. Promotion is decided against
  the champion.
- **Bounds.** At most 4 one-thread workers. Runs used `nice -n 10`.

## Result: 6 generations

Configuration: 6 generations × 3 ES iterations × 8 antithetic pairs, so 48
training games per generation. Evaluation uses 6 noisy seed pairs, 12 games,
per opponent. Horizon 3 s, seed 1, 4 workers. Wall time 1,146 s. In all, 288
training games and 588 evaluation games ran, with 0 blocked and 0 invalid.

| Gen | vs previous | vs champion | Beats ALL earlier? | vs 4 scripted | Gate | Champion | Side balance (gen) | Median training bout |
|---|---:|---:|---|---|---|---|---:|---:|
| 1 | +1.00 | gen0 +1.00 | yes | +1.00 each | promote | gen1 | 0.000 | 1.24 s |
| 2 | +1.00 | gen1 +1.00 | yes | +1.00 each | promote | gen2 | 0.000 | 1.26 s |
| 3 | +0.67 | gen2 +0.67 | yes | +1.00 each | refuse (lower bound −0.33) | gen2 | 0.000 | 1.34 s |
| 4 | +1.00 | gen2 +1.00 | **yes, all 4 at +1.00** | +1.00 each | promote | **gen4** | 0.000 | 1.44 s |
| 5 | −1.00 | gen4 −1.00 | **no**: loses to gen4 | +1.00 each | refuse (−2.00) | gen4 | 0.000 | 1.52 s |
| 6 | +0.67 | gen4 −0.33 | **no**: loses to gen4 | +1.00 each | refuse (−1.33) | gen4 | 0.000 | 1.46 s |

**Ledger Elo per generation** is anchored at the scripted `baseline` = 0 and
refitted on all games so far:

| After gen | gen0 | gen1 | gen2 | gen3 | gen4 | gen5 | gen6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | −431 | +431 | | | | | |
| 2 | −353 | +345 | +729 | | | | |
| 3 | −308 | +302 | +622 | +823 | | | |
| 4 | −275 | +275 | +546 | +710 | +1084 | | |
| 5 | −251 | +257 | +495 | +637 | +1246 | +921 | |
| 6 | −231 | +242 | +478 | +591 | **+1081** | +787 | +905 |

The scripted panel ends at relaxed −408, cocontract −567 and noise −139.
`baseline` was the best scripted policy, ahead of all three others. **Every
generation beats every scripted policy in all 12 games.**

## Red-team findings

### F1 — High — The self-play signal is real and corner-fair, but it is a balance contest

- Side balance is exactly 0.000 in every generation, and cumulatively. Each
  side-swapped pair splits evenly, so the signal is not a corner artifact.
- Learning is monotone up to gen4. Later generations beat **all** earlier
  checkpoints until gen4, which is the owner's anti-cycling question.
- After gen4 the learner regresses. gen5 and gen6 still beat gen0–gen3, but
  lose to gen4. That is not a rock-paper-scissors cycle. It is overshoot
  around a peak, and the champion gate holds gen4. A latest-only view would
  have reported gen6 as "+0.67 vs previous, improving".
- **Degenerate passive equilibrium.** All 288 training and 588 evaluation
  games have zero contact ticks. Every bout ends in a self-caused fall
  between 1.12 and 1.68 s. The median training bout grows from 1.24 s to
  1.52 s.
  - The agents learn postural tone and feedback that delay falling.
    Approaching the opponent would be pure risk, so the league converges on
    standing still.
  - Once both athletes reliably stand until the bell, every bout becomes a
    0–0 bell draw and the signal vanishes again.

**Next steps (jekko, P2/P6):**

1. Register a facing reset at boxing range.
2. Use the reference ending, or adopt a successor of it.
3. Add a contact-incentive guard to the rules before more self-play.
   Examples: a rule against passivity, a ring that shrinks, or points
   that can actually be reached.

### F2 — High (fixed here) — Deterministic evaluation is not independent evidence

The learned policies are deterministic, and evaluation seeds only reseed
generators those policies never read. "8/8 wins" in my first two runs was
**one game repeated eight times**, and the side-swap mirror adds nothing to it.
That overstated the Hoeffding gate and the Elo gaps. It also let a
sign-flipped learner pass the first smoke test; mutation testing caught this.

Fixes:

- Evaluation adds seeded Gaussian action noise (σ = 0.05) from each policy's
  own common-random-number stream, so seeds give genuinely different games.
- The promotion gate now counts a side-swapped pair as one paired sample.

The two earlier runs are kept as development evidence under
`runs/grokbot/selfplay-0014-{prefix,deterministic-eval}`. Their 8/8 figures
must not be cited.

### F3 — Medium (fixed here) — Promoting against "previous" instead of the champion

The first run gated gen4 against the refused gen3. Promotion now always
compares against the champion. Evaluating against every earlier checkpoint
makes regressions visible, as in gen5 and gen6.

### F4 — Low — Region map asymmetry

The tendon-attachment region map gives `arm_l` 88 and `arm_r` 71 + `arm_c` 17
muscles, and `neck_l` without a `neck_r`. MS-Human-700 muscle names are not
suffixed consistently. Both corners use the same map, so there is no corner
bias. The parameterization is not left/right symmetric, though.

## Tests and validation

- `test_reference_env.py` has 16 tests. Chunked `BoutMonitor` equals the
  batch functions on 12 synthetic bouts, and gaps or reordering are rejected.
  In the real env it checks: the first self-caused fall pays the winner +1
  with a telescoped, zero-sum return; a mirror bout draws; short bouts end at
  the bell; and the blocked stress bout returns 0.
- `test_selfplay.py` has 8 tests:
  - Regions drive each muscle exactly once.
  - The policy reads only the actor's own features.
  - Commands stay bounded and re-zero each episode.
  - Evaluation noise is seeded, and a deterministic policy rejects missing
    generators.
  - Rank shaping gives a flat update on flat fitness.
  - More than 4 workers is refused.
  - **CI smoke test:** one fixed-seed generation, with 24 training games and
    10 noisy evaluation pairs, must beat its random initialization at
    ≥ +0.8 and beat all earlier checkpoints.

  The smoke test was calibrated on seeds 0–2:

  | Update | Seed 0 | Seed 1 | Seed 2 |
  |---|---:|---:|---:|
  | Real (lr +0.05) | +1.0 | +1.0 | +1.0 |
  | lr = 0 | +0.6 | −0.1 | +0.4 |
  | Negated lr | +1.0 | −1.0 | −1.0 |

  At the test seed (1), both non-learning mutants fail the threshold. Seed 0
  is a local-minimum initialization where any step helps.
- `test_promotion.py`: a new test checks that mirrored pairs sharing a seed
  count once.
- Full local gate: see the PR body. Ruff format and check are clean.

## Reproduce

```bash
PYTHONPATH=. nice -n 10 uv run --project experiments/v0.0.11/runtime --locked \
    python grokbot/redteam/probes/cycle-0014-selfplay-league.py runs/grokbot/selfplay-0014
```

`grokbot/redteam/data/cycle-0014-league-summary.json` holds the per-generation
rows, all 7 checkpoints' parameters, the evaluation totals, and the SHA-256 of
the full report. The full report is under `runs/`, which is gitignored.
