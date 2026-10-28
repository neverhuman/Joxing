# Grokbot red-team — cycle 0016 / v0.0.12 (2026-10-27): breaking the passive equilibrium

Reviewed `origin/main` at `4ed9133`, with tags `v0.0.11` and `grokbot-v0.0.11`.
jekko has made no commits since `1978da7`. The owner's 22:24 instruction for
v0.0.12 was:

1. Preregister an engagement gate and an anti-stalling rule that does not
   reward stalling.
2. Re-run the champion gate. Elo gains with zero contact do not count.
3. Explain the gen5/gen6 regressions with a fresh-seed check.

## Preregistration

`grokbot/redteam/protocols/v0.0.12.json` (sha256 `33ea705a…`) was pushed at
`ec99623` on 23:03Z, before any v0.0.12 outcome existed. It fixes:

| Element | Registered value |
|---|---|
| Attribution and anti-stalling | `grokbot-ending-reference-v2` plus **anti-stalling**: a result reached with no landed strike by either corner and no points is a **no contest** (no utility, return 0), so outlasting earns nothing |
| Reset | `grokbot-range-0.80m-v0`, i.e. `RangeBoxingEnv`, which uses the env's own reset-only root shift; nothing else changes |
| Training | Starts from the v0.0.11 champion gen4; **training-only** approach shaping, weight 0.5, rewarding a reduced own-hand-to-opponent-head or trunk distance; 6 generations × 3 ES iterations × 8 antithetic pairs; seed 2 |
| Evaluation | 6 noisy, side-swapped seed pairs against every earlier checkpoint and the scripted panel |
| Engagement gate | The candidate lands a strike in **≥ 25 %** of its evaluation bouts |
| Success criterion | At least one generation is promoted under the engagement gate |
| Pre-declared failure reading | "Objective change alone does not break the equilibrium; the next lever is motor capability" |

## Before training

### The v0.0.11 Elo does not survive the new rules

Probe `cycle-0016-recheck-v11.py` re-ran gen0–gen6 under v3 at 0.80 m. Each
checkpoint played every earlier one and the scripted panel: 588 games in all.
Every one had 0 contact ticks and every one was a no-contest, so engagement
was 0.000 for all seven checkpoints. Every promotion in the v0.0.11 lineage
(gen1, gen2 and gen4) is now **refused**, on coverage and on engagement.

### Physical finding: the native keyframe falls backward

The heads move about 1.3 m away from the opponent within 1.2 s. The probe
found no contact at 1.0, 0.8 or 0.6 m for gen4, baseline, relaxed against
cocontract, or noise. A closer reset alone cannot create engagement.

### Fresh-seed regression check (`cycle-0015-regression-check.py`, seeds 100–111)

| | gen2 | gen3 | gen4 | gen5 | gen6 |
|---|---:|---:|---:|---:|---:|
| Median standing time (own first non-foot support, vs baseline) | 1.373 s | 1.437 s | **1.517 s** | 1.503 s | 1.490 s |

- On fresh seeds gen4 still beats gen5 (+0.50) and gen6 (+0.83).
- gen5 and gen6 still beat gen2 at −1.00 from gen2's side, and they beat
  gen3 as well.
- The regressions are small backward steps on the single trait the
  objective rewards, standing time, after a plateau near 1.5 s. Once a
  learner beats the whole archive, its ES fitness is almost all ±1
  self-comparisons, and it drifts.
- This is **not overfitting to gen4**. The refusals stand.

## Result (attempt 03; attempts 01 and 02 failed and are retained)

| Gen | Training contact ticks | Mean approach | Median bout | Evaluation contact ticks | Engaged | No-contest | Blocked | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 0.153 | 1.34 s | 0 | 0.000 | 1.000 | 0 | refuse |
| 2 | 57 (1 engine fault quarantined) | 0.238 | 1.26 s | 0 | 0.000 | 1.000 | 0 | refuse |
| 3 | 0 | 0.271 | 1.28 s | 0 | 0.000 | 1.000 | 0 | refuse |
| 4 | 22 | 0.281 | 1.33 s | 4,094 | 0.000 | 0.604 | 0.396 | refuse |
| 5 | 2,000 | 0.312 | 1.18 s | 4,190 | 0.000 | 0.741 | 0.259 | refuse |
| 6 | 5,030 | 0.322 | 1.08 s | 5,838 | 0.000 | 0.700 | 0.300 | refuse |

The run played 588 evaluation games. Of those, **0 were valid results**, 486
were no-contests and 102 were blocked; none hit an engine fault. They
accumulated 14,122 contact ticks, but the candidate landed **0 strikes**. No
generation was promoted and there is no Elo. Wall time was 1,024 s with 4
nice-10 workers.

**Against the preregistered criterion, v0.0.12 is not a level-up.** No
version tag is claimed for it.

## Findings

### F1 — High — Approach shaping was gamed: the agents lean in instead of punching

Contact did emerge. Training contact went from 0 to 5,030 ticks by gen6, and
the hands ended up 32 % closer to the opponent. Breaking the *zero-contact*
equilibrium was easy.

What the agents learned is not boxing. In four gen6 mirror bouts, all 1,156
inter-athlete contact points were `native_non_scoring ↔ native_non_scoring`.
There were 0 glove contacts and 0 referee events. "Own hand near the
opponent's head or trunk" was satisfied by leaning and collapsing the whole
body forward into the opponent. Attribution v2 classifies that as body-on-body
contact (clinch or shove), so those bouts are **blocked** rather than scored.
Bouts also got shorter, from 1.34 s to 1.08 s, because leaning in costs
balance.

The engagement gate and anti-stalling worked as designed. They refused
stalling, and they refused leaning. Every safeguard held, and none of them
can create skill.

**Next lever (pre-declared):** motor capability.

- A glove-strike primitive, meaning a reach or punch synergy with an arm
  trajectory target, or the pretrained-motor bridge's stance plus a reach
  adaptation.
- Any further shaping must be on **glove** velocity toward a target zone,
  measured with the referee's own approach and active speed, not on hand
  proximity, which body lean satisfies.

### F2 — Medium (fixed) — The league crashed on two classes of non-results

These were preregistered-run failures, retained under `runs/grokbot/` as
`engagement-0016-attempt01-crash` and `…-attempt02-engine-fault`.

| Attempt | Failure | Fix |
|---|---|---|
| 01 | `Ledger.ratings` raised when every bout was a no-contest | `safe_ratings` reports an empty Elo |
| 02 | A MuJoCo "Invalid physics substep" in generation-2 training killed `pool.map` | `play_game` returns an explicit invalid record. ES drops quarantined pairs, so a fault never counts as a result. Evaluation keeps invalid games in the denominators. |

Both fixes are reporting and quarantine only, with the same protocol and
seed. Attempt 03 replayed the same fault in generation 2 and quarantined it.

MuJoCo's own log, retained as `runs/grokbot/engagement-0016/MUJOCO_LOG.TXT`,
pins the fault down: "Nan, Inf or huge value in QACC at DOF 85" at
t = 0.715 s, identically in attempts 02 and 03. DOF 85 is the first degree of
freedom of athlete B, i.e. B's root. It is a deterministic instability under
one perturbed training policy, not hardware noise. The same quarantine path
would be needed by any learner that explores aggressive muscle commands.

### F3 — Low — Gate wording

At gen4–6, blocked bouts appear under "integrity: invalid share". They are
contact without resolvable attribution, not broken engines. The gate already
reports engagement separately. A `blocked_share` line of its own would read
more clearly.

## What shipped

| Module | Change |
|---|---|
| `endings.anti_stalling` | New |
| `ContactClassifier.strikes_by` | Per-corner strike attribution |
| `scenarios.RangeBoxingEnv` | New |
| `ReferenceEndingEnv` | New `anti_stalling=True` option |
| `selfplay` | Range and anti-stalling workers, approach shaping, engagement fields, `init_params`, `safe_ratings`, engine-fault quarantine |
| `promotion` | `min_engaged_share`; no-contests excluded from the broken-game share |

There are 15 new tests in `test_engagement.py`:

- The anti-stalling boundaries.
- Per-owner strike credit.
- The range reset: separation, contract identity and side symmetry.
- No-contest termination in the real env.
- The engagement gate.
- No-contests not counted as broken games.
- Reach.
- A no-contest league with no ratings.
- ES dropping a quarantined pair, checked against an exact expected update.
- A physics fault quarantined instead of crashing.

The protocol, the three probes, `data/cycle-0016-engagement-summary.json`
(per generation, all checkpoints, attempt history, report SHA-256) and this
report are all included.
