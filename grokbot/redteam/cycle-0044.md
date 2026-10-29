# Grokbot red-team — cycle 0044 / v0.0.28 (2026-10-29): self-play recovers by @48, but the base gets only a quarter of the early update

Reviewed `origin/main` at `d0de0d9`. jekko's four new commits cover native
tape qualification and face-audit work; none touches `grokbot/`. CI on
`9850f06` is green, and `d0de0d9` was still running at review time. The
v0.0.28 protocol (`90740cde…`, #43) landed before the run. The run launched
at 06:07:34Z in one command gated on main CI on `07952c8` being green and
load 69.2, ran at nice 19 with no pausing, and took 4,092 s. Attempts: 1,
which completed.

## v0.0.28 — registered level-up **NOT met** (monotonicity); not tagged

The one lever was the frozen v0.0.22 start joining the training opponents.
ES pairs cycle through v0.0.19, v0.0.13 and v0.0.22.

| Matchup (48 games) | Valid | W–L–D | Blocked | Seat A | Seat B | Result |
|---|---:|---|---:|---|---|---|
| Smoke gate (penalty varies vs both external opponents; base h2h 47/48 valid) | | | | | | **passed** |
| **Primary: vs frozen v0.0.19** | 44 | **44–0–0** | 4 | 22–0 / 2 blk | 22–0 / 2 blk | **passed** (28/44 wins with hits) |
| **Hard gate: vs frozen v0.0.13** | 48 | **48–0–0** | 0 | 24–0 | 24–0 | **passed** |
| Monotonicity: @48 vs @16 | 48 | **46–2–0** | 0 | 23–1 | 23–1 | **passed** |
| Monotonicity: @16 vs the v0.0.22 base | 48 | **2–46–0** | 0 | 1–23 | 1–23 | **failed** |
| v0.0.16 (secondary) | 48 | 48–0–0 | 0 | | | — |

**Motor gate against gen4:** stance 1/24, median stand 1.98 s.

`level_up = false` because @16 lost to its start, and there is **no tag**.
For the first time, more training beats less over the second stretch
(@48 vs @16: 46–2). In v0.0.26 that matchup was 24–24, and in v0.0.27 it
was 11–37.

## Diagnostic (post hoc, non-gating; `data/cycle-0044-train-eval-agreement.json`)

**Q1. Does the training measurement agree with head-to-head against the
base?** I ran `motor_gate.trial` with the training fitness on seeds
42000–42023:

| Against the base | Trial fitness | Opponent falls first | Self falls first | Head-to-head |
|---|---:|---:|---:|---|
| base (mirror) | 0.99 | 11/24 | 13/24 | — |
| @16 | **0.69** | 5/24 | **19/24** | lost 2–46 |
| @48 | **1.43** | **20/24** | 4/24 | **43–5 post hoc** (Wilson 0.78–0.95) |

**Yes, they agree.** The registered reading for this outcome ("the solo
fitness rewards behaviour that loses head-to-head; next lever: a
match-outcome term") is **refuted**. The fitness already ranks the base,
@16 and @48 in the same order head-to-head does, so an outcome term would
be nearly collinear with the existing valid-knockdown term. The post-hoc
43–5 is **not a gate and cannot rescue a tag**.

**Q2. Why does @16 dip?**

| Trial fitness | vs base | vs v0.0.19 | vs v0.0.13 |
|---|---:|---:|---:|
| base | 0.99 | 1.50 | 1.50 |
| @16 | **0.69 ↓** | 1.59 ↑ | 1.73 ↑ |
| @48 | **1.43 ↑** | 0.99 ↓ | 1.65 |

Over the first 16 iterations, ES **traded the base matchup for the two
frozen opponents**. It reversed that trade over the next 32 iterations.

## Findings

1. **High: the base got about a quarter of the update, and the frozen
   opponents' direction won.**
   - `motor_gate.train` ranks all pairs' fitness together, whichever opponent
     each pair faced. Against the base, fitness sits around 0.4–0.75.
     Against v0.0.19 and v0.0.13 it sits around 1.0–1.7.
   - I first estimated the base's share at about 20 %. Then I **measured**
     it (`probes/cycle-0044-rank-shares.py`: same task construction and
     fitness, lr 0, rng 99441, 2 batches × 12 pairs at each point):

     | Base's share of Σ\|Δrank\| | At the v0.0.22 start | At v0.0.28@16 |
     |---|---|---|
     | Pooled ranks (as trained) | 0.28, 0.27 | **0.23, 0.26** |
     | Per-opponent ranks | 0.33, 0.33 | 0.26, 0.32 |
     | Nominal | 0.33 | 0.33 |

   - **So the pooled bias is real but modest:** about 0.26 against 0.33.
     Per-opponent ranks would only lift the base to about 0.31. That is too
     small to expect a 2–46 dip to close, so **it is not the next lever**.
     It is recorded for later.
   - The direct reading is that, at about a quarter of the update, the base
     matchup lost to the two frozen opponents' direction over the first 16
     iterations (the trial table above).
2. **Medium: the registered reading was wrong.** I record it and do not
   follow it, because the Q1 diagnostic refutes its premise.
3. **Positive:** this is the first version whose final policy clearly beats
   its own start post hoc (43–5), and the first with @48 > @16.

## Next: v0.0.29, its own preregistration PR, opened off main

**One lever: the base faces half of the ES pairs.** The rotation becomes
v0.0.19, base, v0.0.13, base, so of the 12 pairs, 6 face the base and 3 face
each frozen opponent. The expected update share for the base is about 0.4,
against the 0.23–0.28 measured above. Everything else stays as in v0.0.28.
`motor_gate.train` gains opt-in per-slot rank-share logging, so the smoke
gate and history can **measure** the lever.
- **Risk:** each frozen opponent gets 6 trials per iteration instead of 8,
  so the primary or hard gate may regress.

## What this PR adds

- `data/cycle-0043-self-play-opponent-gate-summary.json`: per-trial and
  per-game records with seat splits, the history, the @16 and final
  parameters, the full-report SHA-256, and the attempts ledger.
- `data/cycle-0044-train-eval-agreement.json` and the three diagnostic
  probes that produced it (`probes/cycle-0044-trial-vs-*.py` and
  `probes/cycle-0044-rank-shares.py`).
- This report, the CHANGELOG entry (with the unpublished-versions table
  extended) and STATUS.
