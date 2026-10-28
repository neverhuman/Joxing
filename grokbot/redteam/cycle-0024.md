# Grokbot red-team — cycle 0024 / v0.0.18 (2026-10-28): real hits, lost stance; beats v0.0.13, loses to v0.0.16

Reviewed `origin/main` at `1dc96e9`. jekko has made no new commits. v0.0.17
shipped as a pre-release labelled "criterion not met" (owners 05:46 and
06:10).

The v0.0.18 protocol (`159642dd…`) landed in #23 before its results were
read. The run's tree `ad49ce58` differs from main `1dc96e9` only in docs:
CHANGELOG.md, STATUS.md and cycle-0023.md.

Owner 06:28 asked for a gate against the frozen v0.0.13 champion. Its
addendum (`protocols/v0.0.18-addendum-v0013.json`, sha256 `6ea4b7e1…`) and
its probe were pushed to this branch at 06:48:16Z, as `5b4d9b1` and
`82bce8c`. That was **before any v0.0.18 gate or head-to-head result was
read**; only the training-iteration logs had been seen.

## v0.0.18 — registered level-up **NOT met**; addendum gate vs v0.0.13 **passed**; not tagged

The candidate was `GroupPDPolicy`, which adds per-group velocity and height
gains to v0.0.17 for 2,078 parameters. It was trained for 48 ES iterations
of 12 pairs with a **hit-only** strike bonus. The run took 3,030 s at nice
19. The v0.0.17 gate re-ran identically per trial.

### Motor gate (registered)

| | Stance | Reach (any strike) | Head/torso hit | Median stand |
|---|---:|---:|---:|---:|
| v0.0.17 (re-run, identical) | 7/24 | 24/24 | **0/24** | 2.682 s |
| **v0.0.18** | **1/24** | 24/24 | **24/24** | 2.314 s |

The gate **failed** on stance. The sign test against v0.0.17 **failed**: only
3 of 23 trials were better, p = 1.0. In training, the hit share rose from
0.12 to 0.92 while mean stand time fell from 2.42 to about 2.0–2.2 s.

### Elo gate vs frozen v0.0.16 (registered, seeds 3000–3023, 48 games)

| Valid | W–L | Win rate (Wilson 95 %) | Mean | lower95 | Elo (lower95) | Wins with hits | Pass |
|---:|---|---|---:|---:|---:|---:|---|
| 44/48 (92 %) | 13–31 | 0.30 (0.18–0.44) | −0.41 | −0.93 | **−151** (−579) | 5/13 | **no**: (a) passed; (b) and (c) failed |

### Addendum Elo gate vs frozen v0.0.13 (preregistered, seeds 4000–4023, 48 games)

| Valid | W–L | Win rate (Wilson 95 %) | Mean | lower95 | Elo (lower95) | Wins with hits | Pass |
|---:|---|---|---:|---:|---:|---:|---|
| **48/48** | **48–0** | 1.00 (0.93–1.00) | +1.00 | **+0.50** | **+669 (+191)** | **48/48** | **yes**: (a), (b) and (c) all passed |

**Decision.** `level_up` is **false** under v0.0.18's registered criteria,
which require the motor gate and the v0.0.16 Elo gate. The addendum states,
as written before any result, that a v0.0.18 tag requires that registered
level-up. Owner 06:48's "tag only if the gate passes" sets the gate as a
necessary condition, not a sufficient one. **No tag.** A labelled
pre-release, like v0.0.17, can follow if the owner asks for one.

## Findings

### F1 — Positive — The hit-only bonus turned glove taps into real hits

In v0.0.17, 0 of 24 gate trials had a head or torso hit. In v0.0.18, 24 of
24 did. Against v0.0.13, all 48 wins include the candidate's own head or
torso hits. This closes cycle 0023's F1 for this policy. These are the
project's first games won with hits.

### F2 — High — Hitting costs balance: v0.0.18 falls first against a standing opponent

Against v0.0.16, the 31 losses end at a median tick of 1,700. The candidate
lands hits (21 of the 31 include its head or torso hits) and then **falls
first**, well before its solo median of 2.31 s. The 13 wins end at a median
of 1,920, when v0.0.16 falls first. Against v0.0.13, which never strikes and
falls at about 1.5 s, the candidate wins every game.

The v0.0.13 pass is therefore genuine, with real hits, valid games and
positive Elo, but it was won against the weaker frozen opponent. **Against
the strongest frozen policy, v0.0.18 is about 150 Elo worse.**

### F3 — Medium — v0.0.18 bundled two changes, so the stance regression is confounded

The damping gains and the hit-only fitness changed together. The training
curves show stand time falling as the hit share rose, which points at the
fitness trade-off rather than at the damping. Pre-declared reading 1 ("no
stand gain, so the linear feedback is saturated") **cannot be cleanly
applied** here. Future versions change one thing at a time, or include an
ablation arm.

## Next step (recommended for v0.0.19; not preregistered yet)

Target the loss mode in F2 directly with a single change. Train from
v0.0.18 with a **"hit and stay up"** fitness: the hit bonus counts only if
the athlete is still up 0.5 s after the hit. Keep the policy class and the
gates unchanged, and use the same three-gate report:

- the motor gate;
- the Elo gate against v0.0.16;
- the Elo gate against v0.0.13.

Tag only on the registered level-up.

## What this PR adds

- `protocols/v0.0.18-addendum-v0013.json` and
  `probes/cycle-0024-v0013-elo-gate.py`, both pushed before the results were
  read.
- `data/cycle-0023-group-pd-gate-summary.json`: per-trial and per-game
  records for all three gates, the training history, the final parameters,
  and SHA-256 hashes of the full reports.
- This report and the STATUS update.

There are no code changes to grokbot modules.
