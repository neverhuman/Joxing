# Grokbot red-team — cycle 0038 / v0.0.25 (2026-10-28): the contact penalty works against its training opponent and does not transfer

Reviewed `origin/main` at `5dd6bdb`, the v0.0.25 preregistration (#37,
protocol `9b65e82e…`).

The run was launched at 21:51:45Z by **one gated command**, which waited
for main CI on `5dd6bdb` to succeed **and** for load to drop below 70 (it
launched at load 24). It took 3,395 s.

## v0.0.25 — registered level-up **NOT met**; not tagged

The one lever was a graded penalty, `fitness −= 0.5 · min(1, contact_ticks /
500)`, starting from v0.0.22 and trained against frozen v0.0.19.

**Smoke gate: PASSED.** Over 24 base trials against v0.0.19, contact ticks
ran from 188 to 559, so the penalty term varies. The bonus fired 24/24.
The 48-game base head-to-head against v0.0.19 was **48/48 valid**
(47–1).

| Matchup (48 games; seats A and B identical) | Valid | W–L–D | No-contest | Blocked | Rule | Result |
|---|---:|---|---:|---:|---|---|
| **Primary: vs frozen v0.0.19, seeds 28000–28023** | **48** | **48–0–0** | 0 | 0 | ≥ 40 valid, ≥ 24 decisive, b, c | **PRIMARY_PASSED** (all 48 wins with hits; Elo +669, lower bound +191) |
| **Hard gate: vs frozen v0.0.13, seeds 29000–29023** | **0** | — | 0 | **48** | ≥ 40 valid | **failed** (v0.0.22: 36/48) |
| Monotonicity: @48 vs @16, seeds 30000–30023 | 38 | 34–4–0 | 10 | 0 | ≥ 40 valid, Wilson > 0.55 | **failed** (validity) |
| Monotonicity: @16 vs the v0.0.22 base, seeds 30000–30023 | 48 | **4–44–0** | 0 | 0 | ≥ 40 valid, Wilson > 0.55 | **failed**: @16 lost to its base |
| v0.0.19 mirror, primary seeds (secondary) | 6 | 2–0–4 | 36 | 6 | | — |
| vs frozen v0.0.16 (secondary) | 44 | 44–0–0 | 2 | 2 | | — |

**Motor gate against gen4:** stance 0/24, hits 22/24, median stand 1.44 s.
The v0.0.19 gate re-check reproduced.

## Contact rate against frozen v0.0.19 (owners 21:50 and 22:05): **not gamed by avoidance**

| | Contact ticks before the opponent's fall | Knockdowns or wins |
|---|---:|---|
| v0.0.22 base, smoke trials | median **380** | valid-knockdown 24/24 |
| Training, iteration 0 → 47 | **353 → 135** (−62 %) | valid-knockdown share **0.75–0.83 throughout** |
| Primary games, the trained policy | median **113** | **48–0**, every win with hits, 0 blocked |

The policy cut contact sharply **while keeping its knockdowns and wins**.
That is the intended effect, not avoidance: a contact-dodging policy would
lose its knockdowns and fail the decisive floor of at least 24/48.

## Findings

1. **High: the lever works against its training opponent and does not
   transfer.**
   - Against v0.0.19, the opponent it trained on, games are 48/48 valid
     with no blocks.
   - Against v0.0.13, which it never saw, **every game is blocked** (48/48),
     with median contact of 285 ticks. v0.0.13 falls earlier, at about
     1.57 s against v0.0.19's 1.78 s, while the candidate is still in
     contact.
2. **High: training is non-monotone.** The 16-iteration checkpoint lost
   **4–44 to its own v0.0.22 start**. The 48-iteration checkpoint beat @16
   34–4, but on only 38/48 valid games. Early training against the penalty
   degraded play before partly recovering it.
3. **Registered reading applied:** "hard gate still fails on blocked games
   vs v0.0.13 → next lever: alternate both training opponents".

## Next: v0.0.26, its own preregistration PR

**One lever:** train against v0.0.19 **and** v0.0.13, alternating per ES
pair. This needs a small `motor_gate.train` change to take an opponent
list. Everything else stays as in v0.0.25: the contact penalty, the
valid-knockdown bonus, the v0.0.22 start, the smoke gate on both opponents,
and the same ordered gates on new seeds.

## What this PR adds

- `data/cycle-0037-contact-penalty-gate-summary.json`: smoke results;
  per-game records for every matchup, including contact ticks; the
  contact-rate table; the training history; the final parameters; and the
  report SHA-256.
- This report, the CHANGELOG entry and STATUS.
