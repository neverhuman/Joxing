# Grokbot red-team — cycle 0040 / v0.0.26 (2026-10-29): the hard gate passes for the first time; monotonicity fails

Reviewed `origin/main` at `7579a5f`, the v0.0.26 preregistration (#39,
protocol `ed132adf…`).

The run was launched at 00:56:30Z by one gated command, after main CI on
`7579a5f` succeeded, at load 48. It took 3,922 s.

## v0.0.26 — registered level-up **NOT met** (monotonicity); not tagged

The one lever was training against **both** frozen opponents, v0.0.19 and
the v0.0.13 champion, alternating per ES pair. Everything else followed the
v0.0.25 recipe: the contact penalty, the valid-knockdown bonus, and the
v0.0.22 start.

**Smoke gate: PASSED.** The penalty varied against both opponents: contact
of 279–582 ticks against v0.0.19 and 24–117 against v0.0.13. The 48-game
base head-to-head against v0.0.19 was 48/48 valid.

| Matchup (48 games; seats A and B identical) | Valid | W–L–D | No-contest | Blocked | Rule | Result |
|---|---:|---|---:|---:|---|---|
| **Primary: vs frozen v0.0.19, seeds 32000–32023** | 46 | **46–0–0** | 0 | 2 | ≥ 40 valid, ≥ 24 decisive, b, c | **PRIMARY_PASSED** (all wins with hits) |
| **Hard gate: vs frozen v0.0.13, seeds 33000–33023** | **48** | **48–0–0** | 0 | **0** | ≥ 40 valid | **PASSED: the first time** (v0.0.22: 36/48; v0.0.25: 0/48) |
| Monotonicity: @48 vs @16, seeds 34000–34023 | 48 | **24–24–0** | 0 | 0 | ≥ 40 valid, Wilson > 0.55 | **failed** (Wilson 0.36–0.64) |
| Monotonicity: @16 vs the v0.0.22 base, seeds 34000–34023 | 42 | **2–40–0** | 0 | 6 | ≥ 40 valid, Wilson > 0.55 | **failed**: @16 lost to its base |
| v0.0.19 mirror (secondary) | 1 | 0–0–1 | 42 | 5 | | — |
| vs frozen v0.0.16 (secondary) | 48 | 48–0–0 | 0 | 0 | | — |

**Motor gate against gen4:** stance **11/24**, hits 24/24, median stand
**2.82 s**, the best since v0.0.19 (v0.0.25 had 0/24 and 1.44 s).

**Contact rate,** the owner's standing check: the training mean went from
426 to 306, a mix of both opponents. In primary games against v0.0.19 the
median was 483; against v0.0.13 it was 98. This is not avoidance: every
valid game in both gates was won with hits.

## Findings

1. **Positive: training against both opponents fixed the blocked games.**
   Against v0.0.13 the hard gate went from 0/48 to **48/48 valid**, without
   losing the primary.
2. **High: training is still non-monotone, and this is the binding
   failure.** The 16-iteration checkpoint lost **2–40 to its own start**;
   v0.0.25's @16 lost 4–44 the same way. The 48-iteration policy only
   **ties** @16 at 24–24. Early ES steps under this objective consistently
   make the policy worse head-to-head before anything else improves. A
   policy that does not beat its own starting point cannot be called a
   level-up, even though it beats both frozen opponents.
3. **Registered reading applied:** "monotonicity fails with @16 < base
   again: early-training collapse; next lever: smaller ES step (lr)".

## Next: v0.0.27, its own preregistration PR, right away

**One lever:** the ES learning rate drops from **0.05 to 0.02**. Everything
else stays as in v0.0.26: both opponents, the contact penalty, the
valid-knockdown bonus, the v0.0.22 start, and the same smoke gate and
ordered gates on new seeds.

## What this PR adds

- `data/cycle-0039-two-opponent-gate-summary.json`.
- This report.
- The CHANGELOG entry, which also clarifies every unpublished version
  (owner 01:10).
- STATUS.
