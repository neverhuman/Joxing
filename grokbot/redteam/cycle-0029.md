# Grokbot red-team — cycle 0029 (2026-10-28): v0.0.21 preregistration (validity first: train against v0.0.19)

Reviewed `origin/main` at `090a0a8`. jekko has made no new commits. v0.0.20
landed as an honest null in #28, and v0.0.19 remains the last level-up. Per
owners 11:46 and 12:05, this PR lands the v0.0.21 preregistration **on its
own, before any run**.

## The one lever: the training opponent (`protocols/v0.0.21.json`)

| | v0.0.19 | **v0.0.21** |
|---|---|---|
| Training opponent in `motor_gate.train` | gen4 posture: falls at about 1.46 s, never guards | **frozen v0.0.19**: stands 3 s, eval noise 0.05 |
| Unchanged | | Fitness `stand/3000 + hit_and_stayed_up(500)`, `GroupPDPolicy`, initialization (v0.0.19), 48 × 12 ES, the 0.5 m reset, ending v2 |

**Why this lever.** Cycle 0028 found two things:

- v0.0.19 lands head hits against the gen4 posture, but makes **0 strikes**
  against a standing copy of itself: 40 of 48 games were no-contests.
- The league never got more than 1 valid game against v0.0.19 per
  generation.

Owner 11:46 suggested sampling from a pool of v0.0.13, v0.0.16 and v0.0.19
with a minimum-valid constraint. v0.0.13 and v0.0.16 are already beaten 48–0
in 100 %-valid games, so they would add no ranking signal. The narrowest
form of that idea is v0.0.19 on its own. The fitness is still the solo motor
fitness, with no match outcome in it, so this is **not** competitive
learning.

## Primary: validity first, then win rate (owner 12:05)

The primary plays frozen v0.0.19 on **new seeds 10000–10023**, both sides,
48 games.

1. **Validity precondition: at least 40 of 48 valid games.** It is checked
   **before** any win-rate criterion. If it fails, the verdict is
   **`VALIDITY_FAILED`**: the win-rate criteria are not evaluated and the
   probe **exits non-zero** (fail loud). That is a failure, not a null and
   not a pass.
2. **Only if validity holds:**
   - (b) Hoeffding lower95 > 0, which is the same as an Elo lower bound > 0;
   - (c) at least half of the candidate's valid wins include its own head or
     torso hits.

**`v0.0.21` is tagged only if the verdict is `LEVEL_UP`.**

**Secondary, reported but not gating:**

- v0.0.19's self-mirror on the primary seeds, as the baseline validity
  (expected near zero);
- the candidate against frozen v0.0.16 and v0.0.13 on seeds 11000–11023;
- the motor gate against gen4, with v0.0.19 re-checked per trial;
- training curves against v0.0.19.

The pre-declared readings cover each verdict, and separate a
`VALIDITY_FAILED` caused by no-contests from one caused by blocks.

## Probe

`grokbot/redteam/probes/cycle-0029-mirror-opponent-gate.py`:

- reuses the v0.0.19 `head_to_head` unchanged;
- takes every seed set from the protocol;
- saves `report.json` after every stage.

A smoke test used a protocol copy with every seed set overridden to
99990–99993 and training shrunk to 1 × 1. It exercised the fail-loud path:
the run exited non-zero with "VALIDITY_FAILED: 0/2 valid primary games <
40", and the saved report was complete. **No game has been played on seeds
10000–10023 or 11000–11023.**
