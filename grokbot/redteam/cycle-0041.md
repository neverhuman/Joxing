# Grokbot red-team — cycle 0041 (2026-10-29): v0.0.27 preregistration (a smaller ES step)

Branches from `origin/main` at `7579a5f`, independently of the v0.0.26
report PR (#40). It changes no CHANGELOG or STATUS text. This is **the
preregistration on its own**: the run starts only after merge, through one
command gated on main CI being green and load below 70.

## Why this lever (cycle 0040)

v0.0.26 passed the primary (46–0 against v0.0.19) and, **for the first
time**, the v0.0.13 hard gate (48/48 valid). It failed **monotonicity**:

- the 16-iteration checkpoint **lost 2–40 to its own v0.0.22 start**;
- the 48-iteration policy only tied @16, 24–24.

v0.0.25 showed the same early collapse (@16 lost 4–44 to its start). Early
ES steps under this objective degrade head-to-head play. v0.0.26's
registered reading for this outcome was "early-training collapse → next
lever: smaller ES step (lr)".

## The one lever (`protocols/v0.0.27.json`)

| | v0.0.26 | **v0.0.27** |
|---|---|---|
| ES learning rate | 0.05 | **0.02** |
| Unchanged | | Two training opponents alternating per ES pair (frozen v0.0.19 and frozen v0.0.13); fitness `stand + valid_knocked_down − 0.5 · min(1, contact/500)`; the v0.0.22 start; σ 0.1; 48 × 12 (seed 17); the 0.5 m reset; ending v2 |

**Smoke gate** (the same as v0.0.26):

1. The penalty term varies against **each** opponent (seeds 35000–35047).
2. A 48-game base head-to-head against v0.0.19 is at least 90 % valid
   (seeds 35100–35123).

**Gates, in order, on new seeds:**

| Gate | Seeds |
|---|---|
| Primary against frozen v0.0.19 | 36000–36023 |
| Hard gate against v0.0.13 | 37000–37023 |
| Monotonicity: @48 vs @16, and @16 vs the v0.0.22 base | 38000–38023 |

**Tag only if every gate passes.** That would be the first level-up since
v0.0.19.

**Registered red-team notes:**

- A smaller step may simply move less. If @16 merely ties its base,
  monotonicity still fails the Wilson > 0.55 bar: **not collapsing is not
  improving**.
- The base already beats both frozen opponents, so monotonicity against the
  base is the discriminating gate.
- The next lever is already pre-declared: a **self-play opponent** (the
  policy's own frozen checkpoint), so that beating the base becomes the
  training signal.

## Code

There is no module change: the probe reads `lr` from the protocol.
`probes/cycle-0041-small-step-gate.py` is the v0.0.26 probe pointed at the
v0.0.27 protocol. **No game has been played on seeds 35000–38023.**
