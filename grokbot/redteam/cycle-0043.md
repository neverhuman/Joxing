# Grokbot red-team — cycle 0043 (2026-10-29): v0.0.28 preregistration (self-play opponent)

Branches from `origin/main` independently of the v0.0.27 report PR, and
changes no CHANGELOG or STATUS text. This is **the preregistration on its
own**: the run starts only after merge, through one command gated on main CI
being green and load below 70, at nice 19, with no SIGSTOP pausing.

## Why this lever (cycles 0038, 0040 and 0042)

| Version | vs frozen v0.0.19 | vs frozen v0.0.13 | **@16 vs its own v0.0.22 start** |
|---|---|---|---|
| v0.0.25 | 48–0 | 0/48 valid | **4–44** |
| v0.0.26 | 46–0 | 48–0 | **2–40** |
| v0.0.27 (lr 0.02) | 48–0 | 48–0 | **0–48** |

Each version beats the frozen opponents yet loses to its own start, and a
2.5× smaller step made that worse. So the collapse is not a step-size
effect: **the objective never rewards beating an opponent of the agent's own
strength.** v0.0.27's registered reading for this outcome was "next lever:
the self-play opponent".

## The one lever (`protocols/v0.0.28.json`)

| | v0.0.26 | **v0.0.28** |
|---|---|---|
| Training opponents, cycled per ES pair | frozen v0.0.19, frozen v0.0.13 | frozen v0.0.19, frozen v0.0.13, **frozen v0.0.22, the start itself** |
| Unchanged | | Fitness (stand + valid knockdown − 0.5 · min(1, contact/500)), the v0.0.22 start, **lr 0.05** (v0.0.27's 0.02 is not carried over), σ 0.1, 48 × 12 (seed 18), the 0.5 m reset, ending v2 |

**Smoke gate:**

1. The penalty term varies against each external opponent (seeds
   39000–39047).
2. A 48-game base head-to-head against v0.0.19 is at least 90 % valid
   (seeds 39100–39123).

**Gates, in order, on new seeds:**

| Gate | Seeds |
|---|---|
| Primary against frozen v0.0.19 | 40000–40023 |
| Hard gate against v0.0.13 | 41000–41023 |
| Monotonicity: @48 vs @16, and @16 vs the v0.0.22 base | 42000–42023 |

**Tag only if every gate passes.**

**Registered red-team notes:**

- **This trains the very matchup the monotonicity gate tests** (on
  different seeds). A pass shows the policy can improve on its start, not
  that it generalises beyond these three opponents.
- **Each external opponent now gets 8 trials per iteration instead of 12.**
- **The base-vs-base mirror is mostly a no-contest under ending v2,** but the
  fitness is the dense solo score, so the signal is not limited to
  outcomes.

## Code

There is no module change: `motor_gate.train` already cycles a list of
opponents. `probes/cycle-0043-self-play-opponent-gate.py` is the v0.0.27
probe with the three-opponent list and the v0.0.28 protocol. **No game has
been played on seeds 39000–42023.**
