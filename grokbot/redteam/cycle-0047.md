# Grokbot red-team — cycle 0047 (2026-10-29): v0.0.30 preregistration (a self-play curriculum)

This branches from `origin/main` independently of the v0.0.29 report PR and
changes no CHANGELOG or STATUS text. It is **the preregistration on its
own**. The run launches only after merge, from the merged commit, once main
CI succeeds on a commit that contains it and load is below 70. It runs at
nice 19 and is never SIGSTOP-paused.

## Why this lever (cycle 0046)

@16 against its own v0.0.22 start follows the base's share of the early ES
update:

| Version | Base's measured update share, iterations 0–15 | @16 vs base |
|---|---|---|
| v0.0.26 / v0.0.27 | 0 | 2–40 / 0–48 |
| v0.0.28 | 0.23–0.28 (measured post hoc) | 2–46 |
| v0.0.29 | 0.44 (logged) | **16–32** |

v0.0.29's pre-declared reading for this outcome named this curriculum as the
next lever. Its accompanying "not a weighting effect" diagnosis is
contradicted by the table above, and that is recorded.

## The one lever (`protocols/v0.0.30.json`)

| | v0.0.29 | **v0.0.30** |
|---|---|---|
| Iterations 0–15 | 4-slot rotation (v0.0.19, base, v0.0.13, base) | **the frozen v0.0.22 base alone** (update share 1.0) |
| Iterations 16–47 | the same rotation | the same rotation |
| Unchanged | | Fitness, the v0.0.22 start, lr 0.05, σ 0.1, 48 × 12 (seed 20, one generator), the 0.5 m reset, ending v2 |

**Smoke gate.** All four checks must pass:

1. The penalty varies against each external opponent (seeds 47000–47047).
2. **Phase-1 lever check:** both lr-0 batches against the base alone give
   `rank_share_by_slot == [1.0]`. That means the whole update comes from the
   base and carries non-zero rank signal (rng 47300).
3. **Phase-2 lever check:** the base's update share is above 0.30, as in
   v0.0.29 (rng 47200).
4. A base head-to-head against v0.0.19 is at least 90 % valid (seeds
   47100–47123).

**Gates, in order, on seeds never played:**

| Gate | Seeds |
|---|---|
| Primary against v0.0.19 | 48000–48023 |
| Hard gate against v0.0.13 | 49000–49023 |
| Monotonicity: @48 vs @16, and @16 vs the base | 50000–50023 |

**Tag only if every gate passes.**

## Registered red-team notes

- **Phase 1 trains exactly the matchup the @16 gate tests,** on different
  seeds. A pass shows that ES can improve on its start head-to-head, not
  that it generalises.
- **@16 never sees an external opponent.** Only the final policy is gated
  against them; v0.0.25, trained without v0.0.13, had 0/48 valid games
  against it.
- **@48 vs @16 may fall,** because phase 2 can trade the self-play gains
  away.
- **@16 is the phase boundary by design.** The gate and its seeds are
  unchanged in kind.

## Code

There is no module change: `motor_gate.train` already takes any opponent
list, and its per-slot logging comes from #45.
`probes/cycle-0047-curriculum-gate.py` is the v0.0.29 probe with the base
alone for the first 16 iterations and the phase-1 lever check. It was
smoke-tested on seeds 99900 and above. **No game has been played on seeds
47000–50023.**
