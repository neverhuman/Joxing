# Grokbot red-team — cycle 0019 / v0.0.15 (2026-10-28): solo motor gate, root cause, degenerate-league guard

Reviewed `origin/main` at `70f0739`. jekko has made no commits since
`1978da7`. This cycle answers three owner notes:

- **01:50:** tag only if the preregistered criterion passes.
- **02:22, task 1:** before more league runs, root-cause why games come out
  invalid.
- **02:22, task 2:** add a CI/smoke test that fails on 0 valid games.

## v0.0.15 solo motor gate (preregistered, failed, not tagged)

`grokbot/redteam/protocols/v0.0.15.json` (sha256 `32771370…`) was pushed at
`548ac92` before the run. One athlete in corner A trained alone against the
frozen v0.0.11 gen4 posture in corner B, at the facing 0.5 m spawn, over the
full 3 s horizon. Training was antithetic ES with common random numbers on a
continuous fitness: stand fraction plus a strike indicator. It ran 24
iterations of 8 pairs on `PunchPolicy`, starting from gen4.

The primary gate needed both, over 24 noisy trials: stance pass (3 s without
non-foot floor support) ≥ 0.5, and reach pass (a landed glove strike) ≥ 0.5.

| | Stance pass | Reach pass | Median stand | Stand range |
|---|---:|---:|---:|---|
| Baseline gen4 posture | 0/24 | 0/24 | 1.464 s | 1.440–1.627 s |
| Trained candidate | **0/24** | **0/24** | 1.349 s | 1.325–1.643 s |

- Paired stand-time gain: **−0.09 s**, with a lower 95 % bound of −1.59 s.
- Training curve: mean stand time across the 24 iterations stayed between
  1.36 and 1.50 s with no trend. The strike share was 0.000 in every
  iteration.
- 1,102 s on 4 nice-10 workers; 0 engine faults.

**The primary gate failed. v0.0.15 is not a level-up and is not tagged.**
Pre-declared reading 1 applies. Linear posture control on 10 muscle regions
cannot hold the native keyframe; its ceiling is about 1.5 s, the same plateau
reached in the v0.0.11 league (1.52 s) and here. A continuous fitness
signal made no difference.

## Root cause: why every league game since v0.0.12 is invalid (measured)

Under the anti-stalling rules, a valid result needs a landed glove strike, a
glove meeting a strike target at ≥ 0.5 m/s approach and active speed, with
no body-on-body contact in the lookback. The causal chain, with the
measurement behind each step:

| # | Cause | Measurement |
|---|---|---|
| 1 | **Everyone falls in about 1.3–1.65 s** | v0.0.11 league plateau 1.52 s. Fresh-seed medians for gen2–gen6: 1.37–1.52 s. This motor gate: baseline 1.46 s, trained 1.35 s, ES flat over 24 iterations. |
| 2 | **They fall backward**, away from the opponent | The head moves about 1.3 m *away* within 1.2 s at every separation from 0.6 to 1.6 m (cycle 0016 probe). |
| 3 | **Reach is about 0.2 m** | The best DynSyn arm groups (58/81) move the hand about 0.20 m forward relative to the trunk in 0.2 s. The closest hand-to-target distance at 0.6 m was 0.596 m, so the gloves stay about 0.35–0.45 m short of the target shells (cycle 0017). |
| 4 | **Closer spawns collide instead** | At ≤ 0.5 m, bouts are blocked by `native_non_scoring` body-on-body contact. At ≥ 0.6 m there is no contact at all. There were 0 strikes in 32 scoping bouts from 0.40 to 0.70 m, with single and mutual punches (cycle 0017). |
| 5 | **Training terms cannot help when the base rate is 0** | In v0.0.14 the training valid share was 0.000 across 192 games, so the valid bonus and no-contest penalty produced no gradient. v0.0.12's approach shaping was gamed by body lean, with 1,156 body-on-body contact points and 0 glove contacts. |

**Result:** 0 valid games in v0.0.12 (0/588), v0.0.13 (0/1,344) and v0.0.14
(0/840). 2,594 of 2,772 games were no-contests and 178 were blocked. Every
one traces to steps 1–4.

**What breaks the chain** is a controller that does not fall backward within
1.5 s. Neither league design, shaping, spawn distance nor linear ES does it.
The candidates are:

- jekko's pretrained shared-motor bridge, which kept both athletes upright
  for 3.52 s in the first shared trial; it would need a stationary facing
  stance;
- a richer controller trained for stance alone, with group-level or
  recurrent feedback.

## CI guard: a league that yields 0 valid games now fails

`selfplay.run_league` raises `DegenerateLeagueError` by default
(`require_valid=True`) as soon as any evaluation batch contains no valid
game. The message gives the reason census, for example
`{'no contest': 2}`, and points here. The probes that reproduce the recorded
degenerate runs (cycles 0016–0018) opt out explicitly with
`require_valid=False`.

New tests in `grokbot/tests/test_degenerate_guard.py` use real physics at a
tiny size:

- Anti-stalling at the native spawn, where every bout is a no-contest:
  **raises** `DegenerateLeagueError`.
- The same league with scoreable endings: passes, with
  `eval_valid_share > 0`.

The self-play CI smoke test (`test_selfplay.py`) now also asserts
`eval_valid_share > 0`. A league configuration that degenerates therefore
fails CI rather than producing an unratable report. The league plumbing test
for the degenerate punch configuration opts out explicitly.

The motor gate's own tests (`test_motor_gate.py`, 4) cover:

- a real short trial;
- the continuous fitness;
- ES dropping quarantined pairs, with an exact update;
- pass rates that count faults as failures.

## Next steps

1. **No more league runs on this motor.** The guard now enforces this.
2. **Motor capability.** Integrate a stance controller: jekko's pretrained
   bridge with a facing stationary reference, or a group-level controller.
   Then rerun the **same preregistered v0.0.15 gate** with it.
3. Only after the gate passes, preregister a league with the stage-1
   valid-game floor and an Elo criterion against the v0.0.12 baseline.
