# Grokbot red-team — cycle 0042 / v0.0.27 (2026-10-29): a smaller step makes the collapse worse; train against the start itself

Reviewed `origin/main` at `65c954a`, the v0.0.27 preregistration (#41,
protocol `4bc77efc…`).

## Attempts (both in the denominator)

| Attempt | Outcome |
|---|---|
| 01 (03:02:44Z) | **Lost at training iteration 18/48 by my error.** During a host load spike (1-minute load 106.8 from other jobs) I paused my own run with SIGSTOP. The non-interactive launch shell treated the stopped child as a finished job (exit 147) and exited, and the orphaned, stopped process group then received SIGHUP. The first governor also stopped itself, because its command line matched its own `pgrep` pattern. Only my processes were affected. The smoke gate had passed. The directory is kept as `runs/grokbot/small-step-gate-0041-attempt01-killed`. **Lesson: never SIGSTOP-pause; yield only by gating the launch and running at nice 19.** |
| 02 (03:51:48Z, load 66.6) | One gated command, nice 19, no governor. Completed. |

## v0.0.27 — registered level-up **NOT met** (monotonicity); not tagged

The one lever was a smaller ES step, lr 0.05 → 0.02, on the v0.0.26 recipe.
The smoke gate **passed**, identically in both attempts.

| Matchup (48 games) | Valid | W–L–D | Seat A | Seat B | Result |
|---|---:|---|---|---|---|
| **Primary: vs frozen v0.0.19** | 48 | **48–0–0** | 24–0 | 24–0 | **passed** |
| **Hard gate: vs frozen v0.0.13** | 48 | **48–0–0** | 24–0 | 24–0 | **passed** (again) |
| Monotonicity: @48 vs @16 | 48 | **11–37–0** | 6–18 | 5–19 | **failed**: @48 lost to @16 |
| Monotonicity: @16 vs the v0.0.22 base | 48 | **0–48–0** | 0–24 | 0–24 | **failed**: @16 lost every game to its start |
| v0.0.16 (secondary) | 48 | 48–0–0 | | | — |

**Motor gate against gen4:** stance 8/24, median stand 2.68 s.
**Training:** contact went from 350 to 333, and the valid-knockdown share
from 0.67 to 0.79.

## Findings

1. **High: the collapse is not a step-size effect.** With a 2.5× smaller
   step, @16 lost **0–48** to its own v0.0.22 start (v0.0.26: 2–40; v0.0.25:
   4–44), and @48 lost to @16 (11–37). Across all three versions, training
   under this objective makes the policy **worse against its own start**.
   Meanwhile every version beats both frozen opponents 46–48 to 0, and so
   does the start itself.
2. **Diagnosis.** The objective measures stand time, knockdowns and contact
   *against the frozen v0.0.19 and v0.0.13*. Nothing in it rewards beating a
   policy of the agent's own strength. ES finds behaviour that keeps
   exploiting the weaker frozen opponents while losing the head-to-head
   against the base. This is exactly the gap the monotonicity gate exists to
   expose.
3. **The registered reading applies:** "@16 still loses to the base → not a
   step-size effect → next lever: the self-play opponent".

## Next: v0.0.28, its own preregistration PR, opened in parallel off main

**One lever:** add the **frozen v0.0.22 base** (the policy's own start) to
the training opponents, so ES pairs cycle through v0.0.19, v0.0.13 and the
v0.0.22 base. Beating its own start, the gate it keeps failing, becomes part
of the training signal. The learning rate goes back to v0.0.26's 0.05, so
the self-play opponent is the only change from v0.0.26.

## What this PR adds

- `data/cycle-0041-small-step-gate-summary.json`, including the attempts
  ledger.
- This report, the CHANGELOG entry (with the unpublished-versions table
  extended) and STATUS.
