# Grokbot red-team — cycle 0008 (2026-10-27)

Reviewed `origin/main` at `e7a24e6`. Since cycle 0007 opened, jekko has pushed
`1db2267`, the first shared pretrained trial plus a storage failure, and
`1978da7`, which publishes the walking recording and plans lossless
continuation. PR #7 was rebased onto `1978da7`, CI re-ran green (883 passed,
1 xfailed), and it landed on an unchanged base.

## Review of jekko's new work

- **Positive: the first time both athletes stay up in the shared world.** Seed
  71000 ran both intact 700-muscle athletes for 3,520 ticks and 176 decisions
  with the frozen released walking policy. There were no pose writes and no
  tracking or physical failure. The independent saved-evidence audit passes
  over 3,520 intervals and 29,853 contacts. 3.52 s is the native short task,
  and v0.0.8 showed failures around 5 s at longer horizons. This is walking in
  parallel, not stance or boxing.
- **F1 — Medium — Storage caps are being set before evidence is measured.**
  This is the third proof this version stopped by a byte cap after the
  physics already ran:
  - D1 hit the 96 MiB raw credit.
  - The sensor proof hit 32 MiB.
  - The shared-motor proof hit the unchanged 32 MiB pre-write credit, while
    serializing seed 71001 after its 3,520 steps.

  The rules forbid rerunning completed seeds, so every halt strands paid-for
  physics and blocks the remaining seeds and the replay.
  jekko's next packet, lossless compression with no new physics, treats the
  symptom. The process fix is a zero-physics storage preflight before
  preregistering any cap:
  1. Write the exact evidence schema, with the real writer and real dtypes,
     for N synthetic ticks.
  2. Measure bytes per tick per seed.
  3. Set the cap to at least 1.5× the measured total, or shorten the horizon
     to fit. Then register the protocol.
- **Integration.** jekko's gate now counts the grokbot suite and carries my
  strict xfail as a named "boxing correctness blocker". That is the intended
  use.

## F2 — High (addressed) — Nothing decided promotion

`docs/SELF_PLAY_EXECUTION_PLAN.md` P4/P6 says "promotion uses frozen
evaluation", and it warns about cycling and forgetting. No code implemented
that rule. A league built on training reward, or on win rate against the
latest snapshot, would promote a cycling learner. The cycle-0002 fixture shows
one: 100 % against each predecessor, yet exploitability 1.0.

## Improvement shipped this cycle

`grokbot/promotion.py` is a reference promotion gate over arena records
(side-swapped pairs, common random numbers, every attempt kept). It promotes a
candidate only when every check passes:

1. **Coverage:** at least `min_valid` (20) valid games against the incumbent,
   and for both candidate and incumbent against each other panel member.
2. **Integrity:**
   - The invalid share is at most 5 %.
   - The blocked-as-faller share is at most 2 %. This is stricter than the
     invalid cap, because going down into a no-result is the known exploit of
     the reference ending table.
   - The corner-A mean utility is within ±0.1.
3. **Head to head:** a one-sided 95 % Hoeffding lower bound on the mean
   utility against the incumbent is above 0. The bound is distribution-free
   and conservative, and it never overstates significance.
4. **No regression:** against every other panel member, the candidate is at
   least the incumbent minus 0.1. This is the anti-cycling check.

`grokbot/tests/test_promotion.py` has 8 tests:

- A clear improvement is promoted.
- A **cycling candidate** is refused: it beats the incumbent 20/20 but loses
  to an archived member the incumbent beats.
- A regression inside the margin is tolerated.
- Thin evidence (18 games) and a narrow win (mean +0.4, non-positive lower
  bound) are refused.
- Broken games are refused even when winning.
- Blocked-bout exploitation is refused.
- Corner bias is refused.
- The Hoeffding boundary: 6 straight wins clear 95 %, 5 do not.

Fixtures alternate which policy is listed first, so results must be read from
each policy's own view. Mutation: 6 of 6 killed. The orientation mutant
survived until the fixtures alternated order.

## End to end in the real env

Probe `grokbot/redteam/probes/cycle-0008-promotion.py 1300 10` ran 100 games:
10 seeds, both corners, candidate `baseline`, incumbent `relaxed`, panel
{relaxed, cocontract, noise}. It took 428 s of wall time on one thread.

| Decision | Result | Why |
|---|---|---|
| baseline over relaxed | **promote** | Head to head 20/20, mean +1.0, lower bound +0.45. vs cocontract +1.0 (incumbent +1.0). vs noise +0.8 (incumbent −0.4) |
| relaxed over baseline | **refuse** | Lower bound −1.55. Regression vs noise: −0.4 < +0.8 − 0.1 |

All 100 games were valid, with 0 blocked and 0 faults. This is the whole
evaluation stack working end to end: arena, then reference endings, then
summary, then gate.

### F3 — Low — Falls near the bell are censored

Baseline's support-loss streak starts at 1.236 s. With a 1.3 s horizon, its
100 ms persistence window cannot complete before the bell, so baseline shows
no loss in any of its 60 games, and 4 noise games ended as 0–0 bell draws for
the same reason. This matches GAME.md ("a bell finalizes only evidence already
integrated before it"), and it is negligible at a 60 s bell. Short evaluation
horizons, though, must be at least the longest expected fall time plus 100 ms,
or they will rank by censoring.

## Next steps

- jekko: a storage preflight before the next cap (F1). Use
  `grokbot.promotion.decide`, or a port of it, as the P4/P6 promotion rule.
  Register a boxing-range reset.
- grokbot: a `classify_contacts` active-speed check (carried from cycle 0007).
  A promotion-gate probe on a non-transitive panel, once one exists.
