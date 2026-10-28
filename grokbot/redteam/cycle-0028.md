# Grokbot red-team — cycle 0028 / v0.0.20 (2026-10-28): self-play league is an honest null; ladder and degeneracy

Reviewed `origin/main` at `ef80a8c`. jekko has made no new commits. The
v0.0.20 protocol (`18cf2ef5…`) landed in #27 before the run, which used
merged main's tree `da34b9b7` at nice 19 and took 3,957 s.

The owner-10:05/10:30 reporting addendum
(`protocols/v0.0.20-ladder-addendum.json`, `4a061b2f…`) and its probe were
pushed to this branch, as `ce75c92`, **before any league output was read**.

## v0.0.20 — preregistered level-up **NOT met** (honest null), not tagged

| | Result |
|---|---|
| League | 4 generations × 4 iterations × 24 pairs (768 training games), seed 10, trained on outcome only, seeded from v0.0.19 |
| Promotions | **None.** Every generation failed coverage, with 0–1 valid games against the incumbent. **Champion = gen0 = v0.0.19.** |
| **Primary**: champion vs frozen v0.0.19, seeds 7000–7023 | **3/48 valid** (40 no-contests, 5 blocked), 0–0–3. (a) failed, (b) failed (lower95 −1.73). **Failed.** As registered, this compares v0.0.19 with itself: an honest null. |
| Last generation (gen4) vs v0.0.19, same seeds | 0/48 valid, with **46 of 48 blocked** |
| Champion vs frozen v0.0.16 / v0.0.13 | 48–0 (34/48 wins with hits) and 48–0 (48/48 with hits). These are unchanged, because the champion is v0.0.19. |
| Champion motor gate | Stance 16/24, reach and hits 24/24, median 3.0 s, as for v0.0.19 |

**`level_up = false`. No tag** (owners 10:50 and 11:25: tag only on the
preregistered gate).

## Degeneracy check (owner 10:30)

**Within the league:**

| Generation | Training valid | Training blocked | Training median end tick | Evaluation valid | Evaluation no-contest | Engaged | Promoted |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.27 | 89/192 | 2,420 | 0.50 | 0.24 | 0.54 | no |
| 2 | 0.31 | 100/192 | 2,390 | 0.40 | 0.18 | 0.50 | no |
| 3 | 0.23 | 94/192 | 2,700 | 0.28 | 0.21 | 0.43 | no |
| 4 | 0.29 | 91/192 | 2,550 | 0.27 | 0.23 | 0.41 | no |

Training games were **47–52 % blocked** and only **23–31 % valid**. The
evaluation valid share fell from 0.50 to 0.27, and the engaged share from
0.54 to 0.41. **The league drifted towards body contact.**

**The v0.0.19 mirror (primary games):** all 40 no-contests ran to the
3,000-tick horizon with **0 strikes by either side**. The same policy that
lands a head hit at about 0.52 s against the gen4 posture lands **nothing**
against a standing copy of itself.

## Round-robin ladder, v0.0.11 to v0.0.20 (owner 10:05/10:30)

Identical policies were played once: v0.0.12 = v0.0.11, v0.0.14 = v0.0.13
and v0.0.20 = v0.0.19. That leaves 7 distinct players, 21 pairings and 252
games, on seeds 9000–9005 with both sides.

| Player | Elo (anchor v0.0.11 = 0) | 95 % bootstrap |
|---|---:|---|
| v0.0.11 (= v0.0.12) | 0 | — |
| v0.0.13 (= v0.0.14) | +25 | −97 to +170 |
| v0.0.15 | +25 | −82 to +131 |
| v0.0.16 | +586 | +427 to +754 |
| v0.0.17 | +520 | +409 to +585 |
| v0.0.18 | +810 | +679 to +990 |
| **v0.0.19 (= v0.0.20)** | **+1,012** | **+885 to +1,164** |

- **Intransitive triads: 0. There is no non-transitive cycling.** The
  ordering is monotone from v0.0.16 onwards, except that v0.0.17 sits just
  below v0.0.16 (v0.0.16 scored +0.25 head-to-head against v0.0.17), with
  overlapping intervals.
- **Overall degeneracy:** 57 % valid, 30 % no-contest, 13 % blocked, median
  end tick 1,620, 0 engine faults.
- **Seven pairings had 0 valid games.** All pairings among v0.0.11, v0.0.13,
  v0.0.15 and v0.0.16 were no-contests, because none of those policies
  strikes. **v0.0.18 vs v0.0.19 was 92 % blocked**, and v0.0.17 vs v0.0.19
  83 % blocked. The ladder ranks v0.0.19 above v0.0.18 only *through
  common opponents*, never directly.

## Findings

### F1 — High — The best policies cannot produce a decisive game against each other

There are two distinct failure modes, and they compound:

1. **Mirror matches never strike.** v0.0.19 against itself: 0 strikes in 40
   games, which end at the bell as no-contests.
2. **Different strong punchers collide.** v0.0.18 vs v0.0.19 is 92 %
   blocked. At 0.5 m their bodies meet, and the loss cannot be attributed.

Self-play needs exactly these games, so its outcome signal is close to zero.
That is why no generation could be promoted. **More self-play games, the
v0.0.20 lever, cannot fix this:** with 768 games, valid share stayed at
23–31 % and fell over training.

### F2 — Medium — The registered fallback reading points at the wrong lever

The protocol pre-declared: "no promotion / (b) fails → the outcome signal is
too weak; next lever: more games per generation". The data show the signal
is not weak but **structurally absent** in the decisive matchups. That
reading is reported as registered, but the recommendation below follows the
measured cause.

## Next single lever (v0.0.21, to be preregistered in its own PR)

**Change the training opponent.** Train the same solo hit-and-stay-up
fitness against **frozen v0.0.19** instead of the gen4 posture. Keep the
same model, initialization (v0.0.19), ES settings, reset and ending.

Why:

- The gen4 posture falls at about 1.46 s and never guards, so it never
  shows the policy a standing, guarding target.
- Both F1 failure modes appear **only** against such an opponent.
- This is not competitive learning (AGENTS.md): the fitness is still the
  solo motor fitness.
- The registered primary would be the a/b/c gate against frozen v0.0.19 on
  fresh seeds, so a mirror that still never strikes fails criterion (a).

## What this PR adds

- `data/cycle-0027-selfplay-league-summary.json`:
  - league generations, with the archive kept as parameter norms only (the
    champion is v0.0.19);
  - every head-to-head per game;
  - the champion's motor gate;
  - the full ladder report;
  - SHA-256 hashes of the full reports.
- The ladder addendum and its probe (already pushed before any result was
  read), this report, and the STATUS update.
