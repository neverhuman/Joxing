# Grokbot red-team — cycle 0026 / v0.0.19 (2026-10-28): first preregistered level-up, and why it took this long

Reviewed `origin/main` at `4ffbd91`. jekko has made no new commits. The
v0.0.19 protocol (`e9b2963c…`) landed in #25 **before** the run. The run
started from merged main, tree `e970249f`, at nice 19 with load 42, and took
3,238 s. Its probe checkpointed after every stage; the final report is
`complete: true`.

## v0.0.19 result — **preregistered primary PASSED, level-up**

**One change from v0.0.18: the training fitness.** A head/torso hit earns
the bonus only if the athlete is still up 500 ticks after its first hit.
Everything else was unchanged: `GroupPDPolicy` from the v0.0.18 parameters,
48 × 12 ES, seed 9.

| | **Primary: vs frozen v0.0.13, seeds 5000–5023** |
|---|---|
| Valid | **48/48** |
| W–L | **48–0** |
| Wins with own head/torso hits | **48/48** |
| Mean, lower95 | +1.00, **+0.50** |
| Elo (lower95) | **+669 (+191)** |
| a / b / c | pass / pass / pass. **`level_up = true`** |

### Secondaries (reported, not gating)

| | Stance | Reach | Hits | Median stand |
|---|---:|---:|---:|---:|
| v0.0.18 (re-run, **identical per trial**) | 1/24 | 24/24 | 24/24 | 2.31 s |
| **v0.0.19** | **16/24** | 24/24 | 24/24 | **3.00 s** |

The **motor gate passes for the first time** (stance 0.67 ≥ 0.5 and reach
1.0 ≥ 0.5). The sign test on stand ticks against v0.0.18 gives 22 of 24
better, p = 1.8×10⁻⁵. In training, mean stand time rose from 2.18 to
2.54 s.

| Head-to-head, 48 games each | Valid | W–L | Hits in wins | Elo (lower95) |
|---|---:|---|---:|---|
| v0.0.18 vs v0.0.13, same primary games | 48 | 48–0 | 48 | +669 (+191) |
| **v0.0.19 vs frozen v0.0.16**, seeds 6000–6023 | 48 | **47–1** | 21 | **+669 (+172)** |
| v0.0.18 vs frozen v0.0.16, same seeds | 44 | 15–29 | 13 | −115 (−424) |

**Improvement against v0.0.16: +1.28** in seed-paired mean utility, from
v0.0.18's −0.32 to v0.0.19's +0.96.

### Red-team reading (pre-declared reading 1 applies)

1. **The primary alone does not separate v0.0.19 from v0.0.18.** v0.0.18
   also wins 48–0 on the identical games, as the protocol's red-team note
   predicted. The level-up is real because the version improved
   **against the last tag**: v0.0.17 has 0 hits, so it fails (c). It is
   *meaningful* because of the secondaries: the first motor-gate pass, and a
   swing from losing to v0.0.16 (15–29) to beating it (47–1).
2. **Most wins against v0.0.16 are balance wins.** Only 21 of 47 include
   v0.0.19's hits, and the games end at a median tick of 1,860, when v0.0.16
   falls first. v0.0.19 stands the full 3 s in 16 of 24 solo trials. It now
   *outlasts* the opponent; it does not yet *knock it down*.
3. **The mechanism is not isolated.** The hit-and-stay-up share was already
   0.96 at iteration 0, so the new bonus rarely changed. What it did change
   is that trials which hit and then fell lost the bonus, so the stand term
   decided more of the rankings. That is consistent with the stance gain,
   but it was not separately tested.
4. **Scope.** This is a solo-trained policy evaluated against frozen
   opponents. It is a grokbot diagnostic, **not** an unassisted boxing result
   (AGENTS.md). jekko's completion battery is not claimed, and native boxing
   games remain 0.

**Decision: tag `v0.0.19`**, the first tag in this lane that marks a passed
preregistered level-up, on this PR's merge, with a CHANGELOG entry and the
summary JSON attached to the release.

## Diagnosis (owner 08:45): why v0.0.13–v0.0.18 did not level up

| Phase | Evidence | Cause |
|---|---|---|
| **League, v0.0.11–v0.0.14: no outcome signal** | 0 valid evaluation games out of 2,772 in v0.0.12–v0.0.14. Training valid share was **0.000 in every v0.0.14 generation**. `centred_ranks` gives tied fitness a **zero** update by design, so the match outcome contributed **nothing** to ES. Training fitness still rose from 0.03 to 0.16 in v0.0.12 and v0.0.13, but only through the shaping terms, which leaning gamed (cycle 0016). v0.0.11's +1,081 Elo came from 0-contact balance. | **Degenerate game, not diversity collapse.** Every bout was a no-contest, so the payoff matrix was all zero and the gradient came only from proxies. The degenerate-league guard (#19) now fails such a league in CI. |
| **Evaluation noise: ruled out** | v0.0.15 reran bit-for-bit. The v0.0.16, v0.0.17 and v0.0.18 gates each re-ran **identically per trial** in the next version's run. The misses were large: stance 0/24, then 7/24 against 12 needed; Elo −151. | — |
| **Motor capability, v0.0.15–v0.0.17** | Median stand rose 1.35 → 2.34 → 2.68 s as the controller grew: 10 regional tones, then 138 group tones, then per-group tilt feedback. | **Capacity.** Linear regional control could not hold the keyframe beyond about 1.5 s. |
| **Mis-specified proxies, v0.0.17–v0.0.18** | v0.0.17's strike bonus was satisfied by **glove taps**: 0 head or torso hits. v0.0.18's hit-only bonus rewarded **hit-then-fall**: stance 1/24, and losses to v0.0.16 by falling first. | **Proxy and evaluation mismatch.** Each solo proxy was gamed in the direction the head-to-head punished. v0.0.19's fix aligned the proxy with "hit and stay up". |
| **No self-play since v0.0.14, by design** | Since v0.0.15 all training has been solo ES against a frozen gen4 posture, because AGENTS.md puts the motor gate before competitive learning. | The league could not help until games could be valid. **They now are**: 100 % valid against v0.0.13 and v0.0.16. |

## Next: v0.0.20 — one lever, **more self-play games** (preregistration in the next PR)

The owner (09:10) offered "bigger model or more self-play games". **I pick
more self-play games** and do not change the model:

- **The model is no longer the bottleneck.** `GroupPDPolicy` (2,078
  parameters) now passes the motor gate, standing 16/24 with hits 24/24.
- **The remaining failure is opponent-dependent.** Wins are by balance
  rather than knockdowns (21/47 with hits), and none of the training so far
  has involved an opponent's response. Only games against opponents give
  that signal; a bigger model gives none.
- **The league finally has a learning signal.** v0.0.19 plays 100 % valid
  games, so the outcome is no longer all zeros, and the degenerate guard
  will verify that in CI.

v0.0.20 will therefore be a **two-player self-play league seeded from
v0.0.19**: frozen-opponent sampling, the existing P4 sampler, and the
degenerate guard on. Its preregistered primary will be the a/b/c gate against
the **frozen v0.0.19** on new seeds. Criterion (c) requires at least half
of the wins to include hits, so a policy that only outlasts its opponent
cannot pass. The model and fitness stay fixed.
