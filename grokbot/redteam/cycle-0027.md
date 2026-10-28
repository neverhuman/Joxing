# Grokbot red-team — cycle 0027 (2026-10-28): v0.0.20 preregistration (one lever: self-play games)

Reviewed `origin/main` at `cd2e545`. jekko has made no new commits. v0.0.19
is tagged, and its release is the first passed preregistered level-up. Per
owners 09:10, 09:30 and 09:50, this PR lands the v0.0.20 preregistration
**on its own, before any run**.

## The one lever: more self-play games (`protocols/v0.0.20.json`)

| | v0.0.19 | **v0.0.20** |
|---|---|---|
| Training games | Solo motor trials against a frozen posture | **Two-player self-play league games** |
| Training signal | Stand time plus a hit-and-stay-up bonus | **Match utility only**: zero-sum, all shaping, bonus and penalty terms at 0 |
| Unchanged | | Model (`GroupPDPolicy`), initialization (v0.0.19's final parameters), σ 0.1, lr 0.05, the 0.5 m reset, ending v2 with anti-stalling |

**Why this lever and not a bigger model** (cycle-0026 diagnosis):

- The model already passes the motor gate.
- The remaining weakness, winning by balance rather than knockdowns (21 of
  47 wins against v0.0.16 had hits), only shows up against an opponent.
- v0.0.19 plays 100 % valid games, so for the first time a league has a
  non-zero outcome signal.

**League:** `selfplay.run_league`.

- Size: 4 generations × 4 iterations × 24 pairs, seed 10.
- Opponents: the P4 mixture of latest, PFSP history and anchors. The anchors
  are the scripted panel plus frozen v0.0.19, v0.0.16 and v0.0.13.
- Evaluation: seeds 100–111.
- Promotion: `decide` against the champion, with an engagement share of at
  least 0.25.
- `require_valid=True`, so an all-invalid evaluation batch aborts.

**Primary, and the only level-up:** the a/b/c gate of the **league champion
against the frozen v0.0.19** on **new seeds 7000–7023**, both sides. It uses
the identical `head_to_head` code from the v0.0.19 probe, imported
unchanged. **`v0.0.20` is tagged only if it passes.**

**Secondary, reported:**

- the champion against frozen v0.0.16 and against v0.0.13 (seeds 8000–8023);
- the last generation against v0.0.19, if the champion is not the last
  generation;
- the champion's motor gate;
- per-generation Elo, side balance, valid share and engaged share.

## Red-team notes, also registered

1. **If nothing is promoted, the champion is v0.0.19 itself.** The primary
   then compares v0.0.19 with itself and should fail (b). That is an honest
   null, not a pass.
2. **Anti-stalling lets a league drift towards pure balance**, because the
   first fall loses. Criterion (c) keeps a champion that only balances from
   passing.
3. **No leakage.** Seeds 7000–7023, 8000–8023 and 100–111 have never been
   played. The plumbing smoke test (a 1-generation league with `group_pd`
   and the frozen anchors) used seed 99996, and its report serializes to
   JSON.

## Probe

`grokbot/redteam/probes/cycle-0027-selfplay-league.py` reads the protocol
unchanged and saves `report.json` after every stage (owner 08:05).
