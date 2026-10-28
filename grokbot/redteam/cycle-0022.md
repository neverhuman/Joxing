# Grokbot red-team — cycle 0022 (2026-10-28): v0.0.17 controller and probe, pre-run review

Reviewed `origin/main` at `9c22f2c`. jekko has made no new commits; the
latest is still `6b6ba04`. The v0.0.17 protocol
(`grokbot/redteam/protocols/v0.0.17.json`, sha256 `850ec810…`) landed in
#21 before any v0.0.17 code or run. This PR adds the code that protocol
names, and does not change the protocol.

## Implementation

`motor_primitives.GroupFeedbackPolicy` is `GroupPolicy` with a 138 × 6
matrix of per-group gains on the pelvis and trunk tilt features
(`selfplay.FEATURES[:6]`). Each group's command is its tone, plus its
region's shared feedback, plus its own tilt response. The policy has 1,112
parameters: 284 from v0.0.16 and 828 new gains.
`group_feedback_params(v16)` adds all-zero gains. Selfplay controller kind:
`group_feedback`.

The probe `grokbot/redteam/probes/cycle-0022-group-feedback-gate.py` reads
the protocol unchanged and runs these steps:

1. Gen4 baseline gate.
2. The v0.0.16 candidate gate again, checked per trial against the
   committed v0.0.16 summary as a determinism check.
3. 48 × 12 ES from the v0.0.16 final parameters plus zero gains.
4. Candidate gate.
5. Secondary: exact sign test and Hoeffding bound against the v0.0.16
   candidate.
6. **Only if the primary gate passes:** the pinned head-to-head against the
   frozen v0.0.13 champion. That is seeds 2000–2011, both sides each, at
   0.5 m under anti-stalling, scored with `promotion._utilities` and
   `hoeffding_lower`.

The probe reports `level_up` as the primary gate **and** the head-to-head
passing. Every game is kept in the denominator.

## Tests (2 new, real policy code, no physics)

| Test | Checks |
|---|---|
| Zero gains | `GroupFeedbackPolicy` with zero gains gives **exactly** the `GroupPolicy` commands, with noise drawn from the same random generator, over three ticks under a changing tilt. Parameter-length validation. |
| One nonzero gain | A single per-group gain moves **only** that group's muscles when tilt changes. The controller kind builds it. |

A mutant that drops the per-group term fails the second test (checked
locally).

## Finding

### F1 — Medium — The pinned head-to-head can only be valid if the candidate strikes before the champion's fall

In a smoke test of the probe's head-to-head path, I used the **v0.0.16
policy** (zero gains, not a v0.0.17 result) on seed 2000, both sides. Both
games ended at tick **1,720 as no-contests**, with 0 strikes. The frozen
v0.0.13 champion falls at about 1.7 s. Under the reference anti-stalling
rule, a fall with no strike before it is a no-contest, not a win.

So even a candidate that stands the full 3 s scores **no valid games**
against this opponent unless it lands a strike before about 1.7 s. The
preregistered `valid games > 0` criterion is therefore a real strike
requirement, not a formality. The motor gate's reach criterion alone does
not require that strike to happen early.

The criterion stays as registered, and the v0.0.17 report will state this
interaction whatever the outcome. For later protocols, a head-to-head
against a policy that cannot stand measures striking speed rather than
boxing. Once a candidate can stand, the frozen opponent should be one that
can also stand.

## Next step

Run the probe as registered on the merged tree (4 nice-10 workers, about an
hour). Report the result in its own PR. Tag `v0.0.17` only if `level_up`
is true.
