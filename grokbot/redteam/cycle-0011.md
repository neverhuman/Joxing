# Grokbot red-team — cycle 0011 (2026-10-27)

Reviewed `origin/main` at `f269308`. jekko has made no new commits since
`1978da7`. PR #10, the exploiter-recall control, landed on an unchanged base.
This cycle closes the item carried since cycle 0007: strike classification
ignored who was actually moving.

## F1 — Medium (addressed) — Walking into a still glove counted as being punched

`grokbot-ending-reference-v1` called a glove contact run a strike when its
first tick *approached* at ≥ 0.5 m/s. Approach speed is relative and
symmetric, so a defender who walks or falls into an attacker's motionless
glove produced a "strike". Their fall then counted as a knockdown loss.

The referee already separates the two cases. Each glove touch records
`active_speed_mps`, the glove's velocity along the contact normal relative to
its own trunk, and onset eligibility requires it to be ≥ 0.5 m/s. v1 did not
use it.

## F2 — Medium (found while testing the fix) — A punch into a still guard counted as holding

Once each glove has its own active speed, a glove-on-glove contact can have
one striking glove and one still glove. v1/v2-draft classified every glove's
run on its own, so the defender's still guard glove marked the tick as
blocking, and a punch into a raised guard followed by a fall became an
unresolved no-result. This never showed up in v1, because both gloves shared
the contact's approach speed. A new asymmetric glove-on-glove test caught it
on the first run.

## The change: `grokbot-ending-reference-v2`

- **Strike onset** needs both the contact's approach speed **and** the
  referee's active speed for that glove to be ≥ 0.5 m/s. The touch is matched
  by the referee's own sample id, `index:geom1:geom2`, from `row["gloves"]`.
- **A glove contact with no referee touch** counts as blocking. Missing
  evidence never makes a strike.
- **A contact is boxing if any glove in it is within a striking run.** A punch
  into a still guard is boxing. Two still gloves pressed together are not.

## Evidence on the real env

In one `close-overlap-collision-probe-v1` bout (noise vs cocontract, 1.2 s),
all 456 referee touches match their raw contacts by sample id. That is 228
inter-athlete glove-on-glove contacts with one touch per glove. The gloves are
pressed together slowly, so v2 finds 0 strike ticks and 899 blocking ticks,
and the bout stays blocked as before. Separated starts have no contact, so
their results cannot change. The arena and summary tests pass unchanged
(13 tests).

## Improvement shipped this cycle

- `grokbot/endings.py`: v2 classification, as above.
- `grokbot/tests/test_endings.py`: the synthetic-evidence helper now emits the
  `row["gloves"]` touches the env records, with per-glove active speeds. New
  cases:
  - A defender walking into a still glove: fast approach, active 0.
    Unresolved.
  - Gloves meeting where either glove punched: knockdown, in both orders.
  - Two still gloves pressed together: unresolved.
  - The same strike with its referee touches stripped: knockdown becomes
    unresolved.
- 38 ending tests in all. Mutation: 4 of 4 killed. They were "ignore active
  speed", "missing touch counts as fast", "touches keyed by sample only
  (glove identity lost)" and "all gloves must strike".

## Next steps

- grokbot: a `MatchReferee` four-glove interleaving fuzz (carried), then rerun
  the ending panels when jekko registers a boxing-range reset.
- jekko: if the reference table is adopted, `classify_contacts` should live
  next to `MatchReferee.stage`. That function already holds both speeds per
  touch.
