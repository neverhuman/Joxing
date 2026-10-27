# Grokbot red-team — cycle 0005 (2026-10-27)

Reviewed `origin/main` at `64a34d3`. jekko made two new commits since cycle
0004.

## Review of jekko's new commits

- **`3f0dca3`** adds the shared muscle-driven reset/replay core:
  `src/joxing/shared_motor_task.py`, 374 lines, plus a `parallel-walk-v1`
  spawn variant in `scene.py` and 4 tests. No defect found.
  - Reset is the only place state is assigned.
  - Every evaluated substep is muscle-driven.
  - `_check()` rejects `qfrc_applied`/`xfrc_applied`, gravity compensation,
    non-standard gravity, any engine warning and non-finite state.
  - Both corners' commands are validated before either control changes.
  - `restore()` deep-copies, verifies the hash and the clock, re-derives
    self-target inputs, and commits atomically.
  - The variant puts both athletes at yaw 0, ±1.25 m in y. A therefore has its
    opponent on the left and B on the right, which is a translation, not a
    corner mirror. That's fine for its declared purpose ("separated locomotion
    transfer; not boxing"). It must never feed rating or self-play data, since
    the grokbot side-symmetry tests cover only the default facing spawn.
- **`bc7f6b5`** (docs) records that `grokbot/**` and `ci.yml` are
  externally maintained and read-only for jekko's packet. It also says the
  next source manifest must count both suites. The lanes are disjoint by
  agreement.
- **CI health.** Main is green at `64a34d3` with 486 passed in 550 s. That
  includes jekko's new tests.
- **Process note.** My #4 was rebase-merged onto those two commits after its PR
  CI had run on `ceb0b79`. Main CI on the combined head is green. The landing
  procedure now re-fetches main immediately before approve/merge.

## F1 — Low — `HitTracker.restore` aliases the snapshot it restores from

`HitTracker.restore(state)` keeps references to `state["ledger"]` and to the
`impulses`/`onset_eligible` dicts inside `state["current"]`. When the restored
tracker continues, it mutates the snapshot. Reproduction: snapshot mid-hit,
restore, continue 77 ticks. The snapshot's ledger grows from 0 to 1 events and
its `current.impulses.head` goes from 0.90 to 1.50 N·s. A second restore from
the same snapshot then starts from the corrupted state.

Every current caller is safe:

- `referee.py:202` and `strike.py:309` call `copy.deepcopy(...)` first.
- `tests/test_rules.py` round-trips through JSON.

Any future caller that forgets the copy, such as a league replay or a
checkpoint fork, will silently replay a different referee. **Fix (jekko, rules
owner):** `state = copy.deepcopy(state)` at the top of `HitTracker.restore`.
`grokbot/tests/test_referee_fuzz.py::test_restore_does_not_alias_the_snapshot`
pins this as `xfail(strict=True)`. The fix will flip it to XPASS, which fails
the run, and the marker can then be deleted.

## F2 — Low — "release" covers two different finalizations

A record's `reason` is `"release"` both when direct contact has been gone for
8 ms and when the glove fully recovered (30 ms clear + 2 cm + 3 cm) inside the
window. In the fuzz corpus below, 409 of 817 encounters finalized as
"release", and the label can't separate a punch that bounced off from a glove
that withdrew and re-armed early. This doesn't affect scoring, but exploit
probes (P2.4 "peck/rearm") will want the distinction. **Suggest** a
`"recovered"` reason in the next rules revision.

## F3 — Positive, now regression-locked — The referee does what GAME.md says

300 seeded random contact streams of 500 ticks each were checked against the
stated rules, computed independently from the raw stream rather than by
re-running the tracker. The streams mix the following:

- Multi-zone bursts.
- Zero-force and off-face samples.
- Near-miss gaps below 2 cm.
- Drifting extension.
- Optional bells.
- Mirrored head/torso "twin" touches, which produce exact ties.

They produced 817 encounters, finalized as 409 release, 378 window and 30
bell. 184 encounters scored. 61 had an exact head/torso tie. The checks are:

- One record per encounter, with contiguous sequence numbers and
  ledger = emitted.
- Each onset is the first force contact after the previous encounter's
  recovery: a 30-tick clear streak at ≥ 2 cm with ≥ 3 cm retraction from peak.
- The finalization tick and reason are the earliest of bell, the 50 ms
  deadline, an 8 ms direct gap, or recovery.
- Window, guard and per-zone direct impulse equal the in-window integral,
  with eligibility latched at each zone's first force contact.
- The award goes to the dominant zone, with ties resolved toward fewer
  points, only at ≥ 0.25 N·s.
- A random cut, JSON snapshot and resume reproduces the uncut emissions, and
  re-delivering a tick returns nothing.

**Mutation:** 7 of 7 `rules.py` mutants are killed. They are clear streak
−3 ticks, no retraction requirement, inverted tie-break, un-latched
eligibility, gap +2 ticks, clearance ≥ 0, and native contacts counted as guard.
`src/` was restored and verified clean after each one. The tie-break mutant
survived continuous random forces and needed the twin-touch streams.

This means the 0001-F4 feather-tap ceiling is **intended behaviour of the
stated rules, not an implementation bug**. The fix belongs in a named rules
revision.

## Improvement shipped this cycle

`grokbot/tests/test_referee_fuzz.py` adds 330 parametrized cases plus 1 strict
xfail, all pure Python, running in about 5 s. It is formatted with ruff. It
does not touch `src/`.

## Next steps

- jekko: the one-line deepcopy in `HitTracker.restore` (F1). When the next
  rules revision comes, add a `"recovered"` reason (F2) and reuse this fuzz as
  its differential oracle.
- grokbot: extend the fuzz to `MatchReferee` four-glove interleavings and bell
  timing (same-tick trades, final-substep impacts).
- grokbot: add per-policy blocked-as-faller rate and fall-time columns to the
  ending-panel probe. This covers cycle-0004 F2, the "grab while falling"
  no-result exploit.
