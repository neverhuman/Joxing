# Deliver a recorded bout

The product target is a replay of two autonomous, muscle-driven athletes sharing
one physical world: approach, guard, punch, miss or block, recover and respond.
The replay must be backed by the same state/contact/event evidence used to score
the match. Current native valid bouts: **0**. Current priority is making this
visible loop executable, alongside the fixed motor pilot; no larger unchanged
walking experiment should delay integration.

## Ordered delivery and acceptance

1. **Conclude the current v10 motor question.** Run the registered two-seed
   residual pilot once after the full Linux gate. Publish initial/final survival,
   actual updates and every failed/not_run evaluation. A failed pilot redirects
   the next motor work to stance/guard/reach in the match scene; it does not
   authorize another larger walking sweep. No policy is called a boxer on the
   strength of this pilot.
2. **v11: record the connected shared loop.** Follow V0.0.11_MISSION.md, including
   the body adapter, simultaneous observations/actions, causal history,
   actual staged contacts, referee and application replay. Wire
   `BoutRecorder.capture()` into the completed-state side of `BoxingEnv.step()`;
   never call it between the two players' observations. Keep the recorder a
   passive consumer. Both corners retain 700 muscles. Record the first
   preregistered scenario and all its failures, rather than selecting a handsome
   take. Deliver a diagnostic MP4 plus full tape and checksum receipt even if
   motor readiness is still blocked. This clip is visibly labeled diagnostic.
3. **Reach an intelligible exchange.** Qualify actual guard/reach/return motor
   controllers in this same scene and test head/torso/guard coverage, self-contact,
   direct-hit versus push, rearming, fouls/falls and contact replay. The first
   unassisted bout target is at least 10 simulated seconds with both athletes
   upright and responding to opponent state, and at least three measured attack
   attempts across the two corners. Show misses and blocks as well as any legal
   hits. Attack attempts need preregistered kinematic definitions; a controller
   label cannot count as a punch. This is a proposed acceptance target for the
   next frozen protocol, not a delivered result or a relaxation of motor gates.
4. **Show learning, then scale.** After game/motor/replay gates, train a learner
   against immutable historical opponents. Record initial/intermediate/final
   bundles on the same fixed scenarios, with a held-out panel and swapped sides.
   Report panel improvement, invalid/fall rates, completed games and full-loop
   games/hour. Scale 1,000 → 10,000 → 100,000 games only when measured learning
   and capacity justify it; diagnostic resets never enter that denominator.

## Recording implementation and remaining integration

Implemented `src/joxing/bout_recording.py`: read-only INTEGRATION state capture
of the complete dual scene; controller identities; frame and tape checksums;
clock/physics-count checks; explicit byte/frame bounds; causal video sampling.
`scripts/render_bout_recording.py` restores only scratch display data, uses
`mj_forward` for visualization and takes **zero new simulation steps**. Its first
format is diagnostic-only; points are explicitly unavailable until the match
referee supplies an audited ledger. It cannot certify a competitive bout.

Recording state samples are not a full application replay: the environment must
also retain history, queued actions, policy state, referee encounters, RNG and
the unsampled contact/event tapes. Never reconstruct forces or award points
from `mj_forward` during rendering. The current HUD's contact peak is supplied
by the live collector, a model prediction, and is not a calibrated impact force.
Upgrade the schema explicitly when real ledger/ending data are connected.

Read first: AGENTS/RTK, MISSION, STATUS, GAME, SELF_PLAY_EXECUTION_PLAN,
V0.0.10_D2_PILOT and V0.0.11_MISSION. One integration owner, no agents; do not edit
third_party, old receipts or a running checkout. New integration changes belong
in the v11 files already enumerated there. Generated scene files go under
artifacts/generated; tapes and videos under bounded runs/v0.0.11. Keep committed
compact receipts and publish large videos/raw as checksummed release assets.

Validation now:

```text
rtk proxy uv run --project experiments/v0.0.10/runtime --locked pytest -q tests/test_bout_recording.py
rtk proxy uv run --project experiments/v0.0.10/runtime --locked python scripts/render_bout_recording.py --recording runs/v0.0.11/first-scenario/recording --out runs/v0.0.11/first-scenario/bout.mp4
```

The second command requires a real saved tape; none is claimed to exist yet.
Rendering needs the visualization dependencies (`uv sync --extra viz` in a
development checkout; freeze a separate runtime for formal v11). Test failure
or wrong model/hash/clock stops publication. Current tests cover passive capture,
both complete bodies, bounds, no overwrite, causal sampling and corrupted tape
metadata rejection. Formal render qualification must decode the resulting MP4,
check frame count/duration/HUD, and verify a render receipt with zero new physics.

Final acceptance artifacts: `bout.mp4`, `recording.json`, `states.npz`, complete
contact/event ledger, initial/final and mid-contact application snapshots,
model/rules/observation/policy/decoder/source hashes, commands/seeds/assistance,
all attempted bout rows and an independent audit. HUD must show clock, A/B
points, event reason/location/impulse, policy identity and assistance status.
Never hide an early fall, accelerate footage without a speed label or call a
scripted/assisted diagnostic learned self-play. The next handoff must name the
next executable command and the remaining motor/game gate, not just a version.
