# H33: pretrained walking works in separated lanes; stopping reference fails

Both intact700-muscle athletes completed all four fixed three-second separated
walking trials without a physical failure. **No stationary stance or boxing was
qualified.** Slowing the own-reference clock over400ms caused falls even without
opponent contact. Do not treat a frozen walking target as a standing controller.

| Condition | Valid trials | Both complete3s | Stance passes A/B | Points |
|---|---:|---:|---:|---|
| Separated lanes, original walking |4/4|4/4|0/4,0/4|All0–0|
| Separated lanes,400mswalk-to-hold |4/4|0/4|0/4,0/4|All0–0|
| Opposed,400mswalk-to-hold |4/4|0/4|0/4,0/4|All0–0|

Every condition includes seeds71000/71016/71032/71048. Walking root excursion is
2.568–3.194m and final-second mean speed0.989–1.103m/s across all eight corners.
Those are useful walking measurements, well outside the registered stationary
limits(.30m/.15m/s). Separated stopping fails at1.272–1.628s; opposed stopping's
first failures occur at.769–2.043s. Some opposed partners remain unfailed when the
pair ends; they did not demonstrate full three-second survival.

All12trials and all three203+17held-command proofs independently replay exactly
in a fresh process: all1400teacher commands, every1mscontact/metric, every saved
native state/frame. No extra teacher decision occurs during the held17steps.
All136fresh target source checks pass. No new pretrained policy optimization.

## Evidence, resources and denominator

Source [e788c5d](https://github.com/neverhuman/Joxing/commit/e788c5dbb4537761ca4e414a14003372c5cc170c)
and [prospective protocol](../experiments/v0.0.11/native-stance-probe-01.json)
were public before launch; protocol canonical SHA
15902121a164e528409772c5792c79bc63f879797d4a878d112e58c09c7ba026.
One formal attempt:1valid/0invalid;12attempted/12valid/0not_run trials, all
six primary/independent held proofs complete. Valid falls are negative outcomes.

xbabe4setup3.829s; original whole257.957s<1200. Source63.285s<120;
physical83.353s<360; audit78.229s<300; recordings5.858–6.464s<30each.
Peak sampled ownedRSS5,575,065,600B<8GiB. All19observed identities including
owner subsequently exited; no owned task remains. Fresh host capacity and exact
PID/start/commands/source/runtime/input hashes are in the receipts/protocol.

23,871primary+23,871independent replay=47,742new native entries; source4096reserve
puts total charge51,838<100000. Successful source loops are statically2674steps,
not a measured entry counter. Raw234files/132,887,715B<384MiB; full tapes remain
onxbabe4at~/joxing/native-stance-probe-20261103-01/repo/runs/v0.0.11/native-stance-probe-01.
Git retains compressed textual outcomes/manifests and small recordings only.

[Conclusion](../artifacts/v0.0.11/native-stance-probe-01/summary.json),
[all12primary outcomes](../artifacts/v0.0.11/native-stance-probe-01/primary-summary.json),
[independent audit](../artifacts/v0.0.11/native-stance-probe-01/audit-summary.json),
[all raw names/bytes/hashes](../artifacts/v0.0.11/native-stance-probe-01/raw-manifest.json),
[ownership/resources](../artifacts/v0.0.11/native-stance-probe-01/owner.json),
[exits/step ledgers](../artifacts/v0.0.11/native-stance-probe-01/exit-and-physics-audit.json).

Development failures remain: phase endpoint rounding and compiler-cache drift
from translating before compilation. The final named placement compiles original
opposed constants then translates only fixed root frames before evaluation;
allother compiled numeric arrays remain bit-exact. No tolerance was adopted.
All four development attempts/6367static fixture steps are [retained](../artifacts/v0.0.11/development/h33-development-01.json).

## Complete fixed recordings and visual limits

- [Original walking,3s](../artifacts/v0.0.11/native-stance-probe-01/recordings/lanes-walk/lanes-walk.mp4).
- [Separated stopping, ends1.28s](../artifacts/v0.0.11/native-stance-probe-01/recordings/lanes-stop400ms/lanes-stop400ms.mp4).
- [Opposed stopping, ends2.06s](../artifacts/v0.0.11/native-stance-probe-01/recordings/opposed-stop400ms/opposed-stop400ms.mp4).

All three are the entire first fixed reset71000, with151/65/104frames at50fps,
independently decoded/hash-verified. All nine first/mid/lastPNGs were visually
reviewed. Walking shows two upright athletes with low hands; one becomes partly
cropped by the final camera frame. Stopping shows leaning/falling and body/head
contact, not a legal punch. All temporal frames remain. Camera framing needs a
prospective fix. Tendons/shadows/reflections are hidden visually only; physics,
allmuscles, actual archived poses and failures remain. [Full media QA](../artifacts/v0.0.11/native-stance-probe-01/media-qa.json).

## Post-hoc reset defect and next action

This additional scratch-only analysis was NOTpreregistered; it executed0physics
and changed no evaluated state. Native walking qpos contains root displacement,
so the original fixed attachment offsets do not fix actual athlete positions.

| Reset seed | Native phase(s) | Actual pelvis XYgap(m) | Reset interpenetration |
|---|---:|---:|---|
|71000|.22|1.148465|None observed|
|71016|.38|.806951|None observed|
|71032|.62|.280446|None observed|
|71048|.76|.084457|Up to.119165m|

Seed71048produces interactor force at the very first1msinterval. Other opposed
stopping contacts begin at48/361/450ms. Contact precedes failure, but this does
not by itself establish its causal contribution. The failed stance criteria and
all valid diagnostic outcomes remain unchanged; boxing initialization is unqualified.
[Exact reset geometry](../artifacts/v0.0.11/development/h33-posthoc-reset-geometry-01.json).

**Next:[H34phase-independent actual reset placement](NATIVE_RESET_PLACEMENT_MISSION.md).**
Correct static spawn frames before motion while preserving native reset states,
qualify clearance/replay/walking controls, then preregister upper-only guardRL
using the proven walking motor. Stop/turn/guard under interaction and legal
strikes still precede historical-opponent self-play and large parallel game runs.
H31's failed motor result/invalid recording attempt remain; wholev0.0.11untagged.
