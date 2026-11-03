# H34: actual starting positions and walking control qualify

The phase-dependent reset overlap is fixed. All eight native starts now place
the actual pelvises at their declared coordinates, with no initial penetration,
at least5cmcross-player surface clearance and no opponent contact force during
the first100ms. Both athletes preserve4/4three-second separated walking trials.
Every trial and both held-command proofs replay exactly in a fresh process.

**This is correct initialization, not boxing skill.** The opposed walkers still
fall after later contact, and all eight trials score0–0. No stationary guard,
legal punch, competitive self-play or new policy learning is demonstrated.

| Placement, original walking controller | Valid native trials | Clear first100ms | Both complete3s | Points |
|---|---:|---:|---:|---|
| Corrected opposed actual starts |4/4|4/4|0/4|All0–0|
| Corrected separated actual starts |4/4|4/4|4/4|All0–0|

Seeds71000/71016/71032/71048 are all retained. Corrected opposed trials end at
1.60/2.02/1.76/2.38s; first force-producing contacts occur at589/548/649/1247ms.
They are thigh/foot/glove-guard/low-body contacts, not qualified scoring punches.
Do not interpret longer time before contact as RL improvement.

## What changed and what was tested

Each reset first retains the original compiled native constants, then translates
ONLYtwo fixed root XYframes before evaluated motion. The actual native joint
qpos/qvel/activation, height, reference clock, actor, .075decoder, excitation,
all700muscles/body, collisions, mass, solver and sporting rules remain unchanged.
The seed/native-state/axes/fixed offsets and reconstruction recipe bind the scene
and episode hashes; changed frames or foreign-seed restores fail.

Maximum actual reset XYerror is2.220446049250313e-16m against the1e-9mgate.
Every reset queries all729cross-player physical surface pairs, including native
meshes. The5cmquery cutoff yields a LOWERBOUND, not an exact minimum distance.
NativeCCD is required and analytic positive/clipped/penetrating controls plus the
original native-overlap negative control pass. [MuJoCo's signed-distance API](https://mujoco.readthedocs.io/en/3.8.0/APIreference/APIfunctions.html#mj-geomdistance)
documents the clipped return semantics.

Source [6be3dde](https://github.com/neverhuman/Joxing/commit/6be3ddeb626bbacd3c3f2e1ca1da712b355bc675)
and [packet02](../experiments/v0.0.11/native-reset-placement-02.json), canonical
SHA5162874c7aa184e9cedc0ebfb00c09316ea655ccef36c685f25eb21766732f55,
were public before entry. All163fresh target controls pass. All1400commands,
every1mscontact/metric, every saved engine/frame and actual203+17held states
reconstruct exactly; no extra inference during a held command. Next-decision
continuation also passes source fixtures with a synthetic teacher; it is not
claimed as an additional actual-checkpoint post-hold inference experiment.

## Both formal attempts remain

Attempt01is INVALID: its source fixture retained eight compiled scenes and sampled
ownedRSSreached8,655,896,576B>8GiB at98.605s. It stopped BEFOREscientific physics;
all eight trials/two proofs/audit/two recordings are not_run. Whole113.285s,
155rawfiles/52,201,503B retained, later all owned identities exited. Its136printed
pass markers do not constitute a completed sourceJUnit or full-source qualification.
[Original fault and all missing rows](../artifacts/v0.0.11/native-reset-placement-01/summary.json).

The repaired fixture keeps at most two scenes and small full-byte model hash
receipts for all eight cases. All252production src/scripts files match the original
74906d5 byte-for-byte. No case, tolerance, scientific criterion, clock or resource
limit was eased. The standalone numeric oracle also passes. All seven development
attempts/12,066static fixture entries remain separately recorded; they precede
the formal study and are not pretrained learning data.
[Development/repair receipts](../artifacts/v0.0.11/development/h34-memory-development-01.json).

Attempt02COMPLETESand qualifies initialization/walking control. Thus two formal
execution attempts:1valid/1invalid. Of16registered trial slots,8valid native trials
and8prior not_run;0invalid solver trials. Nothing is discarded from its denominator.

xbabe4whole299.794s<600; source103.523s<120; physical84.171s<180;
audit80.847s<180; videos8.019/8.472s<30each. Peak sampled ownedRSS4,235,542,528B
<unchanged8GiB. All32observed identities including owner subsequently exited.
20,234primary+20,234independent replay=40,468new scientific physics entries.
Combined current/prior source charge8192 gives48,660formal charge<60000.
The earlier failed sourceclock/cap is not extended. Raw214files/120,494,631B
<256MiB; full tapes remain outsideGit onxbabe4under
~/joxing/native-reset-placement-20261103-02/repo/runs/v0.0.11/native-reset-placement-02.

[Conclusion/all gates](../artifacts/v0.0.11/native-reset-placement-02/summary.json) ·
[all eight outcomes](../artifacts/v0.0.11/native-reset-placement-02/primary-summary.json) ·
[independent audit](../artifacts/v0.0.11/native-reset-placement-02/audit-summary.json) ·
[raw names/bytes/hashes](../artifacts/v0.0.11/native-reset-placement-02/raw-manifest.json) ·
[ownership/resources](../artifacts/v0.0.11/native-reset-placement-02/owner.json) ·
[step ledgers/exits](../artifacts/v0.0.11/native-reset-placement-02/exit-and-physics-audit.json).

## Complete first fixed recordings

- [Corrected opposed walking, entire1.6sevaluation](../artifacts/v0.0.11/native-reset-placement-02/recordings/reset-centered-opposed-v1/reset-centered-opposed-v1.mp4).
- [Corrected separated walking, entire3sevaluation](../artifacts/v0.0.11/native-reset-placement-02/recordings/reset-centered-lanes-v1/reset-centered-lanes-v1.mp4).

Both use first fixed reset71000,81/151frames at50fps including initial/final
states. Independent decode/hashes and all six first/mid/lastPNGs pass visual QA.
The new bounding-sphere camera keeps both whole athletes framed, including the
formerly cropped walking endpoint. It changes visuals only; no temporal frames,
native muscles, physical poses or failures are removed. Opposed footage still
shows low hands, body contact and falls. H33's older cropped video remains intact.
[Media QA](../artifacts/v0.0.11/native-reset-placement-02/media-qa.json).

**Next:[H35upper-only guardPPO](NATIVE_ARM_GUARD_RL_MISSION.md).** Learn a visible
guard while preserving the qualified walking motor, using causal state history,
protected teacher commands, fixed before/after phases and exact native replay.
Then stopping/turning/contact guard and legal strikes precede historical-opponent
games and parallel scaling. Wholev0.0.11remains untagged; prior failed motor and
recording outcomes remain unchanged.
