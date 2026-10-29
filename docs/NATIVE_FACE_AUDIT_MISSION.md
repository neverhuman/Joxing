# Native equipment-face and point-velocity diagnostic

Status: concluded; all192mechanical controls and the independent audit pass.
Read [NATIVE_FACE_AUDIT_RESULTS.md](NATIVE_FACE_AUDIT_RESULTS.md) and
[research/H9.md](../research/H9.md) for scope, measured outcomes and limitations.
The original preregistered mission follows. Read AGENTS.md,
MISSION.md, STATUS.md, GAME.md, NATIVE_SOURCE_PARTIAL_RESULTS.md and
NATIVE_BALANCE_STRIKE_NEXT_PACKET.md before the new protocol and source.

The current fixed native clip has two torso onsets that pass approach and
active speed but fail face cosine/striking-face. The signed contact tape is
incomplete, so those old measured normals/points cannot be reconstructed.
This task checks the adapter's declared mechanical convention on the actual
intact native embodiment. It cannot establish anatomical knuckle alignment,
real force calibration, a successful punch, balance or a boxing result.

## Scope and implementation

No existing scene, referee, rules, actor, normalizer or collector changes.
Retain all 700 muscles per athlete; source compiles the pinned native scene.
Only new observer/evidence/qualification modules, their tests, entry points,
protocol and documentation are owned. No agents, third_party source edits,
old-run writes, other-chat messages or unrelated host changes.

Read src/joxing/scene.py, referee.py (MatchReferee.stage), physics.py,
native_face_audit.py, native_face_evidence.py, native_face_qualification.py,
then tests/test_native_face_audit.py and tests/test_native_face_evidence.py.
The new entry points are scripts/qualify_native_face_audit.py and
scripts/supervise_native_face_audit.py. The frozen machine protocol is
experiments/v0.0.11/native-face-audit-protocol.json.

The reference sphere sits 45 mm distally from the native hand origin, has
62 mm radius and identity rotation in the hand frame. The existing scorer
uses glove-body local −Y and the outward sphere hemisphere. Independently
derive that axis from hand origin to glove center and compare the staged
face scalar and hemisphere. This is a virtual equipment convention; no
independent anatomical landmark is inferred from the symmetric sphere.

Fixed denominator: **192 synthetic rows** = two scratch wrist poses × two
athletes × three prescribed velocity profiles × two hands × head/torso ×
front/back hemisphere × two geom orders. Twelve explicit scratch qpos/qvel
states and four actual A/B 180° spawn-equivariance checks are retained.
Poses are the native key pose and a ±0.05 rad clipped wrist-hinge offset;
the latter is a kinematic probe, not a dynamically qualified reset.

Contacts use the actual glove frame and named opponent target role but do
not lie on the actual opponent surface and do not supply a measured force.
Their 10 N scalar is synthetic. No referee commits or points are generated.
Rear-hemisphere controls must fail the face predicate even with positive
synthetic force. Reversing geom order must preserve all semantic measurements.
An opponent-only velocity profile must not create own active glove motion.

Compute independent material-point velocities from central differences of
scratch FK at q±1e−6·qvel using mj_integratePos/mj_kinematics. Compare native
Jacobian velocities with a fixed 2e−7 m/s tolerance; require a nonzero rotational
witness above1e−4 m/s so a center-velocity substitution cannot pass. Geometry
tolerance is1e−10. No integration routine may run: mj_step/step1/step2 are
guarded and any call faults. Original scratch engine fields stay unchanged.

The separate read-auditor rebuilds the compiled native model, verifies its
binary hash, reconstructs all twelve prescribed states and independently
recomputes geometry/material-point derivatives and staged predicates. It
never calls the production referee/point_velocity or candidate audit. Source
rows, ordered entered/completed events and the full saved row journal must
match exactly. Missing, duplicated, corrupted or relabeled rows fail.

## Validation and bounded execution

Development: **20 tests passed in 9.23 seconds**, Ruff passes. Mutations include
face sign, center velocity, asymmetric wrist/yaw, saved velocity/state/geometry,
duplicate rows, tolerance/model hash, NaN and false physical classification.
The complete report/journal is checked to fit 4 MiB with fault/supervision
reserves. These are development controls, not a formal calibration outcome.

Before formal outcomes: commit and push source/protocol; anonymously byte-verify
public implementation/protocol/mission; inspect host capacity and owned jobs;
create a fresh dedicated xbabe1 checkout:
/home/ubuntu/joxing/native-face-audit-20261029. Initialize only the pinned
MS-Human submodule; no upgrades or source edits. All268 declared source/runtime/
asset files and upstream Git pins must match before and after execution.
The protocol preserves the old native ledger and every prior failure unchanged.

Launch `python3 scripts/supervise_native_face_audit.py --protocol-sha256
<canonical SHA> --source-sha <frozen commit>` from the fresh root. Locked sync
180 s, exact20targeted Linux tests45 s; then proof30 s and separate audit20 s,
combined postgate60 s with termination allowance inside the total. One numerical
worker/thread, **zero simulation steps**, 4 GiB process-group RSS/4 MiB new raw;
256 KiB supervision and2 MiB fault reserves. Resource floors8 GiB available
RAM/10 GiB free disk. CPU forward kinematics only; no rendering or GPU work.

This isolates an observer; it does not qualify a new scientific collector or
repeat the unchanged full physics suite. A fresh full Linux source gate still
precedes collector adoption or any new formal physics. Stop on the first
identity, case, predicate, tolerance, resource or deadline fault. Preserve all
entered/completed/failed/not_run rows and the old incomplete native tape.

Outputs: runs/v0.0.11/native-face-audit-01 (source/scene/rows/events/results)
and native-face-audit-supervision-01 (commands, PID/start/group identities,
progress, logs, exits/faults). On completion independently verify all source/
asset/output hashes, full raw transfer and every owned identity's exit, then
update README/docs/experiment book and push compact reports to public Git.
Do not tag v11 or claim anatomically calibrated punches or learning.

## Following mission

On consistency, preserve the current face rule and move to pretrained balance
adaptation. Inventory compatible pretrained standing/locomotion tasks and
their actual checkpoints first. Build a zero-residual parity interface around
the upstream motor that stayed upright: immutable actor/normalizer, unchanged
reference phase and all700outputs; declare any upper-limb action ownership and
bounded residual explicitly. Do not freeze a walking gait clock to pretend it
is a guard policy. Register the finite adaptation/reset/replay pilot and qualify
new source before new physical outcomes. Competitive learning remains behind
the motor/contact/rules/replay gates, including the known snapshot-alias issue.

On inconsistency, retain every negative row and define a named geometry or
measurement correction with positive/negative parity controls. Never flip an
axis or weaken a face threshold simply to make the old touches score.
