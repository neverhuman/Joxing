# H26: native Mac reference with strict local replay and explicit backend ownership

Status: worker-ready plan only; NOTimplemented/preregistered/run. One integration
owner, no agents. Objective: use spare xbabe4 for original700+700native physics
without pretending its cached muscle/observation math can resume exact Linux
snapshots. H25measured385/1400control differences at<=7.45e-9beforephysics and
FAILED its strict transfer gate; keep it failed. This is a separate Mac execution
profile, not a tolerance increase, repaired Linux result or claim of motor gain.

Non-goals: cross-architecture bit replay, changing model/geometry/muscle dynamics,
source scoring, direct pose control, pretrained teacher transfer, learner training,
GPU/Docker, large sweeps, H23retirement reader changes or a version tag.
Assumptions/dependencies: H24Mac source29/29 and H25source21/21; frozen original
D0package/runtime/assets available in /Users/agent/joxing/mac-replay-preflight-
20261102-01/oracle read-only. Reuse VERIFIED asset cache via fresh dedicated Git
checkout; do not execute in or edit that old target root. Native Mac NumPy2.4.2/
MuJoCo3.8.0/Python3.12.3, no source driver/library services changed. Recheck exact
pins, actual jobs, Macload<=16/reclaimable estimate>=16GiB/disk>=10GiB. No other
owned simulation task may overlap; aggregate<=16 and this packet only1thread/world.

Read first, in order: AGENTS.md,/Users/bentaylor/.codex/RTK.md,README.md,
docs/GAME.md,docs/MISSION.md,docs/STATUS.md,docs/IMPLEMENTATION.md,research/README.md,
research/H25.md,H26.md,docs/MAC_NATIVE_REPLAY_PREFLIGHT_RESULTS.md,
docs/V0.0.11_D0_RESULTS.md and original recording-protocol.json, then original
Git d2b582ee05041c21129b393945240b42e51153f8 sources src/joxing/{env,replay,
boxing_evidence,scene,physics,timing,history,observations,referee}.py.

Ownership and likely edits: new scripts/qualify_mac_native.py,audit_mac_native.py,
tests/test_mac_native_reference.py,experiments/v0.0.11/mac-native-reference-protocol.json,
fresh artifact/result dirs. Existing source/defaults/referee/scene schemas and
ALLthird_party/history/H25/H23roots are READ_ONLY. No concurrent writers.
Do not patch original ReplaySnapshot.restore, sigmoid, force samples or actor
caches. Do not use a cached Linux snapshot with a new label. New wrapper encloses
locally generated original snapshots in a separately hashed backend envelope.

Ordered implementation:
1. Define external BackendProfile JSON fields: execution_id=mac-arm64-f32-v1,
   platform/machine, Python/MuJoCo/NumPy versions+loaded binary SHA, pinned model/
   source/runtime hashes, np.show_config text SHA and CPU math environment/thread
   settings. Avoid credentials or full env. Actual selected process identity and
   discovered binary paths/hashes are receipts; use a prospectively specified
   immutable profile id and compare exact profile digest for replay ownership.
   Snapshot envelope {backend_profile_hash, original_snapshot, envelope_sha256}.
   Reject missing/different profile or snapshot hash BEFORE original env.restore.
   No changes to original source schema and no legacy/raw escape route.
2. Fresh original D0scene constructor; require nu=na=1400,700percorner and exact
   pinned model/source inputs. Freeze both actors' observations before decisions,
   use original untrained diagnostic_command, private seeded RNGs and A1/B2tick
   queues. Native-keyframe reset,1msimplicitfast,20msheld controls, no external
   assistance. Actor is an untrained fixture, not a pretrained boxing motor.
3. Preregister5ordered rows,116physics total:
   source90101 native-keyframe0→40 horizon40 (40steps), capture actual boundary20
   and mid23; boundary replay20→40(20), mid replay23→40(17);
   source90104 close-overlap-collision-probe-v2 0→20(20), capture actual contact1;
   contact replay1→20(19). Same seeds/controller/bundles/timing as D0, separately
   identified Mac outputs. No successful episode selection/retry. Midrestore must
   finish17pendingticks with no new decision. Generated Mac endpoints40/20 and
   complete tapes AREavailable references, unlike assuming original D0stored a
   full application snapshot at220. Native counts77+39=116, allsteps charged.
4. Compare each replay against its OWN saved Mac source tail bit-exactly: complete
   engine hashes, full final integration+application snapshot, held/applied/chosen
  700commands and excitation, decision inputs/histories, arrival/application
   times, RNG/cache/queue ownership, contact geometry/signs/forces/impulses/events,
   referee and simultaneous endings. Use existing saved-evidence auditor with
   explicit profile-aware wrapper, don't weaken integer or floating comparisons.
   Require positive actual force in contact stress; held-interval action count
   must stay exact. Retain wrong/unused actuator, warning/nonfinite failure rows.
5. New tests: cross-profile/forged envelope rejection before constructor/step,
   no changed source/default sigmoid, missing/duplicate/reordered cases, invalid
   pending action/history/side/RNG, tiny changed contact/command/hash rejected,
   false counters/overbudget/partial constructor, independent auditor rejecting
   false pass and importing model/physical stepping on saved-evidence-only path.
   Unit fake worlds test accounting, not native equivalence. Include relevant
   source history/timing/rules/observations/referee controls. Collect exact case
   count and freeze it before target outcome; preserve all source failures.
6. Before targetphysics: implement independent auditor and machine protocol;
   publish exact wrapper/oracle/source/input/runtime/profile/control-test hashes,
   expected5rows and caps, anonymous verify. Native budget160physics (116planned
   +44reserve, any extra auditphysics counted BEFOREstep),60srun/16MiBnewraw/
   4GiBsampled ownedRSS/1simulationworker/thread; setup180s/sourcecontrols30s/
   savedaudit30s, whole300s FIRSTlimit inclusiveallsetup. Bounded supervisor must
   use ORIGINALmonotonic start throughout; post-run audits are separately labelled,
   no virtual rebasing of task clock. Track actual descendants/PIDstart/progress/
   limits/source before/after; don't conflate RSS estimates with Linux exit proof.
7. On first fault/nonfinite/unsupported contact/protocol/ABI/source mutation or
   resource cap: stop own process group, keep allattempted/valid/invalid/not_run
   rows/fault prefix and honest conservativephysics reservation. No unchanged
   retry, no tolerance change or use of H25failed arrays as initial caches.
8. After exit, independent all-row/tail/force/physics/resource/source/profile/
   PIDaudit. Keep raw/arrays under runs only; small compressed summaries, commands,
   hashes/faults in Git. Update README/COMPUTE/STATUS/IMPLEMENTATION/v11book/H26/index
   and push. No v11tag, fitness/policy promotion or boxing claim.

Completion:5strict locally valid/audited cases or complete retained stop. Passing
would qualify only this Mac-native execution profile for a subsequent1/2/4
persistent-worker benchmark at equal workload. Linux strict snapshots/H25remain
incompatible; never pool byte-equality/physics performance across profiles or
silently change frozen opponents' backend. Later fixed-panel motor transfer,
contact sporting attribution/full replay and meaningful training remain gates.
Cross-platform behavioral/tolerance research is separate, not required to first
use Mac as its own deterministic reference. A working physical boxing recording
still requires a stable pretrained motor and qualified arm/contact integration.
