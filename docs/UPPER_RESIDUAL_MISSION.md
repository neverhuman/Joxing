# H10: isolate arm adaptation around the pretrained full-body motor

Status: implemented; formal saved-command qualification pending public freeze.
This is a zero-integration prerequisite, not a balance or boxing result.

## Objective, evidence and boundaries

The fixed native clip leans severely; increasing self-play volume cannot fix an
unqualified motor. Preserve the compatible walking teacher's actor, normalizer,
reference phase, 700 outputs and excitation transform. Add a separately named
bounded arm-command interface, then test its ownership and held-record contract.
No teacher inference, physical integration, optimization, scoring or recording
runs in this packet. No anatomy, rule, old decoder or collector changes.

Read in order: AGENTS.md; /Users/bentaylor/.codex/RTK.md; research/README.md;
research/H10.md; docs/NATIVE_BALANCE_STRIKE_NEXT_PACKET.md; docs/SHARED_MOTOR_PHYSICAL_RESULTS.md;
src/joxing/motor_residual.py; shared_motor_inputs.py; upper_residual.py;
upper_residual_evidence.py; upper_residual_qualification.py; both matching tests;
experiments/v0.0.11/upper-residual-protocol.json; scripts/supervise_upper_residual.py.

One owner; new upper_residual modules/tests/scripts and these docs only.
Forbidden: third_party edits, previous run roots, referee/scene schemas, unrelated
Grokbot work and existing decoder behavior. No agents or overlapping writers.

## Inventory and explicit variant

At the pinned msgym commit ad3aac166dab3577d7d16b4a03faccef94c7aac7,
the three registered tasks are full-body locomotion, legs locomotion and right-arm
manipulation. The [observed Checkpoints release](https://github.com/LNSGroup/msgym/releases/tag/Checkpoints)
contains those three archives; it has no standing or running archive. This is
an observation of that release, not a claim about every available controller.
See development/pretrained-checkpoint-inventory-01.json under artifacts/v0.0.11.
Leg-only and 81-muscle manipulation policies are not drop-in full-body motors.
QFlex is a separate research lead, recorded in research/H13.md; no code adopted.

`native700-arm-file-residual-v1` identifies muscles by the canonical primary XML
arm-file membership and verifies compiled order/tendons/dynamics. It retains all
700 muscles: 122 selected, 578 protected. Exactly 46 of the original 138 groups
touch the arms; no mixed groups exist in this checkpoint. Mixed groups are still
handled by the per-muscle mask and covered by a deliberate unit fixture.

Compact actions map to existing groups, with the closed 0.1 normalized-command
amplitude and float32 arithmetic. All protected command bits remain unchanged
given the same teacher output; zero residual preserves every one of 700 bits,
including signed zero. Caller applies the existing excitation transform once.
Arm muscles can span the torso; preserving commands is not a physical balance
guarantee. Future changed states can also change the frozen teacher's output.

HeldUpperCommand binds actor/game/sample tick, immutable teacher/action/output
bytes, mask/source/group digest, 1ms clock and 20-tick hold. Snapshot restoration
validates the external owner/clock and payload exactly. This is a command-record
roundtrip, not engine/controller/RNG continuation or a replacement for v10 replay.

## Registered offline test and stop conditions

Use the original 176-frame, paired 700-command tape from source71000 and the
byte-identical frozen checkpoint JSON. Do not deserialize weights, infer new
teacher commands or fabricate missing force evidence. Exact paths/hashes in the
machine protocol are prerequisites. Native compilation verifies membership only.

Fixed 2,204 cases: all176 frames × A/B × six patterns (zero, ±1, alternating,
ramp, fixed uniform seed93019), then 46 one-hot arm groups × A/B at frame0.
Require bit-exact independent scalar-oracle output, all578 protected channels,
all352 zero cases, immutable hold at offsets3/19, expiry at20, JSON restoration,
exact ordered entered/completed events and complete independent saved-hash audit.
Invalid, failed and not_run rows stay in the denominator. Stop on any mismatch,
source/runtime drift, worker overlap or first wall/RSS/raw limit; no retry here.

New finite offline allowance: one numerical process/thread, zero simulation
workers/steps, 4GiB process-group RSS, 4MiB aggregate raw, 60s combined postgate
including 1.25s termination reserve; proof35s, audit20s. Locked sync180s and exact
28-test gate45s precede it. These do not amend any old native budget/deadline.
Source/runtime inputs and all pinned model assets are hashed. Require 8GiB free
RAM/10GiB disk and fresh shared-host owned-process inspection before launch.

Ordered execution:

1. Run the two new test files and Ruff; persist development receipt. Freeze and
   push source/protocol; anonymously byte-verify public files before formal work.
2. Fresh xbabe1:/home/ubuntu/joxing/upper-residual-20261029; checkout frozen SHA,
   initialize both pinned submodules without editing them. Verify all manifests.
3. Launch `scripts/supervise_upper_residual.py --source-sha SHA --protocol-sha256 HASH`.
   It locks runtime, gates exact tests, runs proof then separate scalar read-audit.
4. Monitor actual PID/start identities/events/resources. Audit all source/input
   hashes and exits; transfer every raw name/size/hash. Keep full journals outside
   Git; commit small result/count/hash/exit/transfer summaries and all faults.
5. Update H10, README, MISSION, STATUS, IMPLEMENTATION and experiment book; push.
   No micro-version tag or boxing/balance promotion from this offline test.

Development command: `rtk proxy .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_upper_residual.py tests/test_upper_residual_evidence.py`.
Linux uses experiments/v0.0.11/runtime/.venv/bin/python after `uv sync --locked`.
Ruff: `rtk proxy uvx ruff check src/joxing/upper_residual*.py tests/test_upper_residual*.py scripts/*upper_residual.py`.

## Next physical integration packet, after this prerequisite

Do not immediately train or reuse the exhausted old budget. Read the shared
task's begin_control/held-command snapshot paths, teacher bundle and qualified
stored writer. Implement a new opt-in wrapper, leaving SharedMotorTask unchanged.
Teacher sees its original causal walking state/phase once per control interval;
compose arms once, preserve cached output through every substep/restore. Bind
teacher actor/normalizer/reference/group/mask/clock/RNG and model/rules identities.

First qualify zero residual on identical unassisted states against the teacher,
including a mid-hold restore with no extra decision and exact signed-contact
tapes. Run the fresh full Linux suite before activation. Register a *new* finite
pilot protocol with counted source/replay/audit/render stages, complete ZIP_STORED
output size arithmetic/fault reserve and owned CPU capacity. Never reuse a prior
source/test receipt for the changed collector. No launch budget is granted here.

Physical question: can the teacher keep walking while a small arm task changes
hand pose without degrading posture/support? Use paired neutral and arm-task
resets, both sides, fixed seeds/phase/targets; publish every attempt and full fixed
clip per condition, including falls/zero points. Check actual root lean, support,
survival, glove reach/return, command effort, face orientation and exact held
continuation. Choose/freeze numerical gates before outcomes. No external support,
root pose writes or frozen gait clock. If zero-residual transfer fails, stop and
isolate scene/clock/reference effects before training. If arms disrupt balance,
train a motor task with explicit posture/support and reach-return targets before
opponent learning. Standing guard remains a separate skill.

Only after stable motor/contact/rules/replay gates: causal opponent history with
timing queues, immutable historical opponents and held-out league evaluation.
Scale 1→2→4→8→16 owned CPU workers using measured complete-workload throughput,
not save speeds or invalid falls counted as games. GPU physics still needs
exact-model feature/parity proof. No 100k-game sweep substitutes for these gates.
