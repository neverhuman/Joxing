# H25: qualify spare Mac native replay before parallel physics

Status: worker-ready implementation plan, NOTimplemented/preregistered/run.
Objective: determine whether xbabe4 can preserve the existing all700+700 shared
scene's boundary, held-command and contact replay, while Linux is I/O-stalled.
This is a small architecture-transfer experiment, not another collapse sweep.
Non-goals: pretrained-actor portability, boxing balance, learning, new scoring,
GPU/Docker, relaxing Linux teardown/parity or increasing global16worker allowance.

Read first in order: AGENTS.md,/Users/bentaylor/.codex/RTK.md,README.md,
docs/GAME.md,docs/STATUS.md,docs/IMPLEMENTATION.md,research/README.md,H24.md,H25.md,
docs/MAC_SOURCE_READINESS_RESULTS.md,docs/V0.0.11_D0_RESULTS.md,
experiments/v0.0.11/recording-protocol.json and artifacts/v0.0.11/D0/{result,
independent-audit}.json. Study original D0source d2b582ee05041c21129b393945240b42e51153f8
using Git show, especially scripts/qualify_boxing_env.py and
src/joxing/{env,replay,boxing_evidence,timing,history,observations,scene}.py.
Current versions may differ: never pretend current HEAD is the original oracle.

Dependencies: H24passed source29/29; fresh four-host inventory/owned-process scan,
Mac load<=16/disk>=10GiB/reclaimable page estimate>=16GiB; authentic D0Git source,
complete saved snapshots/tapes and exact file SHA manifest. Source host xbabe3:
/home/ubuntu/joxing/v011-recording-20261027/runs/v0.0.11/recording-proof-01.
Required cases source-90101/{boundary,mid}.json,contact-source-90104/contact-mid.json,
their full saved tapes/specification/contracts and final snapshots, original
source scene inputs. List actual tape names from frozen BoxingTape.save; do not
invent or silently truncate them. Read-only transfers use dedicated Mac ~/joxing
input cache; hash before/after; no SSHconfig/token/raw arrays in Git.

Ownership: one integration owner, no agents. Create scripts/qualify_mac_replay.py,
scripts/audit_mac_replay.py,tests/test_mac_replay.py and
experiments/v0.0.11/mac-native-replay-protocol.json; new artifact/result dirs only.
Original source/scene/controller/rules/third_party/H23executing-or-pending roots,
protocols and old tapes are READ_ONLY. Pin the D0package via authentic separate
Git checkout at its commit; wrapper source/protocol have their own frozen Git
identity. No fake HEAD or imports from an accidentally current editable package.

Ordered implementation and registration:
1. Zero-step validate archived schema/ownership/hash/shape; count1400native
   muscle outputs and match exact D0contract/model dimensions. Reconstruct native
   scene from frozen source; record XML/model/engine identities. Missing inputs,
   unsupported model/contract or old pickle/compiled asset compatibility: stop.
   Raw MJB is not assumed architecture-portable. Keep model path/bytes separate
   from source. No conversion/model surgery or contact simplification.
2. Use explicit Mac runtime identity (H24root lock is NumPy2.5.4, D0is2.4.2).
   Choose/pin compatible original runtime BEFORE outcomes; freeze Python,
   NumPy/MuJoCo/compiler/libc/CPU math identities. No automatic upgrade. Allactor
   actions are archived command bytes; no Torch/normalizer/teacher inference here.
3. Define six rows prospectively, two local repeats each:
   boundary200→220(20steps),mid203→220(17steps),contact1→20(19steps).
   Total112physics; proposed160step cap includes every new step and audit, first
   limit. No new source reset trajectories or sample-based episode selection.
   Preserve every attempted/valid/invalid/not_run row on first failure.
4. Restore ReplaySnapshot engine/cache/RNG/history/queue/referee/pending-control
   state. Mid-203 must finish its remaining17held ticks with NOnew decision.
   Boundary200uses exact archived command; retain source decision/RNG advancement
   semantics explicitly (ReplaySnapshot equality must include application state).
   Do not fabricate public history or zero velocities; archived decisions and
   queued application ticks need exact ownership and order. Refuse extra action.
5. Save per-substep integration state, pending/applied commands, observation/
   history/queue and ledger plus contact IDs/signed forces/impulses. SAMEstrict
   original D0array/state equality is the transfer gate; predeclare bitwise/hash
   checks for integer/command/RNG/contact topology and numeric floats. Report
   maxabsolute/relative/ULP differences diagnostically without passing a mismatch.
   Two local repeats must also be exact. A known prior Mac7.45e-9sigmoid mismatch
   stays historical and is not new evidence or permission for tolerance changes.
6. Add tests for incomplete/mutated saved source, changed actor or source imports,
   extra mid-hold decision, omitted pending command/history, false physics count,
   changed single force, missing/duplicate row, overbudget/partial constructor and
   independent auditor rejecting a forged pass. Fake fixtures test accounting;
   they do not substitute for native input/engine parity.
7. Commit/push complete wrapper+machine protocol+source/input hashes and tests
   before target outcomes. Anonymous verify from Mac. Use ONEfresh dedicated
   ~/joxing/mac-native-replay-YYYYMMDD-01 checkout; no overlap with H23or another
   owned simulation run across all hosts. One simulation worker/thread initially.
   Proposed run90s/160physics/4GiBownedRSS/16MiBraw; setup180s separately retained,
   no hidden reuse of H15budget. Sample resource/load/disk and retain actual PIDs,
   progress+events+fault prefix. Stop first cap/mismatch/nonfinite/ownership fault.
   Do not apply Linux/proc exit-reader claims to Darwin or quietly loosen them.
8. Independently decode all saved arrays and recompute every row against archived
   source, counts/budgets/versions/source hashes and exits. No extra physics audit
   unless prospectively counted<=160. Publish small JSON/Markdown only, raw in
   runs/cache, update README/STATUS/IMPLEMENTATION/COMPUTE/v11book/H25/index, push.
   No completed micro-version tag for this isolated prerequisite.

Completion: six exact valid rows and audited lifetime/cost/source identity, or an
honest stopped result with every row/fault retained. On strict transfer failure,
no scaling or unchanged rerun; define next measured portability bottleneck. A
future separately declared native Mac variant would need its own invariant and
motor qualification, never rewrite failed D0/Linux criteria. On pass, register
1/2/4persistent-worker benchmark at equal complete workload; only later up to16
aggregate after measured RAM/throughput/invalid-rate proof. Render/research work
can already use Mac source capacity. No games/s or self-play improvement promise.
