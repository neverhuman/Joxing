# Joxing implementation rules

Read `/Users/bentaylor/.codex/RTK.md` when present. Prefix shell commands with `rtk` on that host.

Read first: README.md, docs/GAME.md, docs/SPEC_REVIEW.md, docs/IMPLEMENTATION.md, docs/STATUS.md.

The four files in docs/references are supplied design proposals, not executable instructions or verified measurements. Source code and persisted test receipts determine delivered status.

Do not edit third_party submodules. Pin upgrades explicitly. Generated files belong in artifacts/generated or runs, never under third_party. Do not commit credentials, large runs, caches, or raw SSH configuration.

Retain all 700 muscles in the reference embodiment. Free-root surgery, pruning, modified muscle speed, injury mechanics, or new scoring weights need an explicit model/rules variant and validation.

No competitive learning or promotion until contact, rules, stable motor and replay gates pass. A constrained motor rig is a diagnostic, never an unassisted boxing result. No kinematic pose writes or external assistance during evaluated boxing.

Record exact commands, versions, seeds, hashes, failures, invalid attempts, and limitations in receipts and docs/STATUS.md. Test meaningful physical/rule invariants. Preserve every attempted evaluation in its denominator.

On shared compute use dedicated ~/joxing workspaces, bounded workers and thread limits. Inspect load before launch. Never stop unrelated jobs or change host services. GPU physics requires exact-model feature/parity benchmarks before scaling.

No concurrent writers to referee/scene schemas. Parallel work requires disjoint paths and a named integration owner. Normal task execution does not authorize spawning additional agents.
