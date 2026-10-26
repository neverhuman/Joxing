# Status and execution log

Date: October 26, 2026 (America/Denver).

## Initial source delivery

- Studied all four supplied specs; preserved originals under docs/references and reconciled decisions in SPEC_REVIEW.md.
- Created private `neverhuman/Joxing` using an existing authenticated neverhuman account; no token was copied into files or output.
- Inspected xbabe1–4 read-only; selected xbabe1 for bounded first runs.
- Pinned anatomy and msgym read-only submodules at supplied commits.
- Compiled full dual anatomy with 1,400 muscles, 84 equalities, 170 qpos/qvel, 165 bodies including four added gloves, 715 geoms. Total mass 121.565046 kg matches expected. Tested actuator dynamics/gains/bias/control ranges match upstream exactly.
- Implemented staged force capture, independent signed-impulse/momentum fixtures, pure candidate hit referee and snapshot/restore, canonical reward shaping, CPU benchmark and constrained muscle-reach search.
- Initial local validation: 27 tests passed in 2.88 s. Fixture momentum error ~1.3e-15 in the axis-aligned case; not empirical glove calibration.

Production scene fit/self-contact, complete simultaneous environment, foul/fall/boundary adjudication, unassisted motor competence, Warp parity and tactical self-play remain incomplete.

## Completed execution and receipts

- Source bootstrap committed as `464ff5b`; provenance fix/probe at `6a1ee06`; GPU variant/replay at `1c0c58c`; physical-scoring integration and formatting at `b0f3772`.
- An initial remote deployment exposed a provenance bug: `git rev-parse` in a copied submodule directory found the parent repo. Fixed by checking the repository root and falling back to committed gitlink pins; added regression test. Reran learning rather than publishing the mislabeled receipt.
- Verified learning command on xbabe1: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/joxing train-reach --workers 4 --generations 16 --population 24 --out runs/reach-seed7-verified`. Completed, no background training left. 384 training episodes plus 32 evaluations each for initial/trained/random controls. Error 0.338643 → 0.043747 m; wall 34.921 s; zero invalid episodes. Code receipt references `6a1ee06` and correct upstream pins.
- Run location: `/home/ubuntu/joxing/bootstrap-20261026/runs/reach-seed7-verified`; local copy `/Users/bentaylor/Documents/ChatGPT/JOXING/runs/reach-seed7-verified`. Every generation stores candidates, rewards, scenario seed, mean/std. Final checkpoint and before/after replay arrays remain there; curated result/checkpoint in `artifacts/receipts`.
- Rendered and visually inspected `artifacts/scene-inspection.png`, `reach-learning.png`, and before/after MP4. Replay shows assisted motor task, not an autonomous boxing match.
- Added real solver-to-referee fixtures: head 2 / torso 1 / guard 0; all emit exactly one event with physical impulse. Full two-human region/ownership integration still pending.
- GPU runtime probe rejected Warp 1.18 on existing driver. Warp 1.12 sees 4090; exact model rejected native mesh margin/MULTICCD combination. Named disabled-MULTICCD variant runs eight worlds but misses short-state parity (6.49e-4 > 1e-4). Preserve all three receipts; no reference change or host driver update.
- Native/locked local tests and lint pass. Final test output and benchmark receipts are stored under `artifacts/receipts`; latest counts appended before handoff.

Final source verification: **31 tests pass on both local Mac and xbabe1 Linux**; Ruff passes. Final xbabe1 single-world diagnostic median: **534.2 physics substeps/s** (three 500-step repeats). Local initial diagnostic median: **876.7 substeps/s**. These are not end-to-end training rates. Controlled Linux-to-Mac replay validation and all failed GPU probes are preserved alongside successful receipts. No training or GPU probe remains running.

## Next implementation gate

Follow P1 in IMPLEMENTATION.md: complete external shell/native-overlap review and explicit self-contact; build fitted anatomical frames and geometry-derived face/clearance/motion evidence; investigate scalar-root fall singularity. In parallel only through disjoint write scopes: isolate GPU divergence on small muscle/contact fixtures. Next learning milestone is unassisted stance and independent reach/return under gravity, before any tactical PPO self-play run.

## v0.0.2 — launched iteration

User requested a concrete next proof iteration and named it v0.0.2. Plan: `docs/V0.0.2_PLAN.md`. This deliberately integrates learned muscle motion with a physical glove target, geometry-derived strike evidence, one-event referee and full mid-contact replay before full-body competition.

Development mechanical check: an explicit conversion of the prior reach controller makes a real torso-target impact, earns exactly one point despite 165 force-producing contact steps, but does not withdraw/rearm. This is the baseline that v0.0.2 training must improve. Candidate rules remain unchanged. Fixed pelvis/zero gravity remain explicit diagnostic assistance.

Package version set to 0.0.2. New strike rig keeps all 700 muscles, adds exactly one inertial glove and a fixed spherical target. Source exposes strike-pilot/train-strike; frozen proof suite uses training seeds 7/19/41, 64 held-out reset scenarios each, and a four-worker CPU cap. Full unassisted stance and self-play remain later gates.

Pilot completed on xbabe1 under source 0eba6fe: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/joxing strike-pilot --out runs/v0.0.2/pilot-01 --workers 4`. 128 training episodes, 16 evaluation attempts, 34.809 s, zero invalids. Initial hit/withdraw rates 100%/0%; fitted policy 100%/100% on 8 development resets. First-contact snapshot at tick 368 reproduced the full final engine state exactly and matched the ledger. Pilot evidence remains in that run directory and local runs/v0.0.2/pilot-01.

Before proof launch, froze a smaller 12×24 training budget per seed based on pilot speed/success. Final evaluation IDs 20000–20063 and diagnostic/replay ID 20000 are disjoint from pilot's 10000–10007. Acceptance remains 80% hits and 75% strike-withdrawal per seed, no invalids, exact ledger/≤1e-9 state replay; physical role controls also require head=2, torso=1, guard=0 and passive=0 for the fixed diagnostic scenario. No final data has been used to select policies or alter thresholds.

## v0.0.2 — completed proof

- Executed source `3f2be02`; frozen protocol `36c229d690f79a60edeab302948529342980c2ae6fa766c40785a3512641848a`.
- Launch command on xbabe1: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/joxing train-strike --out runs/v0.0.2/proof-01 --workers 4`. Main PID was 1920776. Completed in 256.813 s; no worker or training process remains.
- Three seeds 7/19/41 each trained 288 episodes, then evaluated initial/final policies on 64 paired fresh resets 20000–20063. Total 864 training episodes, 384 evaluation attempts, 21 diagnostics, zero invalids. Every seed achieved 64/64 legal single hits and 64/64 strike-withdrawal successes. Baseline achieved 64/64 hits but 0/64 withdrawals for every seed.
- Role controls passed (head 2, guard 0 with actual physical impulse, passive 0). They are target-proxy variants, not anatomical zone or guard-occlusion validation.
- All three mid-contact restores matched the entire final engine integration state and event ledger exactly. Independent Mac replay/audit also passed for all three checkpoints, including raw evaluation accounting and event identities.
- Labeled plot/video inspected; fixed video is seed 7 / scenario 20000, not a selected favorable clip. Curated proof, pilot, protocol, raw archives, local/Linux test logs and replay audit are in `artifacts/v0.0.2/`. Full raw runs remain in both local and xbabe1 `runs/v0.0.2/`.
- Final validation: 40 tests and Ruff pass. The added preservation test confirmed native actuator parameters, body masses and inertias remain unchanged. A missing test import was caught during development and corrected before final verification; no physics/rules change resulted.

Read `docs/V0.0.2_RESULTS.md` for measured scope and reproduction commands. Next iteration should remove assistance through a validated stance/reset and balance gate, not jump directly into self-play. Existing Warp parity failure remains open.

## v0.0.3 — test plan prepared

User requested the next version of tests and a Git push. [V0.0.3_PLAN.md](V0.0.3_PLAN.md) defines the sequential implementation handoff against released source `88b76604e87e62e2681e2adc15af0651636183ac`: native-root stress tests, support/assistance audits, validated resets, a bounded full-body feedback learner, three-seed quiet-stance/push evaluation, replay and compute receipts.

The plan separates a completed test platform from a passed motor gate and measured learning improvement. Formal targets are 61/64 quiet stance and 58/64 recovery successes per seed, zero invalids, with all attempts retained. Initial compute cap is four CPU simulation workers; total planned cap is 40 million physics substeps/four hours, subject to fresh host availability. GPU simulator parity remains unresolved.

This delivery contains planning/documentation only. No v0.0.3 implementation, new physics measurement, training job or evaluation has started. Package/tag v0.0.2 and its evidence remain current. Next action is plan step 1, then the root/support/reset probe before learner work.

Planning validation: `rtk proxy uv run python` checked all local Markdown links and balanced code fences in the four changed documents; passed. Confirmed 768 planned evaluation attempts and unchanged package version 0.0.2. The first link-check invocation used unavailable bare `python`; rerunning through the locked uv interpreter passed. `rtk proxy git diff --check` passed. No executable code changed; the prior 40-test result remains the latest physics/code verification. The planning commit is pushed to `origin/main`; its Git commit identifies this handoff, with no v0.0.3 release tag.

## Public repository and training-guide clarification

At the user's explicit request, changed [neverhuman/Joxing](https://github.com/neverhuman/Joxing) from private to public using `gh repo edit neverhuman/Joxing --visibility public --accept-visibility-change-consequences`. Authenticated metadata confirms `visibility=PUBLIC`, `isPrivate=false`. Independent requests without authorization headers returned HTTP 200 for repository metadata and raw README. HTTPS `git ls-remote` with an empty credential helper and disabled prompts succeeded. Both upstream submodule repositories were independently confirmed public. Sanitized timestamped verification is in [repository-public.json](../artifacts/receipts/repository-public.json); no credential was stored or printed.

Added [documentation index](README.md) and [training guide](TRAINING.md), expanded the project README with the ordered v0.0.3 training/test gates, and linked the current iteration from GAME, COMPUTE and IMPLEMENTATION. The detailed plan now preregisters paired recovery-success improvement as the primary learning metric; quiet stance and stable seconds remain secondary. This prevents selecting a favorable primary metric after final evaluation. No final v0.0.3 evaluation has occurred.

v0.0.3 remains the next implementation version: root/support/reset validation, feedback environment tests, bounded SAC pilot, protocol freeze, three-seed training and 768 planned paired stance/recovery attempts. Pilot seed is 73; continuing to formal training requires at least 16/32 development starts to survive 5 seconds, with zero invalids. Bootstrap seed is fixed at 61003. Package/release remains v0.0.2. No training or compute job was started by this documentation/visibility update.

Validation: nine changed/new Markdown documents passed 61 local-link/anchor checks and balanced-fence checks; the public receipt JSON passed its visibility/HTTP assertions. `rtk proxy git diff --check` passed. No executable code changed, so the source/physics regression results above remain unchanged. Push the documentation and public-access receipt together to `origin/main`, then verify that the public default branch points to the pushed commit.
