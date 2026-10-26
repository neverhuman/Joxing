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

Package version set to 0.0.2. New strike rig keeps all 700 muscles, adds exactly one inertial glove and a fixed spherical target. Source exposes strike-pilot/train-strike; frozen proof suite will use training seeds 7/19/41, 64 held-out reset scenarios each, and a four-worker CPU cap. Full unassisted stance and self-play remain later gates. Pilot/proof commands and receipts are appended after execution.
