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

## v0.0.4 — implementation and pilot launch

User requested a major implemented push. v0.0.3 was planning only; [v0.0.4](V0.0.4_PLAN.md) implements its native-gravity prerequisites and a bounded SAC development pilot. New balance/support/group/environment/training modules preserve 700 muscles, 42 equalities, segment inertia and two 0.283 kg gloves. CPU-only Torch wheels are locked for Linux; simulation remains CPU MuJoCo.

Initial source inspection measured native keyframe equality error 0.010506. Reset-only Newton projection, 200 ms gravity preparation and final position/velocity projection produce accepted bilateral-foot states at <=2 mm penetration and <1e-8 equality error. Both projections and their correction magnitudes are logged; no state writes occur after the measured episode begins. Native-root coordinate singularities are not assumed resolved by a short finite probe.

Development checks: `rtk proxy uv run pytest -q` passed 57 tests; Ruff passed. A 64-transition/two-worker smoke run (`runs/v0.0.4/smoke-01`) completed with four paired evaluation attempts, zero invalids and no policy improvement, as expected before the 10,000-transition learning warmup. The local development probe (`runs/v0.0.4/probe-dev-01`) accepted all 32 resets and completed twelve 0.8-second root-stress cases at 2/1/0.5 ms without numerical faults. These pre-commit development receipts carry the source hash and are not the published pilot proof.

xbabe1 fresh inventory: load ~5.75, 41 GiB available RAM, 328 GiB disk free, RTX 4090 idle. Use the existing dedicated checkout and at most four CPU simulation workers, one BLAS/OpenMP/Torch thread. Commit source before Linux validation/probe/pilot. Planned command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run joxing train-stance --out runs/v0.0.4/pilot-01 --steps 100000 --workers 4 --seed 73 --wall-s 1800 --eval-count 32`. All 64 development evaluation attempts remain in the denominator. Formal three-seed motor proof and self-play require the stated continuation gates; their completion is not implied by launching this pilot.

Source `92db82688ee9930ed40aae60d6510ef4b6d210a1` passed 57 Linux tests (20.70 s) and all 32-reset/twelve-case root probes. Pilot launched on xbabe1 as PID 2868420; log `runs/v0.0.4/pilot-01.log`, live progress/checkpoint in `runs/v0.0.4/pilot-01/`. Source uses CPU PyTorch 2.14.1+cpu / stable-baselines3 2.9.2. Source/wall/physics/seed identity is persisted before learning.

Independent smoke audit caught JSON key-order leakage in support observations: a sorted file changed restored right/left/non-foot channel order. Fixed observations/trace to use explicit semantic order, added a sorted-JSON observation regression, and verified closed-loop smoke continuation now has zero full-state error. This corrects snapshot decoding; normal unsorted live training observations and the running pilot's learning path are unchanged. The failed development audit is preserved before its corrected audit. Snapshot guarantees in this release cover control boundaries (including during a push), not arbitrary interruption inside a Python control step. Learner checkpoints are policy/optimizer bundles; full replay-buffer/RNG training resume is not implemented.

Additional verification: malformed snapshot state, shape, unknown fields and clocks are rejected before mutation. Optional motor integration exercises actual SAC gradient updates, checkpoint reload and external-horizon bootstrap/final observations. Latest Mac suite: 63 tests pass with motor dependencies (16.92 s); Ruff passes. Linux initial suite was 57 tests before these additional restore/learner cases; rerun the full final suite after the pilot exits.

Initial-policy Linux-to-Mac audit preserves fall outcome but misses the predeclared 1e-5 full-INTEGRATION-state tolerance: max state error 2.115e-4 (solver warmstart), qpos error 4.707e-8, qvel error 1.316e-6. Same-host sorted-JSON continuation is exact. Preserve the cross-host failure, do not relax its tolerance or claim closed-loop platform parity. The bounded pilot is also update-bound: warmup rollout throughput is materially higher than throughput after SAC begins gradient updates. Physics-only throughput is not reported as learner throughput.

## v0.0.4 — completed implementation and measured pilot

Pilot source `92db826`, frozen protocol `4591a775b84bb2a278df9a8699fb85edd46f8020e0452c6e58b61774844e0a9c`, seed 73. PID 2868420 finished. Training stopped at its 1,800 s wall cap with 39,040/100,000 control transitions and 388,573 training physics substeps; 60,960 scheduled transitions remain unrun. Final evaluation then completed all 64 planned paired attempts, zero invalids; total wall 1,848.367 s. Initial/final mean survival 1.017531/1.023000 s; paired gain 0.005469 s, 95% reset interval [-0.000281, 0.011625] includes zero. Initial/final five-second survivors 0/32 / 0/32; final twenty-second successes 0/32. No convincing motor improvement. Continuation gate failed; zero of the proposed 768 formal evaluations were attempted.

Source fixes/audit tools at `675caf8`, `de2c09b`, `91a2573` preserve model/action/observation identities and the live training path; source on xbabe1 was deliberately left at `92db826` until the run completed, so run provenance remains accurate. The later code fixes sorted snapshot decoding and validates it before mutation. Final suite: 63 tests on Mac (16.92 s) and Linux (26.54 s); Ruff passes. Linux initial/final policy snapshot continuation has zero full-state error. Mac independently confirms checkpoint/row/summary identities and zero repeated local-restore error, but strict cross-host full-state parity remains failed: trained max error 2.412e-3 (qpos 8.102e-7, qvel 1.600e-5); outcome and tick still match.

Post-run command on xbabe1: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python scripts/benchmark_balance.py --run runs/v0.0.4/pilot-01 --out runs/v0.0.4/benchmark.json`. Three repeats at 1/2/4 workers measured medians 65.71/131.81/240.52 control transitions/s. This includes policy inference/support/reset and excludes optimization/referee. Gradient-phase SAC throughput is roughly 17–18 transitions/s. An in-flight reset diagnostic found a large entropy/reward-scale mismatch; a preregistered ablation is required to establish its effect. No GPU compute was used.

Published portable evidence/policies, raw rows, progress, model/reset/root audits, all failures, test logs, benchmark, independent audits, inspected fixed-scenario plot and video are in `artifacts/v0.0.4/`. The run, evaluation and benchmark are complete; no iteration job remains active. See [V0.0.4_RESULTS.md](V0.0.4_RESULTS.md) for exact reproduction and next gates. Remaining work: reward/entropy and optimizer-throughput experiments, qualified motor skills, full boxing geometry/referee and self-play. Game points are unchanged. Final source/evidence/docs are pushed together and tagged v0.0.4 with this limited scope.

## v0.0.5 — preregistered controlled wave

User requested study, the strongest next test wave, execution/conclusion and a public GitHub push. Plan committed first as `09cbacb`; [V0.0.5_PLAN.md](V0.0.5_PLAN.md) and [experiment book](../experiments/v0.0.5/README.md) define exact interventions, splits, conditional replication and stop rules. No model, muscle, observation, physical failure or boxing-point rule changes.

Implementation adds strict immutable protocol, constant-control sanity test, real optimizer profiling, matched 2×2 reward/entropy SAC screen, conditional two-seed confirmation, independent raw-row/actor/action-replay audits and report curation. Four workers, one thread each; 40,000 transitions/900s per run, at most eight training runs. Confirmation scenarios stay unopened until each scheduled final actor is frozen. The five-second and formal motor gates remain unchanged.

Development smoke: 64transitions, seed149/armB, four workers, 12 optimizer updates, IDs62010/62011, four valid paired evaluations in11.95s, no motor gain. First invocation failed before execution because the parent run folder did not exist; fixed creation while preserving overwrite rejection. Both attempts are retained in `artifacts/v0.0.5/development/smoke.json`. Latest pre-final suite:69passing tests; an additional fault-accounting test is being validated. Source/protocol will be committed before shared-host execution; formal outcomes remain pending at this entry.

v0.0.5 live execution: source `6ac2bad`, protocol SHA256 `7a4ca4a6b841ca39ca8141db83d1e0e05b01d49054408d71197ac28ecff97fe5`, frozen source SHA256 `8b45216daf642dc2ec9e76b37d0e76533310666d26d7c8943e6e9bf33ba20ba0`. Mac70tests/34.72s and Linux70tests/54.91s passed before launch. Main PID3802532, log `~/joxing/bootstrap-20261026/runs/v0.0.5/training-output.txt`, registry/events under `runs/v0.0.5/wave-01/`; source checkout must remain frozen until exit.

B0 completed80valid evaluations; selected constant0.05 averaged1.0414s on32screen starts. P0 passed: optimized91.99vsreference19.62updates/s,4.69×. S-A completed40,000transitions, initial/final1.01775/1.01156s, gain−0.00619s. S-B completed40,000, final1.09094s, gain0.07319s, reset95%CI[0.06841,0.07766]; both arms0/32five-second survivors. At this entry S-C is running, S-D/selection/conditional replication/X0 remain pending. No learning claim is finalized from these partial screen results.

User explicitly requested increased host limits. Registered resource addendum `b3ec872` raises aggregate allowance4→16workers without changing the active scientific protocol. H0 used4/8/12workers alongside the four-worker wave; nine timed samples/14,400transitions, zero invalids, medians220.48/436.01/646.79rollout transitions/s; excludes optimization. H0 completed;16-worker standalone test is reserved until the wave exits. Partial Linux actor/action continuation is exact. Mac action replay errors<7e-11 but closed-loop full-state errors3.58e-5/9.21e-4 exceed unchanged1e-5; repeated local continuation is exact. Failed platform parity is retained. Mechanical reset validity does not imply rest: B0 starts tilt8–9degrees with pelvis-tilt velocity~1.1rad/s, so describe this task as initial-fall recovery.

Original v0.0.5 wave completed in2,358.383s:160,000training transitions,336evaluations, zero invalids,2,148,064accounted physics/preparation steps. S-D finalmean1.120156s/gain0.102406s was best; no arm passed0.25s improvement or five-second continuation. Four original replication slots not_run. Linux independent accounting/actor/action continuation audit passes; Mac accounting passes but strict cross-host closed-loop parity fails. Original job PID3802532 exited. H0 completed all4/8/12/16pool sizes,24,000timed transitions, zero invalids;16-worker median844.67transitions/s with optimization excluded. Both plots and fixed scenario60000/armD video were inspected; instability is visible.

User explicitly authorized increased workers AND training budgets. Extension plan/protocol/source `76f6e5c` was committed before extension outcomes.72tests pass Mac55.40s/Linux83.23s, including two parallel trials, deferred initial evaluation and peer-stop preservation. Separate frozen checkout `/home/ubuntu/joxing/v005-extension-20261026`; source SHA256 `ed1379dccbcbe57fe558f99643f637b4aa8224411ee2dfeae5aac5512ef9d37b`. Launch UTC19:46:43, PID4172953; log `runs/v0.0.5/extension-output.txt`, per-cell logs/registry under `runs/v0.0.5/extension-01`. Four concurrent learner jobs × four simulation workers, one thread each;160,000transitions/2,700seconds per cell. New screen seed547/IDs64000–64031; conditional confirmation seeds761/983/IDs65000–65031. All original physics/learner conditions and gates remain fixed. Preserve both waves, do not edit active extension source, and finish independent audits/results/public push after its final decision. Deployment initially referenced unavailable FETCH_HEAD in the clone; explicit76f6e5c checkout succeeded before any extension experiment.


### v0.0.5 extension screen freeze and state/history review

Source `76f6e5c` remains frozen on xbabe1. All four extension screen trials completed 160,000 transitions and 39,500 updates each. Means A/B/C/D: 0.893156/1.313469/1.245156/1.126563 seconds, initial 1.059063; no five-second survivors. Selection B cleared the registered thresholds (gain 0.254406s, versus A 0.420313s). Before confirmation outcomes, curated every completed screen actor, row, progress record and baseline; independently recomputed paired summaries/selection and wrote `artifacts/v0.0.5/extension/screen-freeze.json`. Screen denominator: 288 new attempts, zero invalids; eight reused diagnostic rows excluded. Confirmation A/B jobs for seeds 761/983 are running under the same parent PID 4172953, four learners / sixteen simulation workers total. No source or action/observation contract changed.

User requested opponent state, motion history, responsive muscles, jitter handling and many short games. Read executable control/contact/rule/scene code and published `docs/STATE_HISTORY_SELF_PLAY_AUDIT.md` with a worker-ready handoff. Actual stance observations include 85 generalized velocities and 700 muscle activations; SAC has no recurrent state, opponent input, tactical referee or league. Offline command audit retains all eight original and extension screen traces. C showed sharp command changes and D saturation in the original preselected trace; these are diagnostics, not causal limb-jitter claims. A new isolated-actor timing script will measure inference separately from the unqualified wall-clock physics loop; no additional training or task alteration is involved.


### State/control diagnostics and upstream checkpoint inspection

Measured eight extension screen actors on Mac with `scripts/benchmark_actor_latency.py`: one Torch thread, 100 warmup calls and 1,000 timed calls per actor/batch case. Single-observation medians 0.062–0.064ms, p99 0.067–0.124ms; no >10ms calls among 16,000 batch-one/batch-four samples. Repeated fixed observations, no physics/transport/optimizer; not a full-game deadline guarantee. `scripts/profile_control_path.py` ran three unprofiled and three profiled recorded-action continuations through FaultRecorder on selected B scenario64000. Median unprofiled139.65control steps/s; middle profiled repeat ~76%native step1/step2, ~6%snapshot, ~2%point velocity. Same physical outcome/tick, maximum integration-state reference error1.997e-11, 6,498diagnostic physics steps within the reserved1million. Both JSON receipts retained under `artifacts/v0.0.5/extension/`; no active remote code changed.

Read pinned msgym observation, DynSyn layer/actor/config and evaluation source. Its public release has a 223,978,082-byte full-body checkpoint archive with SHA256 `2b2f0f1007d530db76463e7d3d5fb309b52c7b2a53649920ddfd9cf8a6880c57`. Downloaded to ignored `runs/upstream-checkpoint-inspection/LocomotionFull.zip`, checked the published hash, inspected only ZIP/JSON (no model/pickle/tensor deserialization). Saved metadata:45million upstream steps,3,601observation values,700outputs,138synergies, separate normalizer. These are upstream counters, not Joxing measurements. Source/metadata indicate scheduled DynSyn amplitude may be lost on load (saved top-levelNone vs schedule0.075); require reproduction before transfer interpretation. Added reproducible `scripts/inspect_upstream_checkpoint.py`, acquisition/metadata receipts and `docs/UPSTREAM_MOTOR_TRANSFER_PLAN.md`. No transfer rollout or upstream training was launched; v5confirmations continue unchanged. Local diagnostic scripts passed Ruff and their actual full-data executions.


### Confirmation-bank timing disclosure before fresh-seed outcomes

Review of `wave_extension.py:run_extension` identifies a distinction from broad preregistration wording: the frozen constant baseline evaluates confirmation IDs before the confirmation learners launch. Therefore the bank was not unopened by every process until final actors froze. Initial/final learner-policy evaluations are explicitly deferred until final actor save; training uses disjoint IDs110000–110063, and selectedB/config/criteria stayed frozen. This is disclosed in results before confirmation outcomes; historical protocol/addendum text is not rewritten. No execution/source mutation or retuning occurred. A future sealed-bank formal proof must defer the constant comparator too, or preregister that exception precisely.


## v0.0.5 completed and audited

All eight extension trials completed 160,000 transitions and 39,500 updates each. Confirmation seed 761: initial 1.022875s, A 0.803000s, B 1.293531s; B minus A 0.490531s [0.452125, 0.532286]. Seed 983: initial 1.010531s, A 1.165406s, B 1.097188s; B minus A −0.068219s [−0.096314, −0.040187]. B improves over own initial and constant on both seeds, but fails registered superiority versus A on 983. Every final policy has 0/32 five-second survivors. Full extension: 576 new valid evaluations / 1,280,000 transitions. Both waves: 912 new valid evaluations / 1,440,000 transitions / 354,000 scheduled updates. The 14,736 completed stochastic motor episodes are not boxing mini-games. No formal motor proof or self-play was launched.

Extension wall time was 3,555.388s, with 13,841,563 formal physics/preparation steps. Both waves total 15,989,627; adding the one-million supplementary reserve stays below 40 million. Every source-unchanged assertion passes. Independent Linux audit passes raw pairing/counts/summaries/selection/actor hashes/accounting and exact policy/action continuations. Mac accounting and repeated local restore pass; strict cross-host closed-loop parity fails (maximum 4.247 in mixed integration-vector units), while recorded-action continuations agree within 1.20e-10. Outcomes/ticks match on the examined snapshots. No tolerance was relaxed; zero invalid evaluations does not erase this failed portability gate.

After completion, ran source-host `scripts/verify_wave_extension.py --run runs/v0.0.5/extension-01 --original /home/ubuntu/joxing/bootstrap-20261026/runs/v0.0.5/wave-01 --out runs/v0.0.5/extension-audit-linux.json`. Ran the same auditor on Mac against copied raw outputs; exit 1 and the failed cross-host receipt are preserved. Ran final `scripts/report_wave.py` against completed extension data. A copy in ignored remote `runs/v0.0.5/report_wave_final.py` used `MUJOCO_GL=egl` for source-host video; source/script/input hashes are retained. Inspected screen/confirmation figures, the stochastic training figure and fixed first-scenario 64000 video preview; falling motion is clearly visible. Registered PIDs 3802532/3901039/4126632/4172953 exited, and no Python job remains in either dedicated checkout (`artifacts/v0.0.5/process-exit.json`). No unrelated process or service was changed.

Final diagnostics found a small accounting omission: latency/profile environment construction performs 200 preparation steps before timing; each control-state inspection performs 400. Preserve published raw measurements and add `diagnostic-accounting-correction.json`: these four diagnostics total 7,698 control/preparation steps, within the supplementary reserve. Future latency/profile scripts count preparation explicitly. Timing conclusions are unchanged. Corrected derived-plot bottom margins; scientific rows are unchanged.

Updated README, documentation index, game/master implementation/training/compute guides, results, changelog, experiment book and evidence index. Curated the machine-readable release summary, every raw attempt/log/policy, audits, visuals and manifests. The frozen extension source passed 72 tests on Mac/Linux; final scripts pass Ruff and actual-data execution. Next work: feasible stationary/recovery resets and proposed upstream teacher compatibility/load tests, then P1/P2/P3/P5 causal opponent/history/jitter/game/league integration. Exact commands, caps, stop rules and read-first files are in `docs/UPSTREAM_MOTOR_TRANSFER_PLAN.md` and `docs/STATE_HISTORY_SELF_PLAY_AUDIT.md`. No active job remains to monitor or resume. Final public push/tag verification follows this completion record.


### v0.0.5 publication verified

Pushed final evidence commit `f737dcb494f87825447937904619251978896bf5` and annotated tag `v0.0.5` atomically to `neverhuman/Joxing`. Anonymous GitHub API confirms `private=false`, `visibility=public`; anonymous raw reads of the tagged README, results, state/history audit and release-summary match local bytes. Tag target was independently resolved and checked. `artifacts/receipts/v0.0.5-publication.json` records this post-push verification; main adds only that receipt/status after the immutable scientific release tag. All requested experiments are concluded, all owned jobs exited, and no follow-up training is running.


## v0.0.6 mission started

User authorized the next value-add phase and continuing20-minute progress checks. Created active heartbeat `advance-joxing-every-20-minutes` in this chat after checking no matching Joxing automation existed. Current mission/ownership/state transitions are in docs/MISSION.md. Registered v0.0.6 loader/normalizer/native-task qualification protocol before loading the released checkpoint or running its motor trajectories:20paired seeds71000–71019,default/restored decoder conditions,40attempts,one CPU world,900s/1Mphysics/1GiBraw first limit. Separate runtime targets checkpoint SB3/Torch/Gym/NumPy versions; root runtime remains separate. xbabe1 preflight load5.80/128logical,~40GiBavailable RAM,~318GiBfree disk. No v6job has launched yet.


### v0.0.6 implementation/preflight

Separate checkpoint-compatible runtime locked and installed. Native path-only adapter preserves source timing/options and700muscles; global SAC alias restoration, immutable normalizer, explicit decoder schedule and corrected save/load are implemented. Eight target tests pass; initial full pinned-runtime Mac suite80passed/64.41s. Initial7pass/1fail was a toy replay harness stepping after valid termination; preserve receipt and correct loop, no physics threshold changes. Float32 action arithmetic is checked against the actual pinned upstream wrapper. Native snapshots retain the real last actor observation and trajectory/RNG state in addition to engine integration state.

Release-source check initially hit anonymous GitHub API rate limit; authenticated read succeeded without storing credentials. Release modelpin c049 differs from Joxing2d, but complete trees differ only in README; no physical asset changed. Native environment patch only changes model-file lookup. Receipts are curated. No large-scale upstream training, native formal evaluation or transfer has launched yet. Current phase is source freeze and Linux deployment/test.


### v0.0.6 formal launch

Frozen source `97565e90012b59bb16db8c45e4aa7f095a905c3c`, source SHA256 `880ff9eecb52b4507ec1cd4e9b6cbb63eb5ffefd06dc36345bb2c54dea19211a`. Linux complete suite80passed/94.03s before launch. Dedicated checkout `/home/ubuntu/joxing/v006-20261026`; previous checkouts unchanged. Archive transfer/hash verified; fresh capacity load4.85/128logical,46GiBavailable RAM,319GiBfree disk. Registered one-world qualification launchedUTC21:30:56,wrapperPID700993. Exact command,source/config/resources/log ownership in artifacts/v0.0.6/launch.json and docs/MISSION.md. No model, reward, physics or criteria changed after launch.


## v0.0.6 completed and audited; next mission ready

Registered native qualification finished all 40 attempts in 77.2294137 seconds with 70,400 physics steps, zero invalid/incomplete/not_run and unchanged source/input hashes. Default and schedule-restored modes both complete 20/20 upstream three-cycle tasks (3.52 simseconds each), zero sampled support failures. Native diagnostic criterion>=16/20 restored passes; this is a frozen upstream walking controller, zero new optimization and no boxing/stance claim. Both modes reach the ceiling, so no schedule-restoration completion benefit is established.

L0 confirms default layer amplitudeNone versus scheduled 0.075. Synthetic normalized-action probes change by mean absolute 0.030902829,max 0.074400783; corrected save/load error 0 and actor tensors unchanged. Normalizer stays frozen. Linux independent audit checks all rows/pairs/physics/actor/normalizer hashes and first-seed 71000 step 10 action/closed-loop/repeat continuation in both modes: all state/observation errors 0, strict<1e-9 passes (9,960 audit steps). Mac accounting/hashes pass but strict compiled contract differs by5.55e-17body mass/9.76e-19inertia; preserve failure before reference replay, no relaxed tolerance or portability pass.

Source-host report initially failed Gym's 480 pxframebuffer; a display-only 640 x 360 repair in an ignored copy generated the final recorded-action video without changing frozen evaluated source. Video adds 3,520 physics steps; plot and fixed first-scenario preview were inspected and show physical walking. Curation checks 135 raw files byte-for-byte, retaining original pending-replay result and using the later audit for final status. All development failures and limitations remain in artifacts/v0.0.6. PIDs 700993/701010 exited; read-only /proc scan found no Python jobs in the dedicated checkout. No unrelated jobs/services changed.

Full 80 tests pass Mac 64.41 s/Linux 94.03 s, final 8 target Mac checks 8.45 s. Updated README, game/master/training/compute/docs index/changelog, experiment book, results and evidence indexes/manifests. Public v 6 push/tag verification follows. Current mission is the worker-ready v0.0.7 glove/timestep/longer-native-motion plan; no v 7 outcomes or job yet. The active same-chat 20-minute heartbeat must inspect actual process/progress/faults/Git and advance the current mission, preserving ownership and bounded resources. It notifies meaningful changes/completion/blockers and stays quiet on unchanged healthy state.


### v0.0.6 publication verified

Pushed final source/evidence commit `bd48f96913b7184121034d54500667cff1e3b4f0` and annotated tag `v0.0.6` atomically. GitHub repository metadata confirms private=false/visibility=public; anonymous tagged README/results/next mission/release-summary reads return200 and match local bytes. Remote annotated tag resolves to the completed release commit. Post-push receipt: `artifacts/receipts/v0.0.6-publication.json`. Main adds this verification after the immutable scientific tag. The 20-minute heartbeat remains ACTIVE, verified from saved configuration, with v0.0.7 as its next bounded mission. No owned simulation job remains active.


## v0.0.7 preregistration and implementation

Heartbeat inspected clean Git and actual /proc ownership: no Joxing Python jobs. xbabe1 load8.31/128logical,~29.6GiBavailable RAM and311GiBfree disk. Registered four variants and80paired long-task attempts before outcomes in experiments/v0.0.7/protocol.json. Native body-ID/reference-index mismatch confirmed by reading both observation and reward paths; implement explicit canonical-name map and exact native oracle first. Maximum4worlds/one thread,1800s/1.7Mrunner physics/1GiBraw plus0.3Mreserved audit/video; no competitive learning or reward changes. Sole integration owner in this chat; third_party and previous evidence remain read-only.


### v0.0.7 development validation

All four MjSpec variants compile with700muscles/85qpos/85qvel, unchanged native masses/inertias/joint/actuator/tendon/equality ordering and explicitly separate0.566kg glove equipment. Reference body-name mapping handles shifted IDs in observations AND reward; native feature/state/reward oracle is exact. Development smoke is retained; no formal72000bank opened. Mac full98tests passed in126.46s. Final targeted checks add parent-only shared-stop ownership after review caught a potential concurrent temporary-file rename race; parent raw-byte sampling now tolerates atomic receipt renames. No scientific thresholds changed. Separate checkpoint-compatible v7runtime locked. Source freeze/Linux full regressions and real v6snapshot continuation remain launch prerequisites.


Timing clarification: bundled117frames at100Hz gives1.17s per cycle and21.06s for18cycles, not the plan's approximate21.12s (which multiplied rounded v6endpoints). Protocol18cycles and native termination remain unchanged. Report sampled endpoint/timestep rounding explicitly; do not change a gate to align durations. This was inspected from saved v6reset metadata before interpreting v7long outcomes.


### v0.0.7 formal launch

Frozen source9e3ee335c3995c860b40237bf6f00ba155d5d62d passed99Linux tests/151.18s. Validation wrapper955614completed and launched finite qualification UTC22:12:19,wrapper972834/Python972838 in dedicatedv007checkout. Fresh load14.90/128logical,41.4GiBavailable RAM and318GiBfree disk. Source/protocol were committed and pushed before outcomes; no executing source is changed. Prerequisite oracle and H1precede T1. Sourcehost command/capacity and final tests retained.


## v0.0.7 completed and audited

All 80 registered long native tasks completed in 141.4419 s with 0 invalid/incomplete/not_run. Native 2 ms/gloves 2 ms/native 1 ms/gloves 1 ms all 0/20 horizon completions; means 4.129/4.171/4.145/4.178 s versus 21.06 s threshold. Original native baseline also fails; no glove-only causal failure or new training gain is established. T 0 native feature/state/reward and actual v 6 continuation are exact. H 1 median 1/4 worker loops 91.24/256.50 TPS, startup-inclusive 4.64/15.53 TPS. Linux independent audit all 80 pairing/actions/hashes/metrics/physics passes, all 12 first-seed replay paths have 0 state/observation error; completion diagnostic remains false.

Post-hoc firstseed 72000 replay ends at native mean joint error 0.06027–0.06139, with foot support, 0 nonfoot load,10.2–11.9 degree tilt and 2.1–3.2 cm drop. Knee/ankle/shoulder deviations and body spatial velocities retained; actual predicted commands match recorded actions exactly. These are sampled tracking failures, not demonstrated physical falls or long physical survival. No gate/reference/reward/normalizer/anatomy was changed.

Counted physics:249,460 formal+7,460 prerequisite/benchmark+38,730 audit+13,510 video+13,510 posthoc=322,670, within 2 Mcap. No optimizer/GPUphysics. All owned PIDs exited and dedicated checkout has no Python jobs. Full 98 Mac tests plus final 27 target,99 Linux tests pass. Plot/video preview inspected. Every outcome and first scenario stays in Git; complete trajectories prepared in 22,225,280 byte public archive with SHA/full manifest. Copy initially lacked local parent, then unchanged rsync passed. Last periodic progress 79/80 is historical; final 80/80 registry/result is audited and preserved.

Updated results, experiment book, adapter contract, README/docs and next worker-ready v0.0.8 mission: substep physical motor evidence and explicitly named continuation after original tracking failure. It will determine the next bounded muscle-learning/reset mission without pretending that walking passes quiet stance or game gates. No v 8 job/outcome yet. Public push/tag/release-asset verification follows this conclusion.


### v0.0.7 publication verified

Pushed completed source/evidence commit `4cad56ddcb4fa83b053d8be3571d2ae601d81023` and annotated tag `v0.0.7`; created the public GitHub release and full raw archive asset. GitHub metadata confirms private=false/visibility=public. Anonymous tagged README/results/next mission/release-summary match local bytes. Anonymous full 22,225,280-byte archive download returns 200 and exactly matches its committed SHA256. Remote annotated tag resolves to the completed scientific commit. Receipt: `artifacts/receipts/v0.0.7-publication.json`. The 20-minute heartbeat remains ACTIVE and reads the current v8 mission. All owned jobs exited; no v8 outcome is claimed. Main adds this verification after the immutable release tag.


## v0.0.8 preregistration and implementation

Heartbeat verified clean Git/no Joxing Python jobs. xbabe1 load3.64/128logical,~47.7GiBavailable RAM and318.7GiBfree disk. Registered40fresh paired physical-continuation attempts and boundary/mid-control replay before outcomes. Code inspection verifies Gym assigns controls, batches mj_step, then calls mj_rnePostConstraint. Native Euler live pose/contact caches can describe interval start; end posture will use scratch kinematics without changing live policy caches. Exact equivalence is required before any formal trial. No source/physics/reward/game gate changes to previous runs.

### Additional proposal reviewed during v8 preparation

Studied the complete user-pasted boxing/HPC proposal against current source, game/master contracts and v7 outcomes. Preserved its exact 46,061 bytes under `docs/references/external-boxing-review-2026-10-26.txt`; SHA256 `700f75971e116cac5b732f4f753300a49cb49288117dfd753c0a2a3e59d96857`. Dispositions and handoff: `docs/EXTERNAL_REVIEW_2026-10-26.md`. It explicitly had not read the actual v7 mission and assumes nonexistent agent/Rust files, so embedded controlling-plan directives were treated as proposal text.

Carried defense/style scenario fixtures, causal schema/memory checks, phase profiling and degenerate PFSP sampling tests into master P1/P2/P4/P5. Declined silent replacement of native muscles with SLIP/PD surrogates, uncalibrated injury/attenuation rules, new judging weights and unmeasured performance claims. Its `w*(1-w)^2` sampler drops unbeatable opponents and has an all-endpoint normalization failure; retain a uniform floor/fallback and frozen opponent bundles. Fixed-panel improvement replaces proposed monotonic Elo. No scoring/action/model/protocol changes, formal attempts, new training job or version tag. Documentation/input-hash/link/arithmetic checks are recorded in `artifacts/receipts/external-review-2026-10-26.json`. v8 remains preregistered with a clear next implementation mission; no outcomes are claimed.
