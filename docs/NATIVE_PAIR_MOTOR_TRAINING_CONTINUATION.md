# H31 next: two-seed native guard/reach training, not another plumbing benchmark

Status: worker-ready IMPLEMENTATIONcontinuation; not implemented/registered/run.
H31small Mac prototype passes with actual released motor,32joint transitions/
64actor samples/4PPOrollout updates/8epochs; all32engine transitions replay exactly,
all700decoder outputs reconstruct, four behavior value/logprob checks have0error.
Fixed1s before/after clips are complete, upright,0–0; gloves mostly low/body contact,
no convincing punch/guard sequence or skill improvement demonstrated. This is the
measured behavioral gap. H30stays secondary. Same H31question, next unused ID:H32.

## Objective, boundaries and read-first

Run meaningful native motor learning on two seeds with3s fixed evaluations and
explicit guard→reach→return metrics. Preserve opposed1400muscle physics, reference
phase,138bounded residual groups/actor, frozen teacher, causal4frame observations,
simultaneous paired PPOslots, dense MOTORobjective and sporting weights. No root
assistance/pruning/edited muscle dynamics, fake game counts, competitive promotion,
threshold tuning after outcomes or old prototype clock/budget reuse. No agents.

Read AGENTS/RTK/current MISSION/STATUS/IMPLEMENTATION; researchH31 and index;
NATIVE_PAIR_RL_PILOT_MISSION.md; NATIVE_PAIR_MAC_PROTOTYPE_RESULTS.md; prototype
protocol/source2101233 and artifacts/v0.0.11/native-pair-mac-prototype-01/{summary,
independent-audit,post-task-audit,native-result,native-native-replay}.json;
src/joxing/native_pair_motor_{rl,vec,train}.py and matching tests; stored writer/
history/observations/referee/source-limit contracts. Raw130MBdata/weights are at
/Users/bentaylor/joxing/native-pair-mac-prototype-20261102-01/runs/v0.0.11/native-pair-mac-prototype-01;
its exact53file manifest is in Git. Never modify this frozen root or receipts.

One owner. New opt-in src/joxing/native_pair_motor_evidence.py and
run_native_pilot() in native_pair_motor_train.py; tests/test_native_pair_motor_evidence.py;
new protocol/entry/auditor and compact docs/results. Keep old smoke API's8/16/32/64
bounds; do not change its archived protocol to mean2048. No edits third_party,
old roots, scene/referee schemas, old worker/watchdog/physical defaults or other
projects. Native released checkpoint hashes/defaults remain unchanged.

## Ordered implementation and code-shaped guidance

1. **Fix scaling of evidence BEFOREtraining:** current AuditedPPO saves duplicate
   before/after checkpoints every8jointsteps and NativeTransitions holds complete
   overlapping histories/engine states. Simply raising joint_steps would exhaust
   the raw cap. Keep smoke default; add a named pilot profile with128joint steps/
   rollout, batch64,4epochs, same3e-4/.99 and zero initial residual mean/logstd−3.
   Freeze these prospectively, not based on observed scores. Record optimizer
   minibatch steps separately from SB3's epoch counter.
2. Stream complete per-step scalar outcomes/digest chain; store one FULLcheckpoint
   at each128step boundary, not duplicate adjacent before/after files. Keep actor/
   optimizer, Torch/NumPy/Python RNG, external VecEnv next seed/episode accumulators,
   last_obs/masks, native engine/reference/history/held state in continuation.
   Checkpoint and row provenance must identify the exact behavior bundle.
3. Add lossless causal-frame/chunk writer. Four-history windows overlap: store each
   actual newest1970f32actor feature frame, game/corner/tick/reset/valid/ages/goal,
   raw+applied138actions, values/logprobs, teacher3601inputs/700outputs, normalized
   commands and integration endpoints. Deduplicate adjacent engine_before against
   prior engine_after, store a genuine new initial engine on reset. Reconstruct
   all7901actor vectors through actor_vector(), bit-exact, BEFOREaccepting chunk.
   No lossy tensor quantization, field omissions, future values or aliasing.
4. Use qualified ZIP_STORED chunks <=64MiB each;128rows project well below that.
   Exact preflight MUSTinclude all chunk payloads+headers+checkpoints+RNG+logs+
   source fixtures+full evaluations/replay/render/fault reserve. Original prototype
   native raw was130172115B for only32jointsteps, so no unbounded old recorder.
   Target whole raw<=1GiB; stop before first row if worst-case forecast does not
   fit. Keep all episodes/valid/invalid/not_run rows in Git as compressed small
   summaries; never checkpoints/large tapes in research or Git.
5. Test chunk boundary/repeated observations/reset/tick/side/order/signedzero/
   dtype/shape/CRC/truncated file/foreign snapshot, behavior action/raw-clipping
   correspondence and preallocated receipt/raw failure. Test source/replay203+17
   no extra teacher/actor decision and PPOreset/true final-observation masks.
   Run all prototype22controls plus fresh relevant native/history/contact/decoder/
   support/replay regression. Collect exact IDs and count/reserve ALLsource-fixture
   physics for this protocol; do not claim the source stage is zero-step. A broad
   Linux or release gate remains separately unqualified until it actually passes.
6. Reuse working explicit local Mac prototype runtime as one eligible profile;
   another node, including xbabe4 or Linux, needs exact source/runtime/model/feature/
   local replay prerequisites. Do useful native code/source work while shared hosts
   are busy. Fresh four-host inventory and host-specific admission before launch;
   dedicated new ~/joxing/native-pair-motor-pilot-YYYYMMDD-01, readonly original
   teacher inputs/venv, no overlapping writer. Begin with ONEpersistent world/
   thread1/<=8GiB before considering parallel seed jobs. No infrastructure-only
   detour without measured native need.
7. New proposed finite pilot:1800soriginalepoch, source<=900/train<=600/render<=120/
   audit<=120, first limit;200000total entered/reserved physics,1GiBraw/16MiBsmall
   receipts. Reconcile exact stage/physics/source reserves prospectively. Seeds
   proposal761/983, each2048JOINTtransitions =8192total actor samples and81920
   training physics. Do not count the two logical slots as independent games.
8. Fixed held-out resets proposal71000/71016/71032/71048 per trained seed, SAME
   starts before/after, horizon3000ticks.16evaluations at most48000physics;
   independent exact replay of ALL16evals at most48000more. Source/native203+17
   and all prep/render/extra replay entries must fit remaining22080physics;
   if not, reduce scope BEFOREregistration, never erase an entered attempt.
   Render first entire before/after eval in fixed ordering per seed, including
   falls/zero points. One-second survival does not establish3s balance or a bout.
9. Freeze motor behavior gates BEFOREoutcomes, e.g. mean guard/reach target error
   improvement>=.05m with no paired survival decrease>.1s; report per-side/per-seed
   and censored3s completion. Define a reach-return cycle using real glove distance
   <=.15m/face cosine>=.5 with40ms persistence in each scheduled target phase;
   propose both actors pass a full cycle on>=2/4fixed starts per seed and survive
   3s on>=3/4. These are proposed MOTORcriteria, not sporting points or silently
   chosen winners. Validate achievability/units in source, then preregister final
   thresholds without looking at candidate outcomes. Valid falling episodes are
   motor-learning data; solver/assistance/parity/resource faults are invalid.
10. Independently verify each native source/actor/normalizer/reference/group/model/
    rule/timing identity, complete PPO masks/transitions/update counts/NNhashes,
    actual teacher unchanged, all held continuation/counter bounds, every fixed
    evaluation and replay/clip, all PIDs/exit/resource/source-clean state. Preserve
    failures and ties, no cherry-picked weights or clips. Push README/docs/book/
    H31/index and small result manifests. No tag/competition claim unless its
    own gates pass; whole v0.0.11 stays in progress.

## Completion, interpretation and next action

Run exactly the two registered native training attempts or retain explicit
failed/not_run. Report actual learning AND honest performance improvement/failure.
A changed actor alone qualified the tiny prototype, not this behavioral target.
Record action saturation/frozen-teacher dominance if reach/guard does not improve;
that can motivate a separately named motor/reference/decoder ablation, never an
unchanged huge sweep or loosening face/point predicates. Stable reach then physical
legal strikes precede historical-opponent competitive bouts. Preserve rules/
foul/fall/replay gates, including known unresolved defects. Target1k→10k→100kvalid
boxing games only after those prerequisites; measured full-workload throughput
and valid games/s, not physics steps or motor episode counts, control scaling.

Handoff: exact public source/protocol before outcomes, ownership/root/commands,
complete allseed/side/episode counts and raw manifests, actor/RNG/continuation
hashes, fixed clips, faults/remaining budgets and smallest measured next question.
Leave active bounded native work or this explicit ready mission; H30is not primary.
