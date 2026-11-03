# H31 admission01 deferred; fresh same-limit admission02 registered

Public607cfe9/05845df8 owner exited after3.713s before source/native stages:
local load8.96875>unchanged7.0limit (14CPUs×.5), memory21.65GBavailable.
0sourcecases/0physics/0nativeattempts; bothseeds/all16evals not_run.
Owner/admission/exit retained in artifacts/v0.0.11/native-pair-motor-pilot-
admission-01. Oldroot/clocksealed; no gatechange or unrelated jobs touched.

Separate admission02:experiments/v0.0.11/native-pair-motor-pilot-02.json,
sha3e25e0291a68a0cf60dd02b2edec0a4cf110008a1a40f85a7a5319c14374dcf2. Only root/output/prior-deferral
provenance differ:834sourcehashes/99cases/2×1024steps/allcriteria/caps unchanged.
Freshsame-limitcapacitymustpass before native launch. xbabe4has512GiBmemory/
lowloadbutreadonlyexistingruntime lacksTorch/SB3/rendering andteacherassets;
no package changes. Prefer recovered eligible localprofile; retainspareMacport.

# H31 two-seed native motor pilot registered — source ready, formal task pending

Implemented full policy/optimizer/RNG/native/VecEnv boundary checkpoints,
bounded serialization (13MiBmodel/1MiBcontinuation), fixed evaluation1msmetrics,
alltraining+all16evaluation independent replay, typed write guard and bounded
Macowner with capturedlogs/durable entered counters.99fresh development source
checks pass UNDERthe actual write guard; prior97/99SB3file-adapter failure retained
and repaired using its public singledispatch extension. Native fixture initial/
postupdate next-action/hold restoration and evaluation traces match exactly.

Frozen prospective protocol:experiments/v0.0.11/native-pair-motor-pilot-01.json,
sha05845df8010d4c8aca881e3f584f1e5905abe8b323667e0140a6e66bec50a2a5.
Two seeds761/983×1024JOINTsteps=4096actor samples/16rollout updates/256optimizer
minibatch steps, four SAME3s resets before/after eachseed, first fixedwholeclips.
Shared1400muscle world/frozen released motor/causal4frame own+opponent features;
motor reward only. Competition remains false. Alloriginalproposed behavioral
criteria frozen unchanged:perseed/side targeterrorimprove>=.05m, survivaldelta
>=−.1s, fullguard→reach→returncycle on>=2/4starts and3ssurvival on>=3/4starts.
Measure40mscontinuous1msdistance<=.15m/facecos>=.5perphase; nocycleacrossschedule.

Newdedicated~/joxing/native-pair-motor-pilot-20261103-01. Original1800swhole incl
setup60/source120/native600/render4×30/audit300(firstlimit); one world/thread1/
8GiB/12process/1GiBraw/16MiBtext/200000physics. Max89197primary+88960audit+
2048source reserve=180205. Audit300s prospectively covers88960replayintervals,
versus old prototype640; no old failed budget is extended. Rawworstforecast
823897216B/text15204352B with240MiBotherraw allocation(source64/eval100/media64/
fault12), not a hard-cap increase. Freshcapacity/source/input/runtime checks
and publicsource verification precede launch. Allresults/falls/invalid/not_run
must be retained; changedactor/return/videoalone doesnotpassbehavior gates.

Next: publish frozen source/protocol, prepare fresh dedicatedroot and start once.
On healthyjob verify durable physicscounter/checkpoint/chunk growth and ownedPIDs;
no sourcechanges orduplicates. Oncompletion read independent-audit+ownerstatus,
audit everyseed/eval/fixedclip/exit/resource/sourcehash, recordnegativeoutcomes,
push smallreceipts/README/docs/H31book. No tag/competitive claim. Wholev11pending.

# H31 streaming source checkpoint — next is the bounded behavioral pilot

Implemented: `src/joxing/native_pair_motor_evidence.py` (`CausalChunkWriter`,
`EvidenceBudget`, `read_chunk`, `training_storage_forecast`), opt-in VecEnv raw
history capture, and `StreamingNativeTransitions` in native_pair_motor_train.py.
55paired controls+39regressions pass. Native engine/synthetic teacher fixture:
16jointsteps/2PPOupdates/five resets, all original arrays bit-exact and behavior
recomputation0error. Candidate tapes+manifests are43.3835%smaller. No new actual
released-controller training or boxing result. See NATIVE_PAIR_STREAMING_SOURCE_RESULTS.md
and artifacts/v0.0.11/native-pair-streaming-source-01/summary.json first.

**Current ready scope:2seeds×1024JOINTsteps**, not the original2048proposal.
Worst-case resets+128row/full-checkpoint cadence+128MiBotherraw/8MiBothertext
forecast706456704Braw/15204352Btext fits unchanged1GiB/16MiB.2×2048requires
1249335552B/22020096B and is declined BEFOREformal registration. No gates are
weakened. Full writer enforcement is still required; forecast alone is not launch
permission. All physical/muscle/reward/rule defaults and proposed behavior gates
below remain unchanged. Same H31; next unused research IDH32.

## Remaining ordered execution packet

One integration owner, no agents or concurrent writers. Read AGENTS/RTK,
MISSION/STATUS/IMPLEMENTATION/experiment book; this result and researchH31/index;
new evidence module and tests, native paired train/vec/core, original source2101233,
prototype result and prior mission below. Edit only opt-in native motor pilot
source/scripts/tests plus current docs/research/receipts. No third_party, frozen
prototype root/protocol/results, scene/referee/model schemas or other projects.

1. Add `PilotAuditedPPO` and `run_native_pilot()` in native_pair_motor_train.py;
   keep old smoke API8/16/32/64bounds unchanged. Freeze new128joint-step rollout,
   batch64/4epochs/lr3e-4/gamma.99/64x64/zero mean/logstd−3 before outcomes.
   Implement ONEfull checkpoint per rollout boundary+initial, not duplicated
   before/after files. Save via bounded in-memory serialization, enforce13MiB
   model and1MiBcontinuation BEFOREdisk writes. Save optimizer, all RNG, native
   snapshot, vec nextseed/returns/length/resetinfos and model lastobs/episode masks.
   Initial bundle must precede first rollout; post-update bundle supplies next
   rollout's behavior. Record real optimizer minibatch steps separately from
   epoch counts. No asynchronous observation/teacher resampling on restore.
2. Bind StreamingNativeTransitions to exact world/behavior checkpoint identity,
   raw state dimension and3072Browcap. Publish complete chunk/hash chain. Test
   initial and post-update restore with an actual next action/hold, bit-exact
   observations/teacher/commands/native state/PPOlikelihoods/RNG. Existing source
  55+39controls stay green; collect exact fixture physics reserves prospectively.
3. Add bounded fixed evaluation writer and independent auditor in NEWscripts,
   e.g. run_native_pair_pilot.py/audit_native_pair_pilot.py. Use the same four
   resets71000/71016/71032/71048 before/after per training seed761/983,3s each;
   retain every fall/zero-point result. Record all actions and complete initial
   continuation so ALL16evals independently replay exactly. Record real glove
   target distances/phase durations/cycles/face and censored survival for the
   original proposed behavior gates. Render first entire before/after eval per
   seed with existing renderer (4clips), no flattering selection. Before policy
   zero residual equals the initial deterministic learner mean; never shift starts.
4. Implement stdlib bounded owner `scripts/qualify_native_pair_pilot.py` and new
   `experiments/v0.0.11/native-pair-motor-pilot-01.json`. Account all serialization,
   checkpoints/chunks/eval/source/media/fault bytes and typed text online, including
   logs and temporary replacements. Forecast02's128MiB/8MiB other allowance is
   a reservation to allocate, not an implemented guard.200000physics proposed cap:
   40960training +48000eval +48000independent eval replay leaves63040 for exact
   source/prep/midhold237/other replay. If auditing ALLtraining too, add40960 and
   leave22080. Count before each entered step; no render physics. Retain both
   attempted/valid/invalid/not_run seed rows from start. New1800sepoch and stage
   bounds below,8GiB/oneworld/thread1; stop at first fault/limit, no retries in an
   old root or budget reset. A valid fall is training data; numerical/ownership/
   assistance/parity faults are invalid and stop. No competitive learning.
5. Freeze exact numerical protocol, source hashes, model/teacher/group/normalizer/
   reference/rule/timing and units before outcomes; publish/sourcecheckfirst.
   Recheck four hosts and chosen profile admission; use a NEWdedicated~/joxing
   root and readonly runtime/inputs. Start only once all prerequisites pass.
   No host service/package changes. Audit all owned PIDs/start/exit/progress/raw
   and exact evaluations, then publish small summaries and fixed clips in Git;
   retain checkpoints/tapes outsideGit. No tag or skill claim just for updates.

Completion: both bounded registered seeds conclude with honest per-side,
per-seed improvement/failure under original behavioral criteria, independently
replayable evaluations and allfaults/denominators/resources retained. If an
implementation or prerequisite fails, stop dependent stages, preserve its receipt,
leave this explicit next action. No pilot is active at this checkpoint.

---
Original prospective2048step proposal below retained for research history;
the current1024step scope and remaining packet above supersede its unrun size.

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
