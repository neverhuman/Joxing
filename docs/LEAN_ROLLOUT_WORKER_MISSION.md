# H28: qualify a Torch-free rollout command worker before more native RL

Status: worker-ready NEXT mission. NOTimplemented, machine protocol pending,
0formal outcomes. One integration owner, no agents or other chats. Primary after
[H27negative result](NATIVE_LEARNER_CONTEXT_PARITY_RESULTS.md); H26Mac-local replay
remains an independent secondary mission. Do not rerun unchanged H27.

## Objective, limits and dependencies

Question: can the exact SB3 2.7.1 Gymnasium pipe command contract run in a child
without importing Torch/SB3, preserving every response/reset/RNG/termination and
exception behavior relevant to Joxing? H27workers retain898.9–921.9MB RSS AFTER
native graph release and the third retirement exceeds100ms. Saved worker maps
contain libtorch. Imported libraries are a plausible removable cost; their causal
share of the failed hold is UNKNOWN. First prove import isolation and command
parity using deterministic fake environments; no native simulation or learner.

Non-goals: relaxing100ms/2s/RSS gates, skipping worker exit, os._exit/madvise tricks,
new scoring/physics/muscles, actual training/native speedup, eight-world adoption,
rewriting SB3 in third_party, changing default start method or legacy behavior.
No reuse of H27failed prefix as a qualified reference. No source-generation from
untrusted proposals. Current Linux locked runtime remains the authority.

Proposed NEW finite source/IPC study, register before outcomes:300soriginal host
clock including admission/setup/tests/four probes/audit;180ssetup,30scontrols,
30s/probe,30saudit reserved by task stop270s;0physics/actors/inference/updates,
16MiBfixtures/1MiBreceipts,1fake-worker at a time/thread1,<=12owned processes,
7GiBplanned/8GiBhard,100ms probe/gap/actualpositive retirement, WHOLE2sclose,
1MiBprivateIPC, existing host16GiBavailable/10GiBdisk and exact fixed PSI policy.
Keep H15old and H27new sunk costs/negative outcomes alongside this distinct study;
no cap extension or deletion. No native launch under H28even if controls pass.

Read first in order: AGENTS.md,/Users/bentaylor/.codex/RTK.md,README.md,
docs/GAME.md,MISSION.md,STATUS.md,IMPLEMENTATION.md,research/README.md,H27.md,H28.md,
NATIVE_LEARNER_CONTEXT_PARITY_RESULTS.md and artifacts/native-learner-context-01,
then src/joxing/{training_context,native_context,process_exit_probe,
deferred_qualification,wave,stance_train,live_resource_release,serial_retirement}.py,
scripts/{qualify_deferred_trial,qualify_native_learner_context,
audit_native_learner_context,host_admission,trial_rss_watchdog,receipt_encoding}.py.
Read actual locked Linux runtime files (do not assume a newer online version):
experiments/v0.0.11/runtime/.venv/lib/python3.12/site-packages/stable_baselines3/
common/{vec_env/subproc_vec_env.py,vec_env/base_vec_env.py,vec_env/patch_gym.py,
env_util.py,monitor.py}. Persist whole-file and relevant AST/body hashes.

## Ownership and exact implementation order

Forbidden: third_party, previous protocols/results/frozen remote checkouts,
referee/scene/model/rules, wave/deferred learner bodies and legacy defaults,
watcher/terminal-reader thresholds. No concurrent writers; do not edit executing
source. New opt-in files only, except explicit schema dispatch in the supervisor
if necessary; add exact new files to the public manifest before formal tests.

1. Create `src/joxing/lean_rollout_worker.py`: stdlib-only entry and top-level
   `lean_pipe_worker(remote,parent_remote,env_fn_wrapper)`, plus a small
   `LeanCloudpickleWrapper` (same state transport as pinned SB3). Heavy modules
   must not appear through annotations/unpickling/fixture module imports.
   Delayed Gymnasium import is allowed; Torch, SB3, MuJoCo forbidden in candidate.
   The SB3 parent may import Torch; actual child's sys.modules AND raw /proc/maps
   must independently prove its absence. A keyword or wrapper class name is not
   evidence. Native-factory integration is explicitly later work.
2. Preserve pinned `_worker` loop commands and semantics exactly, including:
   step five-tuple->VecEnv done; TimeLimit.truncated = truncated and NOTterminated;
   terminal_observation BEFOREautoreset; reset_info preservation on later steps;
   seed/options one-shot; render/get_spaces; env_method args/kwargs;
   get_attr/has_attr with AttributeError; set_attr response; is_wrapped traversal;
   close ordering, EOF/KeyboardInterrupt and unknown-command exception behavior.
   Support Gymnasium Env only in this named opt-in profile; reject legacy Gym
   explicitly instead of silently altering its `_patch_env` behavior. Compare
   normalized AST of unchanged loop and recorded helper substitutions; retain
   SB3license/attribution for copied source. No global package shims/sys.modules
   impersonation or import of `stable_baselines3.common.*` in the candidate child.
3. Create `src/joxing/lean_worker_probe.py`: a top-level deterministic FakeEnv
   factory that imports only Gymnasium/NumPy when CALLED, no native constructor.
   Gymnasium wrappers expose deterministic state and dedicated np.random.Generator;
   bound observations include float32 arrays, nested info and signed zero. Use
   fixed registered actions/options, same seed per pair. No neural actor or RNG
   consumption while observing hashes. Two fresh paired repeats, four rows:
   sb3A/leanA/sb3B/leanB, seeds1097/1098 respectively; fresh empty-preload forkserver
   prepared BEFOREheavy parent imports. Original target `_worker` plus original
   CloudpickleWrapper are the baseline; candidate uses both lean equivalents.
4. Fake episode cases within EACHrow: ordinary step, terminal-only,
   truncated-only, both flags, post-autoreset step, explicit seed/reset/options,
   dict observation/info mutation isolation and array dtype/shape/signed-zero.
   Exercise every pipe command above. Serialize entire ordered command/response
   stream and all fake state/RNG before/after, exclude only declared wall/PID/path
   envelope fields. In fake raw semantic comparison **no tolerance/exclusions**.
   Separate unit controls cover EOF/unknown/error/partial startup; a formal worker
   guard fault stops first and retains unrun rows, never treats a kill as success.
5. Use existing NativeContextProof-style clean server/proc witness WITHOUTnative
   constructors; original private-RAM monitor/copy/serial shutdown. Record actual
   parent/server/worker PID/start/ancestry/thread/mappings, sys.modules digest,
   readiness/events/finalizers, whole exit and positive terminal-mm retirement.
   Guard all MuJoCo step APIs if module accidentally loads; candidate must reject
   import before any step. No os._exit or suppressed finalizers. Record all
   worker RSS measurements before/aftercommands and reference/candidate peak; do
   not compare parent/RAM scopes as worker memory. Baseline must exhibit Torch
   and candidate none. Require candidate steady worker RSS<=75% paired baseline
   and strict command/RNG parity for BOTHpairs, all4valid/100ms/2s/zero work. This
   qualifies IPC/import footprint ONLY, not native state/hold/learning parity.
6. Tests `tests/test_lean_rollout_worker.py` and `test_lean_worker_probe.py`:
   cold imports; pickle/import graph; real fresh forkserver fake pipe controls;
   all commands/bothflags/autoreset/reset_info; wrapper nesting/has_attr/errors;
   private info/snapshot ownership; tiny reward/action/RNG/signed-zero changes;
   corrupted/missing stream and unexpected heavy import; all4denominator;
   pressure/receipt/clock first-limit and owned-only cleanup. Include all315H27
   source nodes unchanged; collect/freeze exact added Linux node IDs. Mac skips
   remain local only. Keep fake source checks distinct from formal probe outcomes.
7. Create `scripts/qualify_lean_worker_probe.py`, independent saved-evidence
   `audit_lean_worker_probe.py` and protocol
   experiments/v0.0.11/lean-rollout-worker-protocol.json. Preserve original host
   epoch through setup/task/audit (no rebase); actual full byte codec/headroom
   preflight BEFOREprobe. Physics ledger0; no environment/policy/native creation
   in auditor. Verify entire terminal/bootstrap/JUnit/source/runtime/payload,
   raw maps/import evidence, all attempted/valid/invalid/not_run, exact pair
   equality, memory scopes and every PIDexit. Audit faults block qualification.
8. Update research/H28.md/index, README/status/book; commit/push source/protocol
   BEFOREformal target. Anonymous verify on currently admitted host. One bounded
   packet, no overlap or duplicate. On completion safe regular-payload transfer
   with retained symlink metadata/full hashes. Small text summaries only in Git.

## Validation, stop and handoff

From /Users/bentaylor/Documents/ChatGPT/JOXING:
`rtk proxy uv tool run --offline ruff check src/joxing/lean_rollout_worker.py src/joxing/lean_worker_probe.py scripts/qualify_lean_worker_probe.py scripts/audit_lean_worker_probe.py tests/test_lean_rollout_worker.py tests/test_lean_worker_probe.py`
`rtk proxy env PYTHONPATH=src:.:scripts .venv/bin/python -m pytest -q tests/test_lean_rollout_worker.py tests/test_lean_worker_probe.py`
Then all315old packet modules plus additions; `--collect-only -q` must match the
registered manifest exactly. Commands refer to files to IMPLEMENT, not existing
runnable tools. New supervisor CLI requires packet/source/protocol SHA and original
host epoch as H27. Keep exact commands/test-JUnit/source/installed-runtime/AST/
process/model=none/normalizer=none/RNG/semantic hashes, control failures and all
outcomes in receipts. Source fails: no probe. Probe fails: no next probe/native.
Audit fails: no qualification. Public source unavailable: no formal execution.

Completion: all4audited exactfake profiles and measured import/RSS gates pass OR
an honest complete stopped report. On PASS only, define a separate worker-ready
native integration packet. It will need an opt-in parent SubprocVecEnv constructor
using lean target/wrapper, and a child-local factory using exact original
TrainingContract/FaultRecorder plus pinned Monitor/ResultsWriter WITHOUTimporting
SB3. Keep original wave/make_worker untouched: new factory can compose original
light classes, with locally scoped factory injection into instrumentation. The
SB3parent methods and learner remain original. Validate Monitor CSV/episode/
reset semantics (only episode.t excluded by existing rule), source AST/helper
hashes/license, all700muscles/state/reset/contact/evaluation and dynamic4world
release, then NEWfresh four-slot actual SACparity. A lean fake worker does not
validate that native integration. No default flip, native training sweep, source
snapshot rewrite,8worldconcurrency or self-play until those gates pass.

Final handoff: public commit/protocol, source tests/full case table, exact module
and measured perworkerRSS evidence, first fault if any, all PIDs/exits/rawhashes,
old/new sunk budgets and next smallest question. Link future results here.
