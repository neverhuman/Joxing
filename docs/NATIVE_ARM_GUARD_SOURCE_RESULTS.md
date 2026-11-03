# H35 guard learner: development source qualification

Updated2026-11-03. **236checks pass in111.574s.** This implements the new
arm-only learning environment,46action paired Vec adapter, full-observation
evidence and checkpoint restoration. It is source qualification using the real
native engine and an explicitly synthetic frozen teacher. The released pretrained
controller study, parallel qualification and guard skill outcomes remain not_run.

## Delivered and tested

- All700muscles perbody remain. Zero46actions reproduce H34walking engine/contact/
  teacher bytes including signed zero. Nonzero actions modify only122arm muscles;
 578protected raw teacher commands perbody stay bit-exact. Excitation happens once.
- Fixed physical reset phase71000 is independent of optimizer/logical seeds.
  Run/logical-seed/serial nonce creates unique episode ownership before histories
  and referee construction. An old snapshot cannot enter a new logical episode.
- Actor input retains four causal own/opponent movement frames and ages. The
  final9values use the current guard goal throughout; world-FK distances agree
  within2e-6m float32 sensor precision. Cached goals remain causal during a hold.
- New linear-distance/head-height/alive/effort reward, measured every1ms, is
  separate from the legacy task reward and sporting points. Continuous guard
  qualification resets on distance, face, pelvis tilt or head-height violations.
- Both46actions are accepted before either body integrates. Any physical ending
  resets both Vec slots; genuine final observations/states/masks survive reset.
- At203ms the remaining17held intervals and the NEXT20msdecision replay exactly
  with nonzero actions. Restore itself makes no teacher call or physics entry.
  Re-signed bad goal/counter/current-metric/hold/nonce/head-anchor snapshots fail
  before mutating the live evaluated world.
- Real PPO fixture:16joint transitions/32actor samples, two rollout updates,
  eight optimizer minibatches, five60msmotor episodes. All16training rows replay
  exactly; behavior values/logprobs have maximum error0.0. Raw Gaussian46actions
  and clipped46commands remain separate. This is a16×16source-fixture network,
  not the planned64×64pretrained experiment.
- Three full checkpoints retain actor/critic/Adam, native/history/guard/Vec state
  and Torch/NumPy/Python RNG. Restored next actions, values, likelihoods, random
  draws and next physical states match. Physical charges never rewind. Corrupt
  external checkpoint/chunk bytes are rejected before engine mutation.

New source: `native_arm_guard.py`, `native_arm_guard_vec.py`,
`native_arm_guard_evidence.py`, `native_arm_guard_pilot.py`; test module
`tests/test_native_arm_guard.py`. No legacy physics/rules/defaults or third_party
files changed. Full trace stores every1ms guard metric/ordered contact hash,
both full native endpoints, both actor observations, actual frozen teacher inputs/
outputs, composed700commands, raw/applied46actions, behavior statistics and epochs.
Chunks≤64rows, ZIP_STORED≤64MiB, typed schemas and external chained receipts.

## Attempts, faults and accounting

Allfour development attempts are retained, separate from formal study credit:

| Attempt | Result | Static native fixture entries |
|---|---|---:|
|01|19/19core controls pass|746|
|02|20pass/1fail; evidence reader wrongly assumed a plain64hex game ID|786|
|03|21/21pass after accepting the inherited `motor-`+32hex format|1486|
|04|236/236full suite pass:163prior+47upper controls+26new|5374|

Total8392static source fixture entries across allattempts, including both oracle
and replay worlds. Fullsuite5374=3348prior+540upper-task+1486new. These counts
are not boxing games or formal learning outcomes. Failed02stops after its first
20msPPOtransition, before any optimizer update; its initial checkpoint/prefix stay
outsideGit. The successful fixture intentionally damages one checkpoint and one
manifest AFTER exact replay for negative controls; those mutated files are not
claimed to be intact scientific run archives. No physical gate was eased.
The source-file hashes identify final04; earlier uncommitted development source
preimages were not separately archived. Their JUnit faults/counts remain, with
no frozen-source formal-registration claim.
One read-only process query returned1because no matching job existed; a source
patch context mismatch made no filesystem change. Neither entered physics.

The full suite command is recorded exactly, including all15modules and thread1
environment, in the command receipt. Reproduce with:

```sh
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_native_arm_guard.py -q
rtk proxy uvx --offline ruff check src/joxing/native_arm_guard.py src/joxing/native_arm_guard_vec.py src/joxing/native_arm_guard_evidence.py src/joxing/native_arm_guard_pilot.py tests/test_native_arm_guard.py
```

## Finite study readiness and next work

Full-training allocation forecast perseed:32×64row chunks,15,552,838bytes each;
training+manifest bound502,933,696B;17checkpoint pairs≤249,561,088B. With80MiB
compact eval/binary snapshots,32MiBwhole media and32MiBtemporary/fault reserves,
total903,489,728B<896MiB(939,524,096B). Text6MiB includes65,536Bunspent fault
reserve; other ordinary text is≤983,040B. This is a prospective allocation, not
an implemented whole-study guarantee. Reusing the full training trace for ALL
evals would exceed the shard allocation. Implement a compact replayable eval tape;
do not drop evaluation rows or raise caps to fit it.

Fresh four-host inventory found no owned jobs. xbabe1had14.56GBavailable and
17.72%I/Ofull; xbabe2I/Ofull8.14%; xbabe3I/Ofull51.85%/memoryfull10.87%.
xbabe4load4.58/32CPUs is the current candidate. No capacity was reserved; fresh
two-sample admission is required before the proposed two workers launch.
A later read-only refresh timed out on xbabe3after30s; its helper did not publish
a combined02snapshot. The fault is retained separately; fresh per-host bounded
03inventory records failures individually. No launch or empty-job inference comes
from an unavailable host. In03xbabe1/2/4show no owned jobs; xbabe3ps times out
after5s and remains unavailable. xbabe4load4.54/32CPU,93%memoryfree and38.14GBdisk.

Continue the ordered remaining implementation in
[the H35mission](NATIVE_ARM_GUARD_RL_MISSION.md): compact fixed evaluation,
independent all-row replay/reward/likelihood audit, two-worker startup parity,
bounded coordinator, exact shape/byte/resource protocol, public registration,
then two2048joint-step released-controller seeds. The primary≥.05munseen gain,
allthree unseen3ssurvival and separate40msguard gates stay unchanged. Bothformal
seeds/all16evals currently not_run. No guard, legal punch, points gain, self-play
win or v0.0.11completion is claimed.

Receipts: [allattempts/source hashes/fixture proof](../artifacts/v0.0.11/development/h35-development-01.json),
[full command](../artifacts/v0.0.11/development/h35-source-04-command.json),
[full JUnit](../artifacts/v0.0.11/development/h35-source-04.xml),
[failed02JUnit](../artifacts/v0.0.11/development/h35-source-02.xml),
[storage allocation](../artifacts/v0.0.11/development/h35-storage-forecast-01.json),
[first four-host capacity](../artifacts/v0.0.11/development/h35-host-capacity-01.json),
[refresh timeout](../artifacts/v0.0.11/development/h35-inventory-fault-02.json),
[bounded post-source inventory](../artifacts/v0.0.11/development/h35-host-capacity-03.json).
