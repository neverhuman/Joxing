# Ready offline native tape writer qualification

Status: **concluded; isolated candidate qualified**. Read
[NATIVE_TAPE_STORAGE_RESULTS.md](NATIVE_TAPE_STORAGE_RESULTS.md) for both exact
rows, 28.72× reference save speed, larger archives, the original size failure,
all attempts, the test-fixture CI correction and scope limits. No collector
integration or new physics. The original preregistered packet follows.

Objective: remove unnecessary LZMA latency from native evidence persistence
while retaining every logical bit, complete proof accounting and bounded output.
This is a new finite offline task, not an extension of the exhausted native
900-second budget. All prior failures, incomplete signed contacts and unexecuted
replays remain. No physics, optimizer or controller/model compilation is allowed.

Read AGENTS.md, /Users/bentaylor/.codex/RTK.md, MISSION.md, STATUS.md,
NATIVE_SOURCE_PARTIAL_RESULTS.md and NATIVE_BALANCE_STRIKE_NEXT_PACKET.md first.
Then read experiments/v0.0.11/native-tape-storage-protocol.json,
src/joxing/native_tape_storage.py, native_tape_qualification.py and the two
corresponding test files. One owner; no agents/concurrent writers. Forbidden:
third_party, old frozen roots/raw, scene/referee/Grokbot schemas and existing
collector behavior. The candidate is opt-in and is not used by any collector.

## Source and implementation

The deterministic ZIP_STORED writer preflights exact archive size and checks
raw credits before opening a temporary. It preserves C-order payload bits,
including signed zero and NaN payloads, across scalars, endian variants and
strided/Fortran inputs. A completed file appears only after finalization/fsync;
an interrupted write retains its temporary and returns no completion receipt.
The caller owns a unique output directory. Existing final/temporary paths fail.

The candidate reader checks the externally pinned container hash and bounds
the central directory before allocation; NPY shapes, dtype, aggregate size and
payload hashes are checked before construction. Independent NumPy/ZIP reads
check every logical descriptor again. The new tests exercise corruption,
oversized declarations, raw fault reserves, failed writes, incomplete source
recognition, two-row fixture flow and independent saved-audit corruption.
All **24 local tests pass in 0.29 seconds**; Ruff passes. These are development
checks, not real-corpus performance or full scientific-source qualification.

## Frozen experimental protocol

Publish source/protocol and anonymously verify exact changed bytes before any
formal measurements. Fresh dedicated xbabe1 root:
/home/ubuntu/joxing/native-tape-storage-20261028. Old corpus roots are read-only.
The protocol authenticates 13 relevant source/runtime/manifest files and both
actual input container sizes/SHA. NumPy 2.4.2/Python 3.12.3; no dependency upgrade.

Two fixed rows, no replacements or padded arrays:

1. Native incomplete prefix: five complete CRC-valid members / 9,119,848 logical
   bytes. Missing signed contacts and original video arrays stay missing.
2. Complete walking reference corpus: all 23 arrays / 41,319,560 logical bytes.
   This is a saved-data compatibility corpus, not a new walking or boxing trial.

For each: write the stored archive; decode with candidate and NumPy readers;
compare every dtype/shape/C-order byte hash. Encode the same original arrays
once using ZIP_LZMA in bounded memory, and independently decode that baseline.
Candidate timing includes planning, hashing, disk write and fsync; memory-only
LZMA timing excludes those disk costs, making the speed comparison conservative.
Retain every measured time/byte/hash and stage entered/completed event.

Gate: both exact rows and separate saved audit; at least 2× encode speedup on
the complete reference corpus; and the registered future native size preflight.
No performance gate on the partial prefix. Qualification only licenses later
integration work, never an unassisted boxing or replay claim.

The future size preflight is arithmetic, not reconstructed missing data. It
uses actual contact count/state width and the complete hypothetical eight-array
layout. Include saved fixed source outputs, archive/receipt overhead, conservative
replay journal/tail, audit/result/supervision/fault reserves and video caps.
Report BOTH variants: original 800×448/2 MiB video allocation (development
arithmetic slightly exceeds 24 MiB), and separately named 640×360/1 MiB variant
already demonstrated by the partial display. The latter is the declared size
gate for a future mission; this does not change old recording criteria/results.
The 41 MB walking reference stored archive does not fit its old 32 MiB proof;
this candidate is not a universal replacement for the existing compressed codec.

## Launch, limits and failure handling

After public freeze, fresh checkout and capacity check, launch:

`python3 scripts/supervise_native_tape_storage.py --protocol-sha256 <canonical SHA> --source-sha <frozen commit>`

The supervisor first runs bounded locked runtime sync (180 seconds), then the
exact 24 targeted Linux tests (20 seconds). A source-test failure leaves both
formal corpus slots not_run. No full unchanged physics suite is repeated for
this isolated offline candidate; integration into a new scientific collector
requires a fresh full source gate before new formal physical outcomes.

Proof 45 seconds, independent audit 15 seconds, combined postgate 60 seconds
with termination allowance held inside that total; 512 MiB process-group RSS,
64 MiB new raw (256 KiB supervision and 2 MiB fault reserves), one numerical
worker/thread, at most 32 MiB ephemeral LZMA output. At least 8 GiB available
host memory/10 GiB free disk before launch. Numerical thread environment is one.
Do not buy compute, change services or stop unrelated processes. Do not launch
a duplicate or edit the executing source.

Source under runs/v0.0.11/native-tape-storage-01; supervision under
runs/v0.0.11/native-tape-storage-supervision-01. Each stage has exact command,
PID/start/group identities, fresh elapsed/RSS/raw progress, exit/fault and
attempt statuses. Negative speed/size outcomes remain complete measurements;
the independent read audit still runs when valid saved rows exist. Corruption,
resource failure or incomplete writes never count as a passing qualification.

On completion independently verify every original/new source/input hash,
logical descriptors, decoded arrays, attempted/not_run rows, exact raw file
names/sizes/SHA and all owned identity exits. Preserve the full raw corpus
outside Git; curate receipts/negative outcomes in artifacts/v0.0.11, update
README/MISSION/STATUS/IMPLEMENTATION/experiment book and push public Git.
No v11 tag and no automatic physical activation. Then implement the zero-step
native face audit and pretrained balance task in the next worker packet.
