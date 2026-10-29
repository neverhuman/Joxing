# Fast exact tape writer qualified offline

The opt-in writer passed both registered saved-data rows, both decoders and
the separate saved-evidence audit. It does not yet replace any collector.
No simulation, model compilation, policy update or new boxing game occurred.
The prior incomplete force tape and unexecuted physical replay remain missing.

| Saved corpus | Logical arrays | ZIP_STORED save | In-memory LZMA save | Encode speedup |
|---|---:|---:|---:|---:|
| Actual complete native prefix |5|0.071510 s|1.271554 s|17.78×|
| Complete walking reference |23|0.364817 s|10.476659 s|28.72×|

The registered reference threshold was 2× and passed unchanged. Candidate
timing includes preflight, hashes, disk write and fsync; LZMA runs in bounded
memory without disk costs. This is a conservative comparison of evidence-save
latency on one host, not physics, optimizer or completed-games throughput.

All 28 arrays retain exact dtype, shape and C-order payload bits in the candidate
reader and independent NumPy/ZIP reader. A separate process reloaded original
corpora and audited every saved descriptor again. An additional local NumPy
audit of the transferred archives also checks all 28 arrays exactly.

The cost is size: native prefix archive 9,121,058 bytes versus 2,011,916 for
LZMA; complete reference archive 41,325,014 versus 19,210,825. ZIP_STORED is useful
where raw credits fit. The reference archive does not fit the old 32 MiB walking
proof, so this candidate is not a universal replacement for the compressed codec.

## Qualification, limits and scope

Frozen public source: `0d467da93b923006ce1cbdafe7fb2b502be6cf36`.
Canonical protocol:
`59c2e77848e28907c195923d70eba2f33025459724ce0404179c222825a09e13`.
Root: `/home/ubuntu/joxing/native-tape-storage-20261028` on xbabe1.
All 15 declared public source/protocol/mission files were anonymously byte-verified
before launch; all 13 relevant source/runtime/manifest files and both original
input containers matched their registered identities before and after execution.
Runtime: Python 3.12.3, NumPy 2.4.2; numerical thread limits one.

The exact **24 targeted Linux tests passed in 0.86 seconds**, followed by two
passed corpus rows and the passed separate audit. Proof wall 15.376661 seconds;
supervised combined postgate 17.512614 seconds against 60. Peak process RSS
168,902,656 bytes against 512 MiB. Independent final raw inventory is
15 files / 50,503,913 bytes against 64 MiB. The supervisor's slightly earlier
raw counter excludes its final stdout growth; use the final inventory for totals.
Every raw filename/size/SHA was verified after transfer, and all seven recorded
PID/start identities exited; no owned process group or root writer remains.

Current Git holds compact results, both archive receipts, events, test logs,
audits, resource/supervision reports and complete raw manifest under
`artifacts/v0.0.11/native-tape-storage-01/`. The two large exact archives and
the complete 15-file raw copy remain in the ignored
`runs/v0.0.11/native-tape-storage-evidence-01/` and their remote source root.
Raw transfer tar SHA256:
`c72c0cb60d4355b5b8d90d62466c637c67bc73f7ffd164ce1b5160ef80b64380`.

## Future native size estimate: retain both outcomes

This is arithmetic from the actual contact count and state width, not fabricated
missing values. The projected eight-array payload is 15,722,456 bytes. Whole
estimates include fixed source JSON/journal, metadata, replay journal/tail,
video, audit/final reports, supervision and fault reserves.

| Separately named output allocation | Whole raw upper |24 MiB gate|
|---|---:|---|
| Original 800×448 / 2 MiB video |25,191,376 bytes|Fails by 25,552 bytes|
| Future 640×360 / 1 MiB video |24,142,800 bytes|Fits; 1,023,024 bytes remain|

The smaller variant was registered before these formal outcomes and already
has an actual 76-frame partial-display example. Its size estimate passes the
new offline protocol's declared gate. It does not change the old recording
criteria, rescue missing contacts or complete a native source/replay proof.
An integrated collector must preflight actual complete outputs and stop if
the estimate or credits fail. No automatic physical launch is authorized by
this qualification result.

## Negative outcomes, monitoring gap and current source distinction

The first publication preflight returned HTTP 404 after a concurrent public
main update rejected the local push. It stopped before checkout or process
launch. Both corpus slots were not_run. Incoming disjoint work was preserved,
merged and pushed; the successful second deployment then verified public bytes
before starting. Across those two deployment attempts: four corpus slots,
two passed and two not_run; the preflight failure remains separately recorded.

The host link became unreachable after launch. The detached bounded supervisor
completed its fixed work independently; no duplicate was started. Connectivity
returned on October 29, and completion, raw evidence and exits were then audited.
No success was inferred while the host was unreachable.

The full-suite CI context also exposed a mistake in the new synthetic test
fixture: it compared its tiny operation with the pytest process's accumulated
peak RSS from earlier model tests. Incoming commit `6e9d9a8` corrects only that
fixture by adding headroom above the pre-existing peak. It is preserved on main;
the 24 focused tests still pass (0.37 seconds locally after merge). Production
guards and the completed benchmark's original **512 MiB** cap are unchanged.
Reproduce the completed experiment from its frozen source, not current main.
This was an isolated offline source qualification, not a full physics-regression
or native-boxing qualification. A fresh full source gate is still required before
adopting the writer into a scientific collector or running new formal physics.

## Next ready task

Follow [NATIVE_BALANCE_STRIKE_NEXT_PACKET.md](NATIVE_BALANCE_STRIKE_NEXT_PACKET.md),
starting with its zero-step native glove-face audit. Read `scene.py`,
`MatchReferee.stage` in `referee.py`, `physics.py`, `score_adapter_fixture.py`
and their existing tests. Verify native glove/wrist axis, target normal, both
corners/yaws, mirrored hands, contact-order sign and point rotational velocity
using scratch FK and positive/negative controls. Synthetic contacts are a
calibration diagnostic, never evidence of a real force-producing punch.

Do not flip axes to make the old touches score or silently alter anatomy/rules.
If a convention is inconsistent, register a named equipment/measurement variant
with new parity controls. Otherwise adapt the pretrained motor that actually
remained upright before asking for glove-face strikes. Preserve all 700 muscles.
Integrate this qualified writer only with a new complete-output preflight,
bounded protocol and fresh full Linux source gate. The old native budget stays
effectively exhausted. No owned job remains, no v11 tag, and no competitive
learning promotion until motor/contact/rules/replay gates pass.
