# H16 exit-memory observation

Frozen66ff041/protocol88c62c95. Six Linux mechanics/data-quality cases pass;
whole4.395757s. Two sequential, single-thread Python heaps produced47nonterminal
missing-RSS samples:1normal-exit/46direct-exit. Every sample shows stateR before
and after, matching PID/start, absent VmRSS/VmSize, seven explicit zero statm
fields, and PF_EXITING on the leader and all observed threads. Scientific verdict:
**supported-synthetic-exit**, not a qualified production reader.

No native model, physics, actor or learning ran. The failed H15worker's precise
flags were not captured; synthetic reproduction supports an explanation, rather
than proving that historical event's exact kernel state. Three blind rereads
did not fix the native failure; no more unchanged RL attempts are authorized here.

The interpretation is consistent with upstream Linux6.8: exit_mm clears the task's
mm before completing later teardown; proc status emits memory fields only when
get_task_mm succeeds, while statm emits zeros without an mm. These are upstream
references, not a complete audit of Ubuntu's patched source. The actual host's
matching kernel header/PF_EXITING definition is separately hashed.
[Exit code](https://raw.githubusercontent.com/torvalds/linux/v6.8/kernel/exit.c),
[proc interfaces](https://raw.githubusercontent.com/torvalds/linux/v6.8/fs/proc/array.c),
[RSS counters](https://raw.githubusercontent.com/torvalds/linux/v6.8/fs/proc/task_mmu.c).

Independent audit verifies727source files/five historical objects/one installed
header, all6JUnit identities, every raw sample/verdict and eight recorded exits/no
writer. All13rawfiles259083bytes transferred exactly; fixture27752/receipts231331
bytes. [Audit](../artifacts/v0.0.11/development/exit-memory-observed-01-integrity.json),
[transfer](../artifacts/v0.0.11/development/exit-memory-observed-01-transfer.json),
[protocol](../experiments/v0.0.11/exit-memory-diagnostic-protocol.json).

H15remains2INVALID/6NOT_RUN,10172known physics/preparation steps and24updates in
reference prefixes; no candidate trial qualifies. A supplementary RETROSPECTIVE
read-only comparison finds all10reference hash histories identical across those
two invalid attempts. This neither repairs their missing final semantic results
nor changes their failed criteria. [Supplementary comparison](../artifacts/v0.0.11/development/deferred-invalid-prefix-comparison-01.json).

Decision: implement a narrow exit-aware, conservative reader and qualify actual
native constructor teardown before further RL. Existing8GiB RSS/100ms/equality
gates, legacy default and all native muscle/model/rules remain unchanged.
[Next worker-ready mission](EXIT_AWARE_RSS_READER_MISSION.md).
