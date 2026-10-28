# Grokbot red-team — cycle 0025 (2026-10-28): v0.0.19 preregistration (hit-and-stay-up)

Reviewed `origin/main` at `573c351`. jekko has made no new commits. Per
owners 07:20 and 07:40, this PR lands the v0.0.19 preregistration **on its
own, before any run**. The run starts only after merge, at nice 19 and with
loadavg below 86.

## Protocol `grokbot/redteam/protocols/v0.0.19.json`

| Item | Registered value |
|---|---|
| **One change** | The training fitness only: `stand/3000 + 1·hit_and_stayed_up(result, 500)`. The bonus needs a head or torso hit, **and** the athlete must still be standing 500 ticks (0.5 s) after its first hit. This targets cycle 0024 F2, where v0.0.18 hit and then fell first. |
| Everything else | Unchanged: `GroupPDPolicy`, initialized from v0.0.18's final parameters; 48 iterations of 12 ES pairs, seed 9; gate seeds and opponent as before |
| **Primary, and the only level-up** | The a/b/c gate against the **frozen v0.0.13 champion** on **new seeds 5000–5023**, both sides, 48 games. It passes only if all three hold: (a) valid ≥ 50 %; (b) Hoeffding lower95 > 0, which is the same as an Elo lower bound > 0; (c) at least half of the candidate's valid wins include its own head or torso hits. **`v0.0.19` is tagged only if this passes.** |
| Secondary: baseline | v0.0.18 on the identical primary games, for context |
| Secondary: non-regression | v0.0.19 **and** v0.0.18 against the frozen v0.0.16, on fresh seeds 6000–6023. Improvement is reported as v0.0.19's mean minus v0.0.18's mean. |
| Secondary: motor | The motor gate, with v0.0.18 re-run per trial against its committed summary; the exact sign test on stand ticks against v0.0.18; the training share of hits followed by staying up |

**Red-team note, registered in the protocol.** v0.0.18 already passed this
gate on other seeds. A v0.0.19 pass therefore shows that the new fitness kept
the hits; it does not by itself show a gain over v0.0.18. The last *tagged*
version, v0.0.17, fails the gate because it lands no hits. The v0.0.16
non-regression and the motor numbers are reported so that a pass is not
over-read. Each outcome has a pre-declared reading.

**Leakage.** No head-to-head has been played on seeds 5000–5023 or
6000–6023. The probe's smoke test used seed 99998.

## Code

| Change | Detail |
|---|---|
| `motor_gate.trial` | Now reports `first_scoring_tick` |
| `motor_gate.hit_and_stayed_up` | New |
| `motor_gate.fitness` and `train` | New `stay_up_ticks` argument. The default keeps the old behaviour exactly. |
| ES history | Records `hit_and_up_share` |
| Test | The boundary cases: still up exactly 500 ticks after the hit gets the bonus; falling 1 tick early, never hitting, or hitting too late does not. A mutant using `>` instead of `>=` fails. |
| Probe | `probes/cycle-0025-hit-stay-gate.py`, which reads the protocol unchanged |
