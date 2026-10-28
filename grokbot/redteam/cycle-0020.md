# Grokbot red-team — cycle 0020 (2026-10-28): v0.0.15 rerun reproduces exactly; v0.0.16 preregistered

Reviewed `origin/main` at `56b65ae`. The degenerate-league guard landed as
#19. jekko's `6b6ba04` added a motor-evidence codec preregistration. It
touched no grokbot or CI paths, and CI on the merge passed 1,093 tests.

## v0.0.15 gate rerun (owner 02:42 / 03:12): miss confirmed, fully deterministic

The preregistered v0.0.15 gate (protocol `32771370…`) was run a second time,
unchanged, on merged main `56b65ae`. That run took 1,228 s with 4 nice-10
workers, and its report has SHA-256 `440ce935…`.

| | First run (`548ac92`) | Rerun (`56b65ae`) |
|---|---|---|
| Baseline gen4: stance / reach / median stand | 0/24, 0/24, 1.464 s | identical per trial |
| Candidate: stance / reach / median stand | 0/24, 0/24, 1.349 s | identical per trial |
| 24-iteration ES history | — | identical |
| Final parameters | — | identical |
| Primary gate | **failed** | **failed** |

The physics, the ES and the evaluation all reproduce bit-for-bit on the same
host. The miss is real and not run-to-run noise. **v0.0.15 is not a
level-up and is not tagged.**

## v0.0.16 preregistration: attacking the measured root cause

`grokbot/redteam/protocols/v0.0.16.json` (sha256 `1ff40c3d…`) lands in this
PR, before the v0.0.16 run. Cycle 0019 found that every league game since
v0.0.12 is invalid because no policy stands longer than about 1.5 s. The
first suspect is controller capacity: 10 regional tones are too coarse to
hold the native keyframe.

| | v0.0.15 | v0.0.16 |
|---|---|---|
| Controller | `PunchPolicy`: 10 regional tones + regional feedback + punch (146 params) | **`GroupPolicy`: 138 per-DynSyn-group tones** + regional feedback applied per group + punch (284 params) |
| Initialization | gen4 + early punch | gen4's regional tones copied to each group. On the 656 muscles whose group lies inside one region, the commands are bit-identical to v0.0.15's starting policy; 4 groups (44 muscles) span two regions. |
| ES | 24 iterations × 8 pairs | 32 iterations × 12 pairs, seed 6 |
| Gate, seeds, opponent, fitness, horizon | — | **identical** to v0.0.15 |

The primary gate is unchanged: stance ≥ 0.5 **and** reach ≥ 0.5 over the
same 24 noisy trials. The secondary measures are the paired stand-time gain
against gen4, and against the v0.0.15 candidate. **v0.0.16 is tagged only
if the primary gate passes.**

Pre-declared readings:

- **No stand-time gain:** tone capacity is not the limit. The next lever is
  feedback or recurrent control, or jekko's pretrained bridge.
- **Stand time improves but the gate fails:** capacity helps. Scale up.
- **Stance passes but reach fails:** retarget the reach.
- **Both pass:** preregister a league with the valid-game floor and an Elo
  criterion against the v0.0.12 baseline.

## What this PR adds

| Item | Detail |
|---|---|
| `motor_primitives.GroupPolicy` and `group_params` | New |
| `selfplay._controller` | New kind `group` |
| Test | 1 equivalence test: `GroupPolicy` initialized from gen4 matches `PunchPolicy` on ≥ 600 muscles over three ticks; parameter-length and noise-generator validation |
| Evidence | The v0.0.16 protocol and probe, and this report |
