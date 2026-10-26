# Changelog

## v0.0.2 — 2026-10-26

- Connect actual MS-Human-700 muscle control to physical glove contact, geometry-derived onset/face/withdrawal evidence and the existing point referee.
- Add complete engine/application snapshots and mid-contact restore validation.
- Add bounded three-seed policy search, immutable proof protocols, atomic progress/checkpoints, held-out evaluation and fixed-scenario visual evidence.
- Frozen proof: trained controllers hit once and withdraw in 192/192 trials, versus 0/192 baseline withdrawals; both retain 192/192 legal hits. Zero invalids; 256.813 seconds on four xbabe1 CPU workers.
- Verify all three checkpoints and contact snapshots independently on Mac; 40 tests pass on both Mac and Linux.
- Preserve fixed-pelvis/zero-gravity assistance and model-force limitations. Unassisted stance, full-body contact/rules, self-play and GPU parity remain future gates.

See [results](docs/V0.0.2_RESULTS.md), [plan](docs/V0.0.2_PLAN.md) and [raw evidence](artifacts/v0.0.2/).
