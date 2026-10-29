# Joxing game contract: candidate v0

Status: design selected for implementation; referee fixtures exist; full game is not yet playable. v10motor pilot concluded with both improvement gates failed; v11shared-loop/diagnostic-recording implementation is active.

Repository: [neverhuman/Joxing](https://github.com/neverhuman/Joxing), public. [v7](V0.0.7_RESULTS.md) confirms native joint-tracking termination can occur with sampled foot support still present; [v8](V0.0.8_RESULTS.md) measures persistent physical failure around5s with all40attempts valid and all12replays exact. Neither defines a sporting loss or changes the points below. The [v0.0.6 native walking qualification](V0.0.6_RESULTS.md) establishes a frozen motor dependency with no opponent/game integration or new learning. The completed [v0.0.5 motor experiments](V0.0.5_RESULTS.md) measure one athlete’s moving-start fall recovery before competitive play; the [training guide](TRAINING.md) describes later stance/recovery gates. Motor rewards and failure criteria do not change the sporting points or resolve provisional match fall/foul rules below.

## Objective and points

Two identical MS-Human-700 athletes share one MuJoCo world. A live round lasts 60 seconds. Higher legal-hit points wins; exact ties draw. The virtual arena is 5 by 5 meters on a larger physical floor. Initial morphology, glove equipment and rules are symmetric. This is a virtual points sport, not professional or Olympic judging.

| Contact | Points | Mechanical condition |
|---|---:|---|
| Front/side head, including chin/temple labels | 2 | Qualified direct glove striking-face impact |
| Front/side chest/abdomen, including liver/solar labels | 1 | Qualified direct glove striking-face impact |
| Gloves, forearms, upper arms | 0 | Guard physically blocks the strike |
| Back of head, back, below belt | 0 | Reviewed foul evidence, never a legal target |
| Floor, own body, incidental non-glove contact | 0 | Remains physical; cannot generate punch points |

Chin, temple, liver and solar plexus are useful telemetry labels. Their separate vulnerability multipliers are deferred to an independently versioned game variant. Head already carries more points than torso. Adding more location weights is easy; validating zone geometry and preventing seam exploitation are the hard parts.

Impulse qualifies a hit. It does not multiply its points. Peak force, vector impulse, normal impulse and location remain visible in telemetry. Forces are model predictions until empirical calibration exists. No injury, concussion, organ-shock or physiological knockout claim is supported.

The current adapter's mechanical striking face is glove-body local −Y and the
outward hemisphere of its spherical collider. The equipment center is displaced
45 mm along that direction from the native hand origin; radius is62 mm. This
documents the existing implementation, not an independently calibrated anatomical
knuckle surface. [Native face diagnostic](NATIVE_FACE_AUDIT_MISSION.md) checks
that convention and point-velocity signs using scratch FK/synthetic contacts;
it cannot establish a real punch or alter the2/1point rules.

## One punch, one score

One state machine per glove tracks an encounter with the opponent across manifold points, scoring zones and brief contact gaps. Initial candidate constants: 50 ms scoring window; 8 ms gap hysteresis; 0.25 N s qualifying direct-zone impulse; 0.5 m/s target-onset approach and active glove motion; face-axis cosine at least 0.5. These are simulation-rule trial values, not biological thresholds.

Start the window at the first actual force-producing opponent contact. A guard brush does not veto a subsequent direct target contact within that original window. Guard force cannot qualify a target hit. Latch each target zone's onset kinematics before deceleration. Sum simultaneous contact points once per physics interval. Select the zone with greatest eligible direct impulse; ties prefer lower points, then stable zone ID. A selected zone must reach its own threshold.

After finalization, held contact cannot score again. Rearming requires 30 ms continuously clear, at least 2 cm geometric clearance from all opponent shells, and 3 cm measured glove retraction. Current implementation consumes these as fixture inputs; production must compute them from geometry/motion, never from an attack action flag. Stable event IDs make log retries idempotent. A bell finalizes only evidence already integrated before it.

## Winning, fouls, falls and boundaries

The intended final referee uses three mechanical fouls for disqualification. Kicks, elbows, active non-glove strikes and sustained drives need their own attributed motion/impulse evidence. Blocks and incidental body contact need explicit exemptions. Unknown physical surface categories fail validation.

Falls and complete arena exits are provisional until foul evidence is adjudicated: an illegal shove must not win by knocking the defender out of bounds. Simultaneous valid losses draw. A non-foot support contact needs force and persistence checks; a glove-floor brush is not automatically a loss. Thresholds, separation controller and attribution are unresolved. Consequently **competition is disabled**, rather than substituting a pelvis-height shortcut or unreviewed loss rule.

An independently valid sporting end terminates both actors. The task's 60-second bell is a real termination. A collector chunk boundary preserves the live match; external resource truncation bootstraps from its genuine final state. Numerical failures invalidate and quarantine a match; they are never wins or silent draws. Preserve attempted/valid/invalid counts and block promotion on unresolved systematic faults.

## Where RL appears

[The executable-code audit](STATE_HISTORY_SELF_PLAY_AUDIT.md) distinguishes current single-athlete feedback from the proposed opponent-aware temporal game below. It specifies missing state/history, timing-jitter, simultaneous action and league interfaces. The current policy observes its own velocities and muscle activation but has no opponent input or recurrent actor memory.

1. **Motor training:** muscle commands learn stance, independent arm reach/return, footwork and perturbation recovery. Goals and imitation rewards may be dense. The first executable experiment is a separate fixed-pelvis, zero-gravity reach rig using all 700 native muscle actuators and learned right-arm excitation pulses. Its results do not count as boxing competence.
2. **Skill compression:** distill competent motor behavior into a state-conditioned deterministic decoder; test 32/64/128 latent dimensions. Motor policy outputs remain muscle excitations, not torque or animation.
3. **Tactical self-play:** a recurrent actor chooses latent actions at 50 Hz; decoder updates muscles at 100 Hz; validated physics advances at 1 kHz initially. Both players decide from the same state. The actor sees own proprioception, visible opponent kinematics and public game state. Privileged critic inputs stay separate.
4. **League:** freeze complete opponent bundles (actor, decoder, normalizers, timing, action map); sample history and challenging opponents at reset. Tactical PPO updates only designated learner trajectories. Keep a payoff matrix, not just Elo.
5. **Proof:** compare frozen candidates against an unchanged evaluation panel using paired starts with swapped sides. Report confidence intervals, all training seeds, falls, fouls, energy proxy, contact stability, invalid matches and equal compute budgets. Training reward or an attractive replay alone is insufficient.

Canonical tactical reward is terminal +1 win, -1 loss, 0 draw with gamma=1. Optional potential shaping uses `Phi=0.25*tanh(score_difference/4)` and `r'=r+Phi(next)-Phi(current)`, with Phi zero at true terminals. Thus total shaped return remains the match objective. No permanent survival, block-force or engagement bonus can silently replace winning.

## Visible demonstration target

A replay should show both athletes, clock, points, contact location, instantaneous force, event impulse, qualifying/rejected reason, policy and decoder versions. A companion panel should show frozen-panel strength versus simulator transitions and wall time, plus invalid rate and sample throughput. Keep constrained motor clips, scripted contact fixtures and learned unassisted matches visibly labeled.


## Implementation sequence and cadence qualification

The [2026-10-27integrated plan](SELF_PLAY_EXECUTION_PLAN.md) preserves this points contract and prioritizes a simultaneous diagnostic BoxingEnv before competitive learning. Its prototype may use the existing composition-smoke physics only under that explicit identity. The qualified native teacher acts at50Hz with2msEuler physics; the50Hz-tactics/100Hz-motor/1kHz-physics schedule above remains proposed and must not be silently applied to that checkpoint. Scoring coverage, self-contact, sporting ending attribution and match-scene motor transfer remain gates. New supplied health/KO/10-point judging proposals are not the current game.
