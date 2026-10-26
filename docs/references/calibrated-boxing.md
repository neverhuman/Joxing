# MS-Human Boxing Engineering Spec

Oct 26, 2026 · @Jepson Taylor

## Scope and definition of done

This spec builds a two-player boxing simulator and self-play training system on top of the unmodified MS-Human-700 XML and the msgym code, producing muscle-driven boxers that stand, move, punch and defend, with every contact measured and scored by location. It extends the prototype in `ms-boxing.zip` (scene builder, hit tracker, two-agent environment) into a production package called `msbox`.

| ID | Deliverable | Done when |
| --- | --- | --- |
| D1 | Model builder: repo XML to a calibrated two-boxer MuJoCo model | Builds from pinned commits; all unit and physics-regression tests pass |
| D2 | Calibration report: glove contact, muscle speed, neck | Gate 0 targets met (section 5) |
| D3 | Environment with CPU and GPU backends | Both backends agree on rewards and hit events over a fixed 1,000-step replay |
| D4 | Skill controller and skill library | Gates 1 and 2 met (section 9) |
| D5 | Bout policy trained in a self-play league | Gates 3 and 4 met; Elo ladder published |
| D6 | Telemetry, replay and fight statistics | Every hit logged with all fields; any bout replays from its log |
| D7 | Evaluation report | Realism metrics compared with published boxing data (gate 5) |

Pinned inputs: [ms-human-700](https://github.com/LNSGroup/ms-human-700) at commit 2d68695, [msgym](https://github.com/LNSGroup/msgym) at ad3aac1, MuJoCo 3.17 (the version every number here was measured on), Python 3.12 or later (msgym's floor).

Out of scope: brain-injury modelling beyond head-kinematics proxies, articulated fingers, a human-versus-agent interface, and transfer to robots.

Open question: team size and compute budget are not given; section 14 assumes three engineers and section 11 sizes compute for two budgets.

## System architecture

The system is four layers built in order: a calibrated two-boxer model, a skill library learned from motion capture, a self-play league that trains the bout policy on top of the frozen skills, and an evaluation layer that logs, replays and rates every bout.

&#91;embedded content: system architecture · 4 layers, 12 components\]

Each layer only consumes frozen artefacts from the one above it (a model hash, a skill-controller checkpoint, a league snapshot), so a layer can be retrained without rebuilding the ones before it.

## Repository layout and module contracts

One Python package, `msbox`, with the two upstream repos as pinned git submodules; nothing upstream is edited, every model change is applied in code through MuJoCo's spec API.

```
msbox/
  third_party/ms-human-700/      # submodule @ 2d68695 (XML + meshes, read-only)
  third_party/msgym/             # submodule @ ad3aac1 (DynSyn groups, SAC_DynSyn)
  msbox/
    model/   build.py surgery.py shells.py sensors.py calibrate.py configs/*.yaml
    sim/     cpu_backend.py warp_backend.py
    env/     bout_env.py single_envs.py obs.py reward.py scoring.py rules.py randomize.py
    skills/  retarget.py tracking_env.py skill_model.py scripted.py
    rl/      ppo.py sac_dynsyn.py league.py ratings.py rollout.py
    telemetry/ logger.py replay.py stats.py
    eval/    benchmarks.py report.py
  tests/   unit/ physics/ smoke/
  scripts/ stage0_calibrate.py ... stage5_realism.py
```

| Module | Main entry point | Input | Output | Invariant |
| --- | --- | --- | --- | --- |
| `model.build` | `build_bout(cfg) -> (MjSpec, MjModel)` | `BoxerConfig` × 2, `ArenaConfig` | Compiled two-boxer model | Same config and commits give a byte-identical XML (`spec.to_xml()` hashed in the run record) |
| `model.surgery` | `freejoint_root(s)`, `prune_inert(s)`, `split_head(s, neck)`, `set_vmax(s, table)` | One boxer's `MjSpec` | Same spec, modified in place | Each is idempotent and logs what it changed |
| `model.shells` | `add_strike_shells(s, tag)` | Spec, `'A'` or `'B'` | Gloves, shells, collision bits, own-body pairs | Shell frames are forward/left/up at the reference pose |
| `model.sensors` | `add_sensors(world)` | Bout spec | Contact, IMU and foot sensors with a name-to-slice map | Sensor layout identical on CPU and GPU |
| `model.calibrate` | `run_rig(params) -> RigResult` | Contact, neck, glove parameters | Peak force, impulse, head accelerations | Deterministic for fixed seed |
| `env.bout_env` | `BoxingEnv.reset/step` | Dict of actions per agent | Obs, rewards, terminations, truncations, infos | Observation is mirror-symmetric between corners |
| `env.scoring` | `HitTracker.update(data) -> list[Hit]` | `MjData` (CPU) or sensor rows (GPU) | Finished hits | CPU and sensor paths emit the same hits on a replay |
| `skills.skill_model` | `encode(clip) -> z`, `act(obs, z) -> excitations` | Proprioception, skill code | 508 excitations per boxer | Excitations in \[0, 1\] |
| `rl.league` | `sample_opponent(agent_id) -> snapshot` | Win-rate table | Frozen policy | Every snapshot immutable and content-addressed |
| `telemetry.logger` | `log_step`, `log_hit`, `close()` | Step and hit records | Parquet files per bout | Any bout reproducible from its seed and logged actions |

Configuration is one YAML file per run, validated into dataclasses: `boxer` (surgery switches, vmax table, neck and glove parameters), `arena` (ring size, contact settings, time step), `env` (control rate, delay, randomisation ranges, reward weights), `train` (stage, algorithm, hyperparameters) and `league`. The run record stores the config, both submodule commits, the MuJoCo version and the compiled-model hash.

## Model build pipeline

`build_bout` applies twelve operations to two copies of the stock `MS-Human-700.xml`, in this order; every row marked "measured" was run in MuJoCo 3.17 for this spec. Result per bout: 178 qpos, 176 DOF, 1,016 muscles.

| # | Operation | Exact parameters | Measured or expected effect |
| --- | --- | --- | --- |
| 1 | Load and clean | `MjSpec.from_file`; delete keyframe `init` and the `floor` geom | Two copies can be attached without name or keyframe clashes |
| 2 | Freejoint root | Delete `pelvis_tz/ty/tx/tilt/list/rotation`; add freejoint `root` on `pelvis` | Measured: stock root lying on its side has a root mass-matrix condition number of 3×10^18, and MuJoCo flags singular inertia and a NaN/huge acceleration, then auto-resets; freejoint: condition 91, clean |
| 3 | Prune inert muscles | Delete the 192 actuators with zero moment arm on every DOF, their 192 tendons and the 384 sites only they use | Measured: lossless (zero generalised force), single-boxer step 1,286 → 1,637 steps/s (+27%); bout model 742 → 864 steps/s (+16%); 508 muscles remain |
| 4 | Split head from neck | New body `skull_head` at the top of C1 (reference-pose world (−0.071, 0, 1.587) m); hinges `C0_FE` ±0.4, `C0_LB` ±0.3, `C0_AR` ±0.6 rad; stiffness 20/20/10 N·m/rad, damping 0.5/0.5/0.3 N·m·s/rad, armature 0.002; move `skull` and `jaw` meshes and 14 skull-side sites (incl. 4 mastoid insertions); head 3.0 kg, inertia diag (0.016, 0.014, 0.012) kg·m²; cervical remainder 0.664 kg | Measured: 14 neck muscles now cross C0 (sternocleidomastoid, cleido-occipital, clavicular trapezius, splenius and semispinalis capitis); head angular acceleration on a chin shot 1,176 → 3,396 rad/s² (section 5) |
| 5 | Gloves and strike shells | Glove: sphere r 0.062 m, 0.283 kg on `hand_*`. Massless shells: head ellipsoid (0.100, 0.085, 0.125) m on `skull_head`; chest (0.135, 0.175, 0.190) on `thoracic12`; abdomen (0.125, 0.150, 0.095) on `lumbar5`; forearm capsule r 0.040, upper-arm r 0.048. Skull mesh collision off | Prototype: blocks and landed hits classified correctly |
| 6 | Collision filtering | contype/conaffinity: floor and ropes 1/1, boxer A 2/5, boxer B 4/3. Own-body pairs: glove–own head, glove–own chest, glove–glove, forearm–own head; plus leg pairs femur–femur, tibia–tibia, foot–foot so crossed steps still collide | The bitmasks remove the stock model's leg-against-leg contacts, hence the extra pairs |
| 7 | Glove contact | Pair solref (2·dt, 1) = (0.004, 1) for glove against opponent geoms; solimp default; friction 0.5; condim 3 | Measured on the impact rig: 3,394 N, 51.5 g at 2 ms step (section 5) |
| 8 | Muscle speed | Per muscle `vmax = 10 · Lopt / L0`, with Lopt the optimal fibre length from the source OpenSim model and L0 MuJoCo's lumped length; same factor on `biasprm[6]`. Fallback: global ×2 | Measured, arm-only jab: peak glove speed 4.65 m/s at the stock 1.5 L0/s; 5.43 at ×2; 6.04 at ×4; 6.35 at ×6.67 |
| 9 | Excitation range | ctrlrange \[0, 1\]; do not reuse msgym's sigmoid wrapper (caps at 0.924) | Full muscle force available |
| 10 | Sensors | Contact sensors per (glove, target shell): `data="found force pos normal" reduce="maxforce" num="1"` (10 floats); accelerometer, gyro and framequat at a site on the `skull_head` centre of mass; one contact sensor per foot against the floor | Measured: contact sensor compiles as type 42, dimension 10, force in the contact frame |
| 11 | Arena | Floor plane; four rope boxes at ±3.05 m, solref (0.05, 0.5); A at −x facing +x, B at +x yawed 180°; prefixes `A_`, `B_` | 6.1 m ring |
| 12 | Global options | timestep 2 ms, Euler integrator, Newton solver, elliptic cones | Measured: Euler 763 steps/s vs implicitfast 390 on the bout; elliptic costs the same as pyramidal |

Where the time goes: kinematics, which includes the 5,712 sites (before pruning) and tendon wrapping, is 65% of each step; collision and the constraint solver are about 20%. Pruning (step 3) and the GPU backend both attack that 65%.

Optional step 13, weight classes: scale segment masses by m/60.2 kg and muscle peak forces by (m/60.2)^(2/3), the usual geometric-similarity assumption; validate by re-running the stage-1 gate.

## Physics calibration

Calibrate on a scripted impact rig before any learning, because rewards, damage and knockdowns all read contact force and head kinematics. At the production 2 ms step the glove already lands within 1% of the published punch force; head angular acceleration is the gap, 4–5× low on the stock head and about 2× low after the head split.

**Rig (`model.calibrate.run_rig`):**

- **Defender:** boxer B, pelvis welded to the world, all excitations 0 (passive); a second pass at 30% neck co-contraction represents a braced boxer.
- **Impactor:** free sphere, radius 0.062 m, mass 2.9 kg (the effective punch mass in Walilko 2005), gravity-compensated, launched at 9.14 m/s (their mean fist speed).
- **Shots:** straight to face centre, straight to chin, hook to the side of the jaw, uppercut to the chin.
- **Measures:** peak normal force, impulse, contact duration, and peak linear and angular acceleration of the `skull_head` centre of mass (finite difference at the physics step, cross-checked against the IMU sensors).

| Shot | Head model | Peak force (N) | Impulse (N·s) | Head linear (g) | Head angular (rad/s²) |
| --- | --- | --- | --- | --- | --- |
| Straight, face centre | Stock | 3,394 | 22.3 | 51.5 | 1,544 |
| Straight, chin | Stock | 3,315 | 45.3 | 45.0 | 1,176 |
| Hook, side of jaw | Stock | 3,464 | 22.6 | 48.2 | 1,648 |
| Uppercut, chin | Stock | 3,047 | 14.3 | 14.0 | 238 |
| Straight, face centre | Split | 3,151 | 21.9 | 49.8 | 604 |
| Straight, chin | Split | 3,554 | 40.4 | 37.6 | 3,396 |
| Hook, side of jaw | Split | 3,111 | 21.2 | 43.3 | 3,364 |
| Uppercut, chin | Split | 2,998 | 14.2 | 11.2 | 1,467 |
| Olympic straight punches ([Walilko 2005](https://pubmed.ncbi.nlm.nih.gov/16183766/)) | Hybrid III dummy | 3,427 | — | 58 | 6,343 |

All rows: 2 ms step, glove pair solref (0.004, 1), split-head neck at the step-4 defaults. Softening the glove contact on the stock head (face-centre shot) moved peak force from 3,394 N (4 ms time constant) to 1,994 N (20 ms) while impulse stayed at 22.3–22.7 N·s; halving the step to 1 ms or 0.5 ms raised peak force by up to 10% at the stiffest setting. Neck stiffness from 2 to 20 N·m/rad changed chin-shot angular acceleration by under 8%: inertia, not the spring, governs the first milliseconds. Contact lasted 26 ms on the stock-head face-centre shot at the production setting, inside the 18.5–28.8 ms measured for gloved rear straights by Nakano, Iino and Kojima (2009).

**Procedure:**

1. Free parameters: glove pair solref time constant (≥ 2·dt) and damping ratio, solimp width; head mass (2.8–4.5 kg) and inertia; C0 position (±1 cm); C0 stiffness and damping; head-shell semi-axes and a chin extension of the shell.
2. Objective: mean squared relative error of peak force, peak linear and peak angular head acceleration over the straight-punch shots, against the Walilko means.
3. Search: CMA-ES, 200 rig evaluations; one rig run is 80 ms of simulation, well under a second of CPU.
4. Accept when force is within ±15%, linear acceleration within ±20% and angular acceleration within ±25% of target, and impulse is unchanged to within 5% across the accepted settings.
5. Freeze the result as `calib.yaml` and as golden values in the physics regression tests (section 13); any later model change that moves a golden value by more than 5% fails CI.

**Muscle speed** is calibrated separately: build the per-muscle vmax table (step 8 of section 4), then check that an isolated, pinned-pelvis elbow extension at full triceps excitation stays below the source model's published joint-velocity range, and that the stage-2 straight punch reaches at least 7 m/s glove speed with legs and trunk free.

## Environment spec

`BoxingEnv` is a batched two-agent environment with a PettingZoo-style parallel API: physics at 500 Hz, muscle control at 100 Hz, bout decisions at 25 Hz, and opponent information delayed by about 100 ms.

**API**

```python
env = BatchedBoxingEnv(cfg, n_envs=4096, backend="warp")   # or "cpu" (threads)
obs, info = env.reset(seed)                      # obs[agent]: float32 [n_envs, obs_dim]
obs, rew, term, trunc, info = env.step({"A": a_A, "B": a_B})   # a: [n_envs, act_dim]
```

`term` is true only for knockouts, credited knockdowns and slips; running out of time is `trunc` (msgym sets both, which corrupts bootstrapping). Instabilities (NaN, MuJoCo's huge-acceleration warning) set `trunc` plus `info["unstable"]`, and those transitions are dropped from training.

**Timing**

| Quantity | Value |
| --- | --- |
| Physics step | 2 ms |
| Skill-controller (muscle) step | 10 ms, 5 physics steps, actions held constant |
| Bout-policy step | 40 ms, every 4 muscle steps |
| Opponent observation delay | 100 ms default (10 muscle steps), randomised 80–150 ms per bout |
| Training episode | 30 s exchange, truncated and bootstrapped |
| Evaluation bout | 3 rounds × 180 s, knockdown count kept across resets |

**Observations** (per boxer, egocentric: the heading frame is the pelvis forward axis projected on the floor; all values float32, normalised by fixed constants stored in the config)

| Block | Contents | Size | Used by |
| --- | --- | --- | --- |
| Joints | Angles of the 40 independent non-root joints (37 stock + 3 new C0); velocities of all 46 independent DOFs | 86 | Both levels |
| Root | Pelvis height; gravity in pelvis frame; pelvis linear velocity (heading frame); pelvis angular velocity | 10 | Both |
| Key bodies | Positions relative to pelvis, heading frame: chest, head, both hands, both forearms, both heels, both toes | 30 | Both |
| Feet | Contact flag and normal force per foot | 4 | Both |
| Muscles | Activation, force / peak force, normalised length, normalised velocity for 508 muscles | 2,032 | Skill controller only |
| Skill code | z from the bout policy | 32 | Skill controller only |
| Opponent | Positions and velocities of pelvis + the 10 key bodies (heading frame), relative heading as cos/sin, delayed; last 3 bout steps stacked | 3 × 68 = 204 | Bout policy |
| Match | Own position in ring (2), distance to nearest rope (1), round clock (1), round one-hot (3), score difference (1), own damage (1), own mean fatigue (1), knockdowns each (2) | 12 | Bout policy |

Skill controller input: 86 + 10 + 30 + 4 + 2,032 + 32 = 2,194. Bout policy input: 130 + 204 + 12 = 346. The critic gets both boxers' undelayed 130-value body blocks, both damage and fatigue values, and the match block: 276 values.

**Actions**

- Skill controller: 508 excitations in \[0, 1\] per boxer, written to `ctrl`; MuJoCo's activation dynamics (10 ms up, 40 ms down) sit behind them.
- Bout policy: z ∈ \[−3, 3\]^32, a residual on the skill prior's mean (section 8).
- Stage-1 and stage-2 single-boxer environments expose the 508 excitations directly.

**Resets**

- Pelvis separation U(1.4, 2.0) m, lateral offset U(−0.3, 0.3) m, yaw noise U(−15°, 15°), bout centre anywhere within 1.5 m of ring centre.
- Stance orthodox or southpaw at 50/50, via the left-right mirror map; initial pose from the mocap guard library plus joint noise of 0.02 rad; zero velocities.
- 20% of resets start from states sampled from recent bouts (clinches, ropes, post-hit staggers), as reference-state initialisation.
- No scoring for the first 0.2 s after a reset.

**Randomisation per episode**: glove contact time constant ±25%, floor friction 0.6–1.2, segment masses ±5%, muscle peak forces ±10%, reaction delay as above, observation noise 5 mm on positions and 0.05 m/s on velocities, one optional physics step of action latency.

**Mirror map**: built by name rules (`_r`↔`_l`, `_R`↔`_L`, and unsuffixed↔`_L` for neck and trunk muscles), then verified by checking that mirrored poses give mirrored moment arms to 1e-6; anything unmatched goes in a hand-written table that the unit tests cover.

## Scoring and reward spec

Every landed punch is a hit object produced by a per-glove state machine; points are zone multiplier × impact, and the bout reward is the points difference plus knockout terms, with shaping that is annealed away.

**Hit state machine** (one per glove, evaluated every physics step):

1. **Idle.** On the first step with a contact between this glove and any opponent geom, and at least 100 ms since this glove's last hit ended, open a hit.
2. **Onset (first contact step).** Record closing speed v\_c = (v\_glove − v\_target) · n, where n points from glove centre to contact point; fist-side flag = cos(n, forearm axis) > 0.3; target shell; contact point in the shell's frame.
3. **Accumulate.** Each step add F\_n · dt to impulse J; track peak F\_n and the defender's peak head linear and angular acceleration. A one-step gap does not end the hit, so chattering contacts merge.
4. **Close.** When contact is gone for 2 consecutive steps, or the hit has lasted 80 ms (longer is pushing), compute points and emit the hit.
5. **Refractory.** 100 ms before this glove can open a new hit.

**Zones** (contact point p in the shell frame, divided by the shell semi-axes to give u; x forward, y left, z up):

| Shell | Rule, in order | Zone | Multiplier |
| --- | --- | --- | --- |
| Gloves | any | Blocked | 0 |
| Forearms, upper arms | any | Blocked | 0 |
| Head | u\_x < −0.3 | Back of head (foul) | −2 |
| Head | u\_z < −0.35 and u\_x > 0 | Chin | 3 |
| Head | abs(u\_y) > 0.6 and u\_z > −0.1 | Temple | 2 |
| Head | u\_x > 0.3 | Face | 1.5 |
| Head | otherwise | Side of head | 1 |
| Chest, abdomen | u\_x < −0.3 | Back (foul) | −1 |
| Abdomen | u\_y < −0.2 (defender's right) | Liver | 2 |
| Chest | u\_z < −0.4, abs(u\_y) < 0.3, u\_x > 0.5 | Solar plexus | 1.5 |
| Chest, abdomen | otherwise | Body | 1 |
| Pelvis, sacrum, legs | any | Below the belt (foul) | −3 |

**Points for one hit**

```latex
I = \min\!\left(\frac{J}{J_{ref}},\,3\right), \qquad I_{head} = 0.5\,\min\!\left(\frac{J}{J_{ref}},3\right) + 0.5\,\min\!\left(\frac{\alpha_{peak}}{\alpha_{ref}},3\right)
```

```latex
P = w_{zone} \cdot I \cdot \mathbb{1}[\text{fist side}] \cdot \mathbb{1}[v_c \ge 1\ \mathrm{m/s}] \quad (w_{zone} > 0), \qquad P = w_{zone} \quad (\text{fouls})
```

J\_ref = 20 N·s, so a calibrated Olympic-level face punch (about 22 N·s on the rig) scores about 1.1 impact units; α\_ref = 6,000 rad/s². The head blend switches on only after gate 0 passes. Three fouls in a bout disqualify the boxer (terminal, counted as a loss).

**Knockdowns, damage, fatigue**

- **Knockdown:** pelvis below 0.55 m, or any geom other than heels and toes touching the floor. Credited to the opponent if a scoring hit landed on the faller in the previous 1.0 s; otherwise it is a slip.
- **Damage D per boxer** (initial values, to tune): each head hit adds (α\_peak/α\_ref)² + 0.5 (a\_peak/60 g)²; a liver hit adds 0.3 (J/J\_ref)², other body hits 0.1 (J/J\_ref)². D decays with a 60 s time constant.
- **Stun:** a single hit adding more than 1.0, or D above 3: muscle peak forces ×0.6 and reaction delay +100 ms for U(1, 3) s.
- **Knockout:** D above 6, or a single hit adding more than 2.5. Terminal, credited.
- **Fatigue:** a three-compartment (active, fatigued, rested) model per muscle scales its peak force; use MyoSuite's CumulativeFatigue implementation of the 3CC-r model (Xia and Frey-Law 2008, with the rest multiplier of Looft et al. 2018) and its defaults F = 0.00970, R = 0.00091, r = 15.

**Step reward** for corner A at each muscle step (corner B symmetric; the bout policy sums its 4 steps):

```latex
r_A = (P_A - P_B) + 20\,(K_A - K_B) - 5\,S_A + \beta \left[0.02\,\cos\theta + 0.02\,e^{-((d-0.75)/0.25)^2} + 0.05\,\mathbb{1}[z_{head} > 1.4]\right] - 0.002\,\frac{\sum act^2}{100} - 0.01\,\mathbb{1}[\text{idle} > 10\,s]
```

K is a credited knockdown or knockout of the opponent, S an unforced slip, θ the angle between the boxer's heading and the line to the opponent's head, d the horizontal head-to-head distance in metres. β anneals linearly from 1 to 0 over the first 20% of stage 4; energy and idle terms stay. The skill controller also pays 0.01 · mean((u\_t − u\_{t−1})²) for excitation chatter.

## Policy architecture

Five networks: a skill encoder, skill prior and skill controller learned together from motion capture (a conditional VAE over 32-number skill codes), and a recurrent bout policy that outputs skill codes, trained with a centralised critic.

| Network | Input | Body | Output | Approx. parameters |
| --- | --- | --- | --- | --- |
| Skill encoder q(z \| clip) | Next 10 reference frames at 100 Hz (40 joint angles + 30 key-body positions each) and current body block | 1-D conv (3 layers, 256 channels) + MLP 256 | Mean and log-variance of z ∈ R^32 | 0.6 M |
| Skill prior p(z\_t \| s\_t, z\_{t−1}) | 130-value body block + previous z | MLP 512-256 | Mean and log-variance of z | 0.2 M |
| Skill controller π\_L(u \| s, z) | 2,194 values (section 6) | LayerNorm, MLP 1024-1024-512, SiLU | 508 logits; excitation = sigmoid(logit) | 4.1 M |
| Bout policy π\_H(z \| o) | 346 values | MLP 512, GRU 256, MLP 256 | Residual Δz ∈ R^32 (Gaussian, learned state-independent log-std, init −1); z = prior mean + Δz, clipped to \[−3, 3\] | 0.8 M |
| Centralised critic V(s) | 276 privileged values | MLP 512-512-256 | Scalar value for corner A | 0.5 M |

Design rules:

- **Gaussian in logit space** for excitations, then a sigmoid, so the full \[0, 1\] range is reachable (msgym's wrapper stops at 0.924).
- **Optional synergy head:** the skill controller can output msgym's DynSyn group values instead of 508 logits; the 138 full-body groups are re-indexed after pruning; 4 groups consist only of pruned muscles and are dropped, leaving 134 (6 more lose some members).
- **Residual skill codes** keep the bout policy near motions the skill controller can execute; the KL between the residual and the prior is logged, and a penalty of 0.001 · KL keeps it from drifting.
- **Recurrence** sits only in the bout policy, which must track opponent rhythm and feints; the skill controller is feed-forward for speed.
- **Mirror augmentation:** every transition is also stored mirrored (section 6 map), doubling data and making orthodox and southpaw the same network.
- **Critic asymmetry:** the critic sees both boxers undelayed, plus damage and fatigue, which the actors cannot see.

The skill controller is evaluated 100 times a simulated second per boxer, the bout policy 25 times; at 4,096 parallel bouts that is about 820,000 skill-controller forward passes per simulated second of all bouts, a small load next to the physics on one modern GPU.

## Training pipeline by stage

Five learning stages follow calibration; skills are learned on a single boxer and frozen before any opponent appears, and no stage starts until the previous gate passes.

| Stage | Environment | Algorithm | Main reward terms | Gate to pass |
| --- | --- | --- | --- | --- |
| 1 Stand and move | Single boxer; stance holding with random pushes; step-to-target and pivot commands | PPO on GPU, or msgym's DynSyn-SAC on CPU | Head above 1.4 m; centre of mass over the support polygon; command tracking exp(−‖p − p\*‖² / 0.1); energy | Stands 60 s under pushes in 95% of trials; reaches commanded spots within 10 cm in 90% |
| 2a Punch tracking | Single boxer tracking retargeted clips, skill encoder in the loop | PPO with the conditional-VAE skill model, KL weight 0.01 | msgym tracking terms (below) plus glove-position tracking | Mean key-body error under 5 cm on held-out clips |
| 2b Target striking | Single boxer, a 6 cm target sphere placed in reach and moving at 0–1 m/s | PPO fine-tune of the skill controller with a KL penalty to the 2a prior | Impulse on target / 10 N·s; +2 for the requested punch type; energy | Lands every punch type on a moving target in 80% of attempts; straight punch glove speed ≥ 7 m/s |
| 3 Bout vs scripted | Bout; skill controller frozen; four scripted styles | PPO bout policy with centralised critic | Section 7 step reward, shaping weight β = 1 | Beats every scripted style in at least 70% of bouts |
| 4 Self-play league | Bout; league opponents (section 10) | PPO + PFSP league | Section 7, β annealed to 0 | Evaluation Elo flat (< 25 points over 3 snapshots); no regression against scripted styles |
| 5 Realism pass | Bout with 100 ms delay, fatigue, damage and stun | PPO; skill controller unfrozen at learning rate 10^−5 with a KL penalty to its frozen copy; motion-discriminator penalty | Section 7 plus naturalness penalty | Glove speed, punch rate and accuracy inside the published ranges (below); under 1 slip per 3 rounds |

**Stage 2a tracking reward** reuses msgym's full-body tracking terms with a glove term added: 50 · (−Σ joint-angle error²) + 50 · (−Σ key-body position error²) + 100 · (−pelvis position error²) + 50 · (−Σ glove position error²) − energy; an episode ends when the mean joint error exceeds 0.10 rad (msgym's configs use 0.06–0.10).

**Retargeting procedure** (`skills/retarget.py`):

1. Resample each clip to 100 Hz and scale it to the model's height (skull top at 1.74 m in the reference pose).
2. Match 19 source keypoints (pelvis, lumbar, thorax, neck, head, both shoulders, elbows, wrists, hips, knees, ankles, toes) to sites added on the corresponding MS-Human bodies.
3. Solve inverse kinematics per frame with mink over the freejoint root and the 40 independent joints, with joint limits, a smoothness term toward the previous frame, and coupled joints set from their masters afterwards; reject frames with mean keypoint residual above 3 cm.
4. Clean contacts: pin stance feet to the floor, remove floor penetration.
5. Validate by kinematic replay with gravity off (the same check as msgym's `kinematic_play`) and label each clip: jab, cross, lead and rear hook, lead and rear uppercut, slip, roll, step in, step out, pivot, guard idle.

**Motion data.** No large public marker-based boxing dataset exists; the usable public material is a few minutes of punching clips, so plan a commissioned capture and check every licence before training.

| Source | Content | Format | Licence | Use here |
| --- | --- | --- | --- | --- |
| [CMU Graphics Lab mocap](http://mocap.cs.cmu.edu/) ([index](https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/cmu-mocap-index-text.txt)) | Boxing trials 13\_17, 13\_18, 14\_01–03, 15\_13, 17\_10, 79\_08, 80\_10; subject 144 punch sequences and blocks | ASF/AMC, BVH conversions | Free for research; may ship in commercial products, not resold | Primary public source. Won et al. 2021 trained their boxers on about 90 s of CMU boxing from one lightly trained actor |
| [ACCAD Male2 Martial Arts Punches](https://accad.osu.edu/research/motion-lab/mocap-system-and-data) | 15 clips: jab, cross, hook, uppercut, backfist, both sides, body shots; no defence | C3D | CC BY 3.0 | Punch technique |
| [HDM05](https://resources.mpi-inf.mpg.de/HDM05/cuts/index.html) | 8 punch classes, 177 cuts | C3D / ASF | CC BY-SA 3.0 | Punch variety |
| [Kyokushin karate strikes](https://www.nature.com/articles/s41597-021-00801-5) | 3,229 punches and kicks | C3D | CC BY 4.0 | Supplementary strikes; karate technique differs |
| [AMASS](https://amass.is.tue.mpg.de/license.html) | SMPL-H versions of the above and more | npz | Non-commercial research only; bars training networks for commercial use | Non-commercial projects only |
| Video datasets ([BoxingVI](https://arxiv.org/html/2511.16524v1), [BoxingWeb](https://github.com/gouba2333/BoxingWeb)) | Punches and defence from video, 2D pose or estimated SMPL | Video, keypoints | Varies or unstated | Punch-distribution statistics; discriminator data only after a licence check |
| [Mixamo](https://wwwimages2.adobe.com/content/dam/cc/en/legal/servicetou/Mixamo-Addl-Terms-en_US-20210623.pdf) | Boxing animations | FBX | Additional terms forbid training machine-learning models | Do not use |

**Commissioned capture (recommended):** two or three experienced amateur boxers, about 2 hours of marker-based capture in 10–12 oz gloves: every punch type from both stances, slips, rolls, blocks, footwork, clinch, plus two-person sparring captured together for interaction data. Deliver C3D and SMPL-X fits.

**Retargeting tools:** [mink](https://github.com/kevinzakka/mink) (Apache-2.0) solves the per-frame IK on the MuJoCo model with limits; [GMR](https://github.com/YanjieZe/GMR) (MIT) already retargets SMPL-X, AMASS and BVH through mink; [MuscleMimic](https://github.com/amathislab/musclemimic) (Apache-2.0) shows the same path into a 416-muscle MuJoCo body on MuJoCo Warp. The MS-Human group's own tracking pipeline, [MS-Emulator](https://lnsgroup.cc/research/MS-Emulator/index.html), fits SMPL-X then solves IK for a modified MS-Human-700, with code listed as coming soon; ask the authors before rebuilding it.

**Published ranges for gate 5:**

| Measure | Published value | Source |
| --- | --- | --- |
| Jab (lead straight) peak fist speed | 7.16 ± 0.48 m/s elite; 5.56 m/s at contact | [Liu et al. 2022](https://frontiersin.org/articles/10.3389/fphys.2022.1015154/full) |
| Cross peak fist speed | 8.1 ± 2.1 m/s elite; 8.22 ± 1.08 m/s | [Dinu & Louis 2020](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.598861/pdf); [Piorkowski et al. 2011](https://knowledge.lancashire.ac.uk/id/eprint/3757/1/piorkowski-et-al-2011-punch-kinematics.pdf) |
| Hook peak fist speed | 11.2 ± 2.0 m/s elite | Dinu & Louis 2020 |
| Uppercut peak fist speed | 10.2 ± 1.8 m/s elite | Dinu & Louis 2020 |
| Straight punches to a dummy's face | 9.14 m/s, 3,427 N, 58 g, 6,343 rad/s² | [Walilko 2005](https://pubmed.ncbi.nlm.nih.gov/16183766/) |
| Gloved contact duration | 18.5–28.8 ms (rear straight into a 4.36 kg target) | [Nakano, Iino & Kojima 2009](https://ojs.ub.uni-konstanz.de/cpa/article/view/3427/3226) |

**Hyperparameters (starting values)**

| Setting | Bout policy (stages 3–5) | Skill level, PPO (stages 1–2) | Skill level, DynSyn-SAC (CPU option) |
| --- | --- | --- | --- |
| Parallel environments | 4,096 bouts | 4,096 boxers | 64 (msgym default) |
| Rollout length | 128 decisions (5.1 s) | 32 steps (0.32 s) | — |
| Discount γ / GAE λ | 0.995 / 0.95 | 0.99 / 0.95 | 0.99 / — |
| Learning rate | 10^−4, linear decay | 3 × 10^−4, adaptive to KL 0.01 | 5 × 10^−4, linear to 0.25× (msgym) |
| Epochs × minibatches | 4 × 8 | 5 × 4 | 1 gradient step per env step |
| Clip / entropy / value coefficient | 0.2 / 0.001 / 0.5 | 0.2 / 0 / 0.5 | Target entropy −(number of groups) |
| Recurrence | GRU, truncated backprop over 32 steps | None | None |
| Normalisation | Running mean and variance of observations; reward scaling | Same | VecNormalize on observations (msgym) |
| Network | Section 8 | Section 8 | 1024-1024-1024 (msgym full-body config) |

## Self-play league

A scaled-down AlphaStar-style league: one main agent that plays both corners, one main exploiter, two league exploiters and four scripted styles, with opponents drawn by prioritised fictitious self-play (PFSP) so the main agent keeps training against what it still loses to.

| Member | Trains against | Snapshot and reset rule |
| --- | --- | --- |
| Main agent | 35% its latest self; 50% PFSP over all frozen snapshots; 15% PFSP over exploiters and scripted styles | Frozen snapshot every 10^8 muscle steps (20 over stage 4); never reset |
| Main exploiter | 100% the current main agent | Snapshot and reset to the stage-3 policy when it beats the main agent in 70% of bouts, or after 4 × 10^7 muscle steps |
| League exploiters (2) | PFSP over every league member | Snapshot every 2 × 10^8 muscle steps; reset with probability 0.25 at each snapshot |
| Scripted styles (4) | Not trained | Heavy bag (guard only), pressure fighter, counter-puncher, stand-off mover, built from skill-code scripts |

**PFSP weighting:** the probability of drawing opponent j is proportional to f(x\_j), where x\_j is the learner's win rate against j (exponential moving average over about 200 bouts, draws count 0.5).

```latex
f_{hard}(x) = (1 - x)^2 \quad \text{(main agent vs snapshots)}, \qquad f_{var}(x) = x\,(1 - x) \quad \text{(vs exploiters and scripted)}
```

**Bout result for the league:** knockout or disqualification decides; otherwise points, with a draw when the score difference is under 0.5.

**Ratings and promotion:** Elo (K = 16) is computed only on a separate evaluation pool so it is not inflated by the training distribution; the full payoff matrix is also kept to spot non-transitive cycles. A snapshot is promoted to the evaluation pool when its Elo is within 50 of the best and it has not regressed against any scripted style.

**Implementation:** `rl/league.py` keeps the payoff table in SQLite, snapshots are content-addressed by the SHA-256 of their weights, and the rollout workers ask the league for an opponent at every reset. All members share the frozen skill controller, so a snapshot is only the 3.2 MB bout policy.

**Why this mix:** the main agent's 35/50/15 split is AlphaStar's ([Vinyals et al. 2019](https://www.seas.upenn.edu/~cis520/papers/RL_for_starcraft.pdf)), which ran three main agents, three main exploiters and six league exploiters; this spec scales that down. [Bansal et al. 2018](https://ar5iv.labs.arxiv.org/html/1710.03748) found that training humanoids only against the latest opponent was the worst option and sampling from the more recent half of history the best, and annealed their dense exploration reward to zero over roughly the first 10–15% of training, which is where the β schedule in section 7 comes from.

**Stop rule:** end stage 4 when the main agent's evaluation Elo has gained less than 25 points over the last 3 snapshots and gate 4 is met.

## Infrastructure, throughput and compute

The final bout model runs at 864 physics steps/s on one CPU core, so the lean training plan needs about 5,800 core-hours of physics (roughly a day on 256 cores) and a 10× budget about 9.5 days; GPU throughput for this model is unmeasured and is the first benchmark to run.

| Measurement (one Xeon core, 2.8 GHz, MuJoCo 3.17) | Steps/s |
| --- | --- |
| Single boxer, stock | 1,286 |
| Single boxer, pruned to 508 muscles | 1,637 |
| Bout, freejoint + split head, 1,400 muscles | 742 (median of 3) |
| Bout, final model, 1,016 muscles | 864 (median of 3) |
| Bout, final model with implicitfast integrator | about half of Euler (measured 390 vs 763 on the unpruned bout) |

**Budget** (control-step counts are planning assumptions, not measurements; msgym's full-body walker used 50 million steps at 50 Hz for one skill):

| Stage | Scene | Muscle steps (100 Hz) | Core-hours of physics |
| --- | --- | --- | --- |
| 1 Stand and move | Single boxer | 2 × 10^8 | 170 |
| 2 Punch skills | Single boxer | 1 × 10^9 | 850 |
| 3 Bout vs scripted | Bout | 5 × 10^8 | 800 |
| 4 Self-play league | Bout | 2 × 10^9 | 3,200 |
| 5 Realism pass | Bout | 5 × 10^8 | 800 |
| Total, lean |  | 4.2 × 10^9 | about 5,800 |

Core-hours = muscle steps × 5 physics steps ÷ steps/s ÷ 3,600. Learning, inference and logging add overhead; plan for 1.5×.

**Rollout architecture (CPU backend):** one `MjData` per bout, 32–64 bouts per worker process stepped in turn (MuJoCo steps each `MjData` on one thread), one worker per core; policy inference batched on a GPU inference server; one GPU learner. Experience flows as fixed-length segments of 128 bout-policy steps.

**GPU backend benchmark (first task of stage 0):** compile the final model with MJX-Warp, run 1k, 4k and 16k parallel bouts on one A100 or H100, record physics steps/s and memory, and verify hit events against the CPU backend on a fixed replay. Choose the backend with the lower cost per physics step. For reference, a [MuJoCo Warp issue](https://github.com/google-deepmind/mujoco_warp/issues/1733) reports 36.6 ms per step for 4,086 worlds of a 162-muscle model on an A100.

**Storage**

| Item | Size |
| --- | --- |
| Evaluation bout replay (2 × 81 bodies, position + quaternion, 25 Hz, 540 s) | 61 MB raw, about 20 MB as compressed Parquet |
| Hit log | about 100 bytes per hit |
| Bout-policy snapshot (0.8 M parameters, fp32) | 3.2 MB; 500-snapshot league about 1.6 GB |
| Skill controller + encoder + prior | about 20 MB, shared by all snapshots |

**Reproducibility:** pinned container image (MuJoCo, PyTorch, CUDA versions), submodule commits, compiled-model hash, seeds per worker; CPU MuJoCo is deterministic for a fixed build, so any bout replays from its seed and logged actions.

## Telemetry, evaluation and replay

Training logs every hit and every episode summary; evaluation bouts also log full body state at 25 Hz so any bout can be replayed, re-scored and turned into fight statistics.

**Hit record** (one Parquet row per hit, written by `telemetry.logger`):

| Field | Type | Notes |
| --- | --- | --- |
| bout\_id, t\_start, t\_end | string, float s, float s |  |
| attacker, glove | A/B, lead/rear + r/l | Lead and rear from the attacker's stance |
| punch\_type | straight / hook / uppercut | Glove velocity at onset in the attacker's heading frame: forward share > 0.7 straight, else the larger of lateral and vertical |
| target\_shell, zone | string | Section 7 |
| landing\_point | float\[3\] | Shell frame |
| closing\_speed | float m/s |  |
| impulse, peak\_force | float N·s, float N |  |
| head\_lin\_peak, head\_ang\_peak | float g, float rad/s² | Defender's skull\_head |
| fist\_side, blocked, foul | bool |  |
| points, damage\_added | float |  |

**Thrown punches** are detected separately so accuracy can be computed: a glove speed peak above 3 m/s, moving away from the boxer's own chest, with the elbow extending, segmented with the same 100 ms refractory period. Accuracy = landed scoring hits / thrown punches, per punch type.

**Per-bout statistics:** punches thrown and landed by type and hand, accuracy, mean impulse and peak head acceleration delivered and received, time at each distance band (out of range, long, mid, clinch), time within 1 m of the ropes, knockdowns, fouls, slips, mean muscle activation and fatigue by region, result.

**Evaluation suite** (runs on every promoted snapshot):

1. **Strength:** 200 bouts against each member of a frozen evaluation pool (scripted styles plus a fixed ladder of past snapshots); Elo with K = 16 and a fixed anchor.
2. **Realism:** distributions of glove speed by punch type, punch rate per round, accuracy, impulse and head acceleration, compared with published boxing data (section 9 targets).
3. **Naturalness:** a motion discriminator trained on the boxing mocap scores 1-second windows of key-body trajectories; report the mean score and the share of windows below a threshold.
4. **Stability:** falls, slips and unstable-simulation truncations per 100 rounds; peak penetration depth at impacts.

**Replay:** `telemetry.replay` rebuilds the model from the logged config hash, sets qpos from the log and renders with MuJoCo's offscreen renderer (EGL on headless machines), with hit markers and zone labels overlaid; a bout can also be re-simulated from its seed and logged actions to re-score it under new scoring rules.

## Test plan

Three layers: fast unit tests on every commit, physics regression against golden values on every model change, and training smoke tests nightly.

| Layer | Test | Pass condition |
| --- | --- | --- |
| Unit: model | Compile the bout | nq 178, nv 176, nu 1,016; all names unique |
| Unit: model | Mirror map | All 700 stock muscles and all joints paired; mirrored pose gives mirrored moment arms to 1e-6 |
| Unit: model | Pruning is lossless | Random states: generalised forces equal with and without the 192 removed muscles, to 1e-9 |
| Unit: model | Mass bookkeeping | Each boxer 60.22 kg + 2 gloves (0.566 kg), to 0.01 kg |
| Unit: model | Collision matrix | A–B shells collide; A–A only through the listed own-body pairs; both collide with floor and ropes |
| Unit: model | Shell frames | Each shell's axes are forward/left/up at the reference pose for both corners |
| Unit: scoring | Synthetic contacts | Each zone rule, fist-side check, closing-speed threshold, gap merge, 80 ms cap, 100 ms refractory, fouls |
| Unit: rules | Knockdown attribution | Fall within 1.0 s of a scoring hit is credited; otherwise a slip |
| Unit: env | Shapes and flags | Observation sizes 2,194 / 346 / 276; time-out gives `trunc` only |
| Unit: env | Corner symmetry | Swapping corners and mirroring gives identical observations |
| Unit: env | Determinism | Same seed and actions give identical trajectories on CPU |
| Physics | Impact rig golden values | Every section-5 row within 5% |
| Physics | Side-lying stability | 2 s lying on the side, any orientation: no warnings, finite state |
| Physics | Open-loop jab | Peak glove speed within 5% of the golden value |
| Physics | Sensors vs contact loop | Contact sensors and the Python tracker emit the same hits on 100 recorded bouts |
| Physics | CPU vs GPU backend | Same hits on a fixed replay; observations within 1e-4 |
| Smoke | Stage-1 PPO, 10^6 steps | Mean return rises; no NaN |
| Smoke | Two-snapshot league, 30 min | Runs without errors; Elo table updates |
| Smoke | Throughput | Steps/s within 10% of the recorded baseline |

The prototype in `ms-boxing.zip` predates this spec: its hit tracker has no gap merge or 80 ms cap, uses J\_ref = 10 N·s, and the model has no freejoint, head split or pruning. Port it under these tests rather than reusing it as is.

## Milestones

With three engineers the plan runs about nine months, November 2026 to July 2027, and each stage starts only after the previous gate (section 9) passes.

&#91;embedded content: proposed schedule · assumes three engineers and the lean compute plan; dates are estimates\]

Staffing assumed: one engineer on model, physics and calibration then realism; one on skills (mocap, retargeting, stages 1–2); one on environment, infrastructure and the league (stages 3–4). The longest pole is stage 4; the biggest schedule risk is gate 0 (head rotation) and gate 2 (punch speed).

## Risks and open questions

The two technical risks most likely to cost months are head rotation that stays below target after calibration and punch speed that stays below 7 m/s after the muscle fix; both have fallbacks that keep the project moving.

| Risk | Early signal | Mitigation |
| --- | --- | --- |
| Head angular acceleration stays about 2× low | Gate 0 misses the ±25% band after the CMA-ES search | Articulate the cervical spine into 2–3 segments; meanwhile score head shots on impulse only and scale damage per zone |
| Punches stay slow | Stage-2 straight punch under 7 m/s | Raise vmax toward the fallback, add series-elastic tendons through a custom actuator, unfreeze the skill controller in stage 5 |
| Too little boxing mocap | Fewer than 20 minutes of usable clips per punch type | Mirror and time-warp clips; add target-hitting rewards so RL fills gaps without mocap |
| Retargeting errors (scapula coupled to shoulder elevation, no fingers) | High IK residuals at the shoulders | Joint-limit-aware IK, per-joint tracking tolerances, drop the worst frames |
| GPU backend slower than CPU | Stage-0 benchmark | CPU cluster: the lean plan is about a day of physics on 256 cores |
| Stand-off or clinch equilibria | Punch rate falls below 10 per minute in league bouts | Idle penalty, exploiters rewarded for activity, forced break after 2 s of clinch |
| Solver exploits | Hits with penetration over 2 cm, or scores that collapse when contact is randomised | Contact randomisation, penetration monitor in the evaluation suite |
| Upstream changes | New commits to the two repos | Pinned submodules; upgrades only through the physics regression suite |

Both upstream repos are Apache-2.0, so derived models and code can be shared; mocap licences are checked per dataset in section 9.

Open questions for the project owner:

- **Rules:** amateur (Olympic) or professional rules and judging? This changes round count, scoring and how much defence is rewarded.
- **Athlete:** the model is one 60.2 kg body. Is one weight class enough, or are mixed-weight bouts needed?
- **Priority:** human-like realism or competitive strength? The weight on the naturalness penalty in stage 5 depends on it.
- **Resources:** team size and compute budget (section 14 assumes three engineers).
- **Source models:** which OpenSim models MS-Human-700 was assembled from, needed for the per-muscle vmax table. The arm muscle names match the MoBL-ARMS upper-limb model; the authors should confirm the rest.

## Sources

Measurements in sections 4, 5 and 11 were run for this spec in MuJoCo 3.17 on the pinned commits; everything else is cited below.

**Model and code**

- [LNSGroup/ms-human-700](https://github.com/LNSGroup/ms-human-700) and [LNSGroup/msgym](https://github.com/LNSGroup/msgym)
- [MS-Human-700 paper](https://arxiv.org/abs/2312.05473), [DynSyn](https://arxiv.org/abs/2407.11472), [QFlex](https://arxiv.org/abs/2601.19707), [MS-Emulator project page](https://lnsgroup.cc/research/MS-Emulator/index.html)
- [MuJoCo MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html); [MuJoCo Warp issue 1733](https://github.com/google-deepmind/mujoco_warp/issues/1733)
- [OpenSim `Muscle.cpp`](https://github.com/opensim-org/opensim-core/blob/main/OpenSim/Simulation/Model/Muscle.cpp): default maximum contraction velocity of 10 optimal fibre lengths per second
- [MyoSuite fatigue validation](https://myosuite.readthedocs.io/en/latest/fatigue_validation.html)
- [mink](https://github.com/kevinzakka/mink), [GMR](https://github.com/YanjieZe/GMR), [MuscleMimic](https://github.com/amathislab/musclemimic)

**Learning methods**

- [Won, Gopinath and Hodgins 2021](https://research.facebook.com/file/559394838468486/Control-Strategies-for-Physically-Simulated-Characters-Performing-Two-player-Competitive-Sports.pdf), two-player boxing and fencing
- [Vinyals et al. 2019, AlphaStar](https://www.seas.upenn.edu/~cis520/papers/RL_for_starcraft.pdf), league training and PFSP
- [Bansal et al. 2018](https://ar5iv.labs.arxiv.org/html/1710.03748), opponent sampling and reward annealing

**Boxing biomechanics**

- [Walilko et al. 2005](https://pubmed.ncbi.nlm.nih.gov/16183766/)
- [Liu et al. 2022](https://frontiersin.org/articles/10.3389/fphys.2022.1015154/full)
- [Dinu and Louis 2020](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.598861/pdf)
- [Piorkowski et al. 2011](https://knowledge.lancashire.ac.uk/id/eprint/3757/1/piorkowski-et-al-2011-punch-kinematics.pdf)
- [Nakano, Iino and Kojima 2009](https://ojs.ub.uni-konstanz.de/cpa/article/view/3427/3226)

**Motion data and licences**

- [CMU Graphics Lab mocap](http://mocap.cs.cmu.edu/) and [trial index](https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/cmu-mocap-index-text.txt)
- [ACCAD motion lab data](https://accad.osu.edu/research/motion-lab/mocap-system-and-data), [HDM05 cuts](https://resources.mpi-inf.mpg.de/HDM05/cuts/index.html), [Kyokushin strike dataset](https://www.nature.com/articles/s41597-021-00801-5)
- [AMASS licence](https://amass.is.tue.mpg.de/license.html), [Mixamo additional terms](https://wwwimages2.adobe.com/content/dam/cc/en/legal/servicetou/Mixamo-Addl-Terms-en_US-20210623.pdf)
- [BoxingVI](https://arxiv.org/html/2511.16524v1), [BoxingWeb](https://github.com/gouba2333/BoxingWeb)
